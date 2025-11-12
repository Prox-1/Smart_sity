import sumolib as traci
from threading import Thread
from flask import Flask, request, jsonify
from typing import Optional
from dataclasses import dataclass
import queue
from utils_traci.test_utils import *
from utils_traci.accident_utils import AccidentManager
from tqdm import tqdm
import csv
from utils_traci.sumo_utils import get_tls_controlled_edges
from utils_traci.q_learning import QLearningAgent, create_state_table
import os
import random
import sys
import numpy as np
from pathlib import Path

# ----------------- Параметры -----------------
MAX_SIMULATION_STEPS = 10000

# SUMO

# Зафиксируем хеш-семя для воспроизводимости поведения hash()
os.environ["PYTHONHASHSEED"] = "0"
random.seed(42)
np.random.seed(42)

if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = r"C:\Program Files (x86)\Eclipse\Sumo"

if 'SUMO_HOME' in os.environ:
    tools_path = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools_path not in sys.path:
        sys.path.append(tools_path)
else:
    sys.exit("Environment variable 'SUMO_HOME' is not set.")

# Бинарник SUMO (можно поменять на полный путь)
sumoBinary = "sumo-gui"  # при необходимости укажите полный путь к sumo-gui.exe

PROJECT_DIR = Path(__file__).resolve().parent.parent
relative_cfg = Path("sumo_config") / "2025-09-20-14-52-18" / "osm.sumocfg"
candidate_cfg = (PROJECT_DIR / relative_cfg).resolve()
sumoConfig = str(candidate_cfg)

sumoCmd = [sumoBinary, "-c", sumoConfig, "--seed",
           "42", "--no-warnings", "--verbose", "false"]

current_script_dir = os.path.dirname(os.path.abspath(__file__))
agents_folder_path = os.path.join(
    current_script_dir, '..', 'agents', 'total_reward_lr01_df099_epd0999_30_20_10_0_100eps_7200steps(l_reward_ 1.5 1.2 0.7 g_reward_ 1 1.0 0.5)')

# Разрешённые действия для агента управления светофорами (смещение фазы)
actions = [+30, +20, +10, 0, -10, -20, -30]

# --- Параметры аварий ---

ENABLE_ACCIDENTS = True
ACCIDENT_MODE = "obstacle"  # "lane_block" или "obstacle"
ACCIDENT_PROB_PER_STEP = 0.0  # вероятность за шаг (на всю сеть)
ACCIDENT_MIN_DURATION = 100     # шаги
ACCIDENT_MAX_DURATION = 300     # шаги
ACCIDENT_MAX_CONCURRENT = 3     # одновременно активных аварий

# --- HTTP ---

# start_sim_gui.py

def start_http_api(command_queue, host="127.0.0.1", port=8081):
    """
    Запускает HTTP API в отдельном демоническом потоке.

    API эндпоинты:
    - POST /api/spawn_lane
        Тело JSON: { "lane_id": str, "pos_m": float, "duration_steps": int, "mode": str, "ignore_max_concurrent": bool }
        Добавляет команду SpawnCmd в очередь команд для создания аварии по идентификатору полосы.
    - POST /api/spawn_geo
        Тело JSON: { "lon": float, "lat": float, "duration_steps": int, "mode": str }
        Конвертирует геокоординаты в полосу и добавляет команду SpawnCmd в очередь.
    - POST /api/clear_lane
        Тело JSON: { "lane_id": str }
        Добавляет команду ClearCmd для очистки аварии на конкретной полосе.
    - POST /api/clear_all
        Без тела. Добавляет команду ClearCmd с lane_id=None для очистки всех аварий.
    - GET /api/health
        Возвращает {"ok": True} — проверка работоспособности сервера.

    Параметры:
    - command_queue: очередь (queue.Queue) для передачи команд основному циклу симуляции.
    - host: адрес хоста для Flask.
    - port: порт для Flask.

    Сервер запускается в daemon-потоке, чтобы завершаться вместе с основным процессом.
    """
    app = Flask(__name__)

    @app.route("/api/spawn_lane", methods=["POST"])
    def api_spawn_lane():
        """
        Обработчик /api/spawn_lane — принимает JSON с параметрами аварии на конкретной полосе
        и кладёт SpawnCmd в очередь команд.
        """
        data = request.get_json(force=True)
        lane_id = data.get("lane_id")
        pos_m = data.get("pos_m")
        duration_steps = data.get("duration_steps")
        mode = data.get("mode")
        ignore_max_concurrent = bool(data.get("ignore_max_concurrent", False))

        if not lane_id:
            return jsonify({"ok": False, "error": "lane_id required"}), 400

        command_queue.put(SpawnCmd(lane_id=lane_id, pos_m=pos_m, duration_steps=duration_steps,
                          mode=mode, ignore_max_concurrent=ignore_max_concurrent))
        return jsonify({"ok": True})

    @app.route("/api/spawn_geo", methods=["POST"])
    def api_spawn_geo():
        """
        Обработчик /api/spawn_geo — принимает JSON с геокоординатами (lon, lat),
        преобразует их в полосу внутри process_commands и кладёт SpawnCmd в очередь.
        """
        data = request.get_json(force=True)
        lon = data.get("lon")
        lat = data.get("lat")
        duration_steps = data.get("duration_steps")
        mode = data.get("mode")

        if lon is None or lat is None:
            return jsonify({"ok": False, "error": "lon and lat required"}), 400

        command_queue.put(SpawnCmd(lon=float(lon), lat=float(
            lat), duration_steps=duration_steps, mode=mode))
        return jsonify({"ok": True})

    @app.route("/api/clear_lane", methods=["POST"])
    def api_clear_lane():
        """
        Обработчик /api/clear_lane — принимает JSON с lane_id и кладёт ClearCmd для очистки аварии на данной полосе.
        """
        data = request.get_json(force=True)
        lane_id = data.get("lane_id")
        if not lane_id:
            return jsonify({"ok": False, "error": "lane_id required"}), 400
        command_queue.put(ClearCmd(lane_id=lane_id))
        return jsonify({"ok": True})

    @app.route("/api/clear_all", methods=["POST"])
    def api_clear_all():
        """
        Обработчик /api/clear_all — удаляет все аварии (помещает ClearCmd с lane_id=None).
        """
        command_queue.put(ClearCmd(lane_id=None))
        return jsonify({"ok": True})

    @app.route("/api/health", methods=["GET"])
    def api_health():
        """Простейший health-check для API."""
        return jsonify({"ok": True})

    # запуск в отдельном потоке
    t = Thread(target=lambda: app.run(host=host, port=port,
               debug=False, use_reloader=False, threaded=True), daemon=True)
    t.start()
    print(f"[HTTP] API started at http://{host}:{port}")

# --- Очередь ---

command_queue = queue.Queue()

@dataclass
class SpawnCmd:
    """
    Команда для создания аварии (spawn).

    Поля:
    - lon, lat: опционально — геокоординаты для конвертации в дорожную полосу.
    - lane_id: опционально — идентификатор полосы SUMO (например "edge_0_0").
    - pos_m: позиция вдоль полосы в метрах (позиция появления аварии).
    - duration_steps: сколько шагов симуляции авария будет активна (если None — берётся по умолчанию в менеджере аварий).
    - mode: режим аварии (например "obstacle" или "lane_block").
    - ignore_max_concurrent: если True — попробовать создать аварию, даже если достигнут лимит одновременно активных.
    """
    # либо geo, либо lane_id/pos
    lon: Optional[float] = None
    lat: Optional[float] = None
    lane_id: Optional[str] = None
    pos_m: Optional[float] = None
    duration_steps: Optional[int] = None
    mode: Optional[str] = None
    ignore_max_concurrent: bool = False

@dataclass
class ClearCmd:
    """
    Команда для очистки аварий.

    Поля:
    - lane_id: если указано — очистить аварию только на этой полосе.
               если None — очистить все аварии в сети.
    """
    lane_id: Optional[str] = None   # None => clear_all

def process_commands(accident_manager: AccidentManager):
    """
    Обрабатывает команды из глобальной очереди command_queue и применяет их к accident_manager.

    Алгоритм:
    - Пока в очереди есть команды, достаём команду без ожидания (get_nowait).
    - Для SpawnCmd:
        - Если указаны lon/lat — конвертируем геокоординаты в lane_id через traci.simulation.convertRoad.
        - Формируем lane_id и pos_m, вызываем accident_manager.create_accident_at.
        - Пишем результат в stdout.
    - Для ClearCmd:
        - Если указан lane_id — вызываем accident_manager.clear_accident(lane_id).
        - Иначе — вызываем accident_manager.clear_all().
    - Исключения ловятся и логируются в stdout, чтобы не ломать основной цикл симуляции.
    """
    while True:
        try:
            cmd = command_queue.get_nowait()
        except queue.Empty:
            break

        if isinstance(cmd, SpawnCmd):
            try:
                if cmd.lane_id is None and cmd.lon is not None and cmd.lat is not None:
                    # Конвертируем geo -> дорога
                    edge_id, lane_pos, lane_index = traci.simulation.convertRoad(
                        cmd.lon, cmd.lat, isGeo=True)
                    lane_id = f"{edge_id}_{lane_index}"
                    pos_m = float(lane_pos) if cmd.pos_m is None else cmd.pos_m
                else:
                    lane_id = cmd.lane_id
                    pos_m = cmd.pos_m

                if lane_id is None:
                    print("SpawnCmd: no lane resolved")
                    continue

                acc = accident_manager.create_accident_at(
                    lane_id=lane_id,
                    duration_steps=cmd.duration_steps,
                    pos_m=pos_m,
                    mode=cmd.mode,
                    ignore_max_concurrent=cmd.ignore_max_concurrent
                )
                if acc:
                    print(
                        f"[BOT] Accident created at {lane_id} pos={pos_m} mode={cmd.mode}")
                else:
                    print(f"[BOT] Failed to create accident at {lane_id}")
            except Exception as e:
                # Логируем ошибку, но не прерываем обработку следующих команд
                print(f"[BOT] spawn error: {e}")

        elif isinstance(cmd, ClearCmd):
            try:
                if cmd.lane_id:
                    ok = accident_manager.clear_accident(cmd.lane_id)
                    print(f"[BOT] clear {cmd.lane_id}: {ok}")
                else:
                    n = accident_manager.clear_all()
                    print(f"[BOT] clear_all: {n}")
            except Exception as e:
                print(f"[BOT] clear error: {e}")

# Запускаем HTTP API для управления авариями через HTTP
start_http_api(command_queue, host="127.0.0.1", port=8081)

agents = {}
controlled_edges_dict = {}
try:
    # Запуск TraCI (Sumo)
    traci.start(sumoCmd)

    # Получаем список всех светофоров в сети
    tls_ids = list(traci.trafficlight.getIDList())

    tls_to_lanes = {}
    for tls_id in tls_ids:
        # Получаем управляемые светофором полосы (dedup из utils.test_utils)
        lanes = dedup(traci.trafficlight.getControlledLanes(tls_id))
        tls_to_lanes[tls_id] = lanes

        # Получаем ребра (edges), контролируемые этим светофором
        controlled_edges_dict[tls_id] = get_tls_controlled_edges(
            tls_id)

        # Создаём таблицу состояний для Q-learning агента (реализовано в utils.q_learning)
        states = create_state_table(
            tls_id, controlled_edges_dict[tls_id])

        # Инициализируем QLearningAgent с нулевым learning_rate и epsilon=0
        agents[tls_id] = QLearningAgent(
            tls_id=tls_id,
            states=states,
            actions=actions,
            learning_rate=0.00,
            discount_factor=0.8,
            epsilon=0.00,
            epsilon_decay=1,
            min_epsilon=0.00
        )

        # Загружаем заранее обученную Q-таблицу из директории агентов
        agents[tls_id].load_q_table(
            os.path.join(agents_folder_path, f"q_table_{tls_id}.npy")
        )

    if ENABLE_ACCIDENTS:
        # Все полосы в сети (кроме внутренних ":"), используются для спавна аварий
        all_lanes = list(traci.lane.getIDList())

        # Используемые классы ТС — чтобы корректно закрывать полосу только для реально существующих классов
        try:
            vtypes = traci.vehicletype.getIDList()
            used_vclasses = set(traci.vehicletype.getVehicleClass(t)
                                for t in vtypes)
        except Exception:
            used_vclasses = set()

        rng = random.Random(42)  # воспроизводимо

        # Инициализируем менеджер аварий — реализован в utils.accident_utils
        accident_manager = AccidentManager(
            all_lanes,
            used_vclasses,
            rng=rng,
            mode=ACCIDENT_MODE,
            prob_per_step=ACCIDENT_PROB_PER_STEP,
            min_duration_steps=ACCIDENT_MIN_DURATION,
            max_duration_steps=ACCIDENT_MAX_DURATION,
            max_concurrent=ACCIDENT_MAX_CONCURRENT,
            min_margin_from_ends_m=10.0,
            enable_markers=True,
            marker_color=(255, 0, 0, 255),  # красный
            marker_layer=10,
            marker_size=(12, 12),
            marker_type="ACCIDENT",
            marker_label="ДТП",
        )

    # Основной цикл симуляции (итерации шагов)
    for step in tqdm(range(MAX_SIMULATION_STEPS)):
        # обработаем команды от бота (HTTP API)
        if ENABLE_ACCIDENTS and accident_manager is not None:
            process_commands(accident_manager)

        # Сохраняем текущие фазы светофоров для возможного анализа
        last_phase_idx = {tls_id: traci.trafficlight.getPhase(
            tls_id) for tls_id in tls_ids}

        # Делаем шаг симуляции в SUMO
        traci.simulationStep()
        sim_time = traci.simulation.getTime()

        # Закрытие истёкших аварий (менеджер сам управляет списком активных аварий)
        if ENABLE_ACCIDENTS and accident_manager is not None:
            accident_manager.step(step)

        # Если в сети больше нет ожидаемых транспортных средств, симуляция может завершиться раньше
        if traci.simulation.getMinExpectedNumber() == 0 and step > 1:
            print(
                f"Simulation ended early at step {step} due to no more vehicles.")
            break

    # Корректно закрываем соединение с traci
    traci.close()

except traci.exceptions.TraCIException as e:
    # Ловим и печатаем ошибки TraCI для отладки
    print(f"TraCI error: {e}")

finally:
    # В блоке finally стараемся аккуратно завершить менеджер аварий и закрыть TraCI,
    # игнорируя возможные ошибки при завершении.
    try:
        if ENABLE_ACCIDENTS and accident_manager is not None:
            accident_manager.shutdown()
    except Exception:
        pass
    try:
        traci.close()
    except:
        pass