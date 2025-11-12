import os
import sys
import random
import libsumo as traci

from tqdm import tqdm
from pathlib import Path
from utils import q_learning, sumo_utils
from utils.accident_utils import AccidentManager
from utils.metrics_cache import RewardMetricsCache, edge_from_lane, unsubscribe_all_safe

USING_LIBSUMO = True
# Установка служебного имени SUMO_HOME
if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = r"C:\Program Files (x86)\Eclipse\Sumo"

if 'SUMO_HOME' in os.environ:
    tools_path = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools_path not in sys.path:
        sys.path.append(tools_path)
else:
    sys.exit("Environment variable 'SUMO_HOME' is not set. Please set it to your SUMO installation directory.")

sumoBinary = "sumo"  # sumo-gui

PROJECT_DIR = Path(__file__).resolve().parent.parent
relative_cfg = Path("sumo_config") / "2025-09-20-14-52-18" / "osm.sumocfg"
candidate_cfg = (PROJECT_DIR / relative_cfg).resolve()
sumoConfig = str(candidate_cfg)

script_dir = os.path.dirname(os.path.abspath(__file__))
output_base_dir = os.path.join(
    script_dir, "..", "agents", "total_reward_lr01_df099_epd0999_30_20_10_0_100eps_7200steps(l_reward_ 1.5 1.2 0.7 g_reward_ 1 1.0 0.5)")
os.makedirs(output_base_dir, exist_ok=True)

sumoCmd = [sumoBinary, "-c", sumoConfig, "--no-warnings",
           "--no-step-log", "true",
           "--verbose", "false"]
# При необходимости можно управлять шагом интегрирования:
# "--step-length", "1.0",

actions = [+30, +20, +10, 0, -10, -20, -30]

# --- Параметры аварий ---
ENABLE_ACCIDENTS = False
ACCIDENT_MODE = "obstacle"  # "lane_block" или "obstacle"
ACCIDENT_PROB_PER_STEP = 0.001  # вероятность за шаг (на всю сеть)
ACCIDENT_MIN_DURATION = 100     # шаги
ACCIDENT_MAX_DURATION = 300     # шаги
ACCIDENT_MAX_CONCURRENT = 20    # одновременно активных аварий

# --- Параметры обучения Q-learning ---
NUM_EPISODES = 100      # Количество эпизодов обучения
MAX_SIMULATION_STEPS = 7200  # Макс число шагов в эпизоде (например, 1 час)

# Снимок нулевого состояния для быстрого сброса мира между эпизодами
STATE_SNAPSHOT_PATH = os.path.join(output_base_dir, "initial_state.xml")

print("Starting SUMO simulation and data extraction...")

agents = {}

try:
    # Первый и единственный запуск процесса SUMO
    traci.start(sumoCmd)

    # Инициализируем инфраструктуру один раз
    tls_ids = traci.trafficlight.getIDList()

    controlled_edges_dict = {}
    for tls_id in tls_ids:
        controlled_edges = sumo_utils.get_tls_controlled_edges(tls_id)
        states = q_learning.create_state_table(tls_id, controlled_edges)
        agents[tls_id] = q_learning.QLearningAgent(
            tls_id=tls_id,
            states=states,
            actions=actions,
            learning_rate=0.1,
            discount_factor=0.99,
            epsilon=1.0,
            epsilon_decay=0.999,
            min_epsilon=0.01
        )
        controlled_edges_dict[tls_id] = controlled_edges

    # Все полосы в сети (кроме внутренних ":"), берём один раз
    all_lanes = list(traci.lane.getIDList())

    # Используемые классы ТС — соберём один раз
    try:
        vtypes = traci.vehicletype.getIDList()
        used_vclasses = set(traci.vehicletype.getVehicleClass(t)
                            for t in vtypes)
    except Exception:
        used_vclasses = set()
    # Все релевантные рёбра — объединение тех, что под управлением светофоров
    relevant_edges = set().union(*controlled_edges_dict.values())
    # Отфильтруем полосы (исключаем внутренние ":" сразу)
    external_lanes = [l for l in all_lanes if not l.startswith(":")]
    # Инициализируем кэш (подписки навесим сразу)
    metrics_cache = RewardMetricsCache(traci, relevant_edges, external_lanes,
                                   waiting_cache_enabled=True,
                                   waiting_cache_period=5,
                                   waiting_accumulated=False,
                                   waiting_among_waiting_only=True)
    metrics_cache.subscribe_all()

    unique_edges_count = len(set().union(*controlled_edges_dict.values()))

    # Сохраняем снимок состояния на t=0 (до любых шагов)
    # Это позволит очень быстро возвращать мир в исходную точку.
    traci.simulation.saveState(STATE_SNAPSHOT_PATH)
    for episode in tqdm(range(NUM_EPISODES), desc="Episodes"):
        traci.load(sumoCmd[1:])
        traci.simulation.step()

        # Новый менеджер аварий на эпизод (лёгкий режим без маркеров для скорости)
        accident_manager = None
        if ENABLE_ACCIDENTS:
            # стабильная воспроизводимость по эпизоду
            rng = random.Random(12345 + episode)
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
                enable_markers=False,              # отключаем маркеры ради скорости
                marker_color=(255, 0, 0, 255),
                marker_layer=10,
                marker_size=(12, 12),
                marker_type="ACCIDENT",
                marker_label="ДТП",
            )

        total_reward_episode = {tls_id: 0.0 for tls_id in tls_ids}
        last_states = {tls_id: None for tls_id in tls_ids}
        last_actions = {tls_id: None for tls_id in tls_ids}

        # Инициализация начальных состояний по каждому TLS
        for tls_id in tls_ids:
            last_states[tls_id] = q_learning.create_state_for_tls(
                tls_id, controlled_edges_dict[tls_id]
            )

        # Сохраняем предыдущие фазы вне цикла шагов, чтобы не дёргать TraCI лишний раз
        prev_phase_idx = {tls_id: traci.trafficlight.getPhase(
            tls_id) for tls_id in tls_ids}

        for current_step in range(MAX_SIMULATION_STEPS):
            try:
                traci.simulationStep()
            except traci.exceptions.FatalTraCIError:
                print(f"FatalTraCIError at simulation step {current_step}")
                raise
            try:
                metrics_cache.update_from_subscriptions()
            except Exception as e:
                print(
                    f"Metrics cache update failed, attempting resubscribe: {e}")
                try:
                    metrics_cache.resubscribe()
                except Exception as e2:
                    print(f"Resubscribe failed: {e2}")
            # Тик менеджера аварий
            if ENABLE_ACCIDENTS and accident_manager is not None:
                try:
                    accident_manager.step(current_step)
                except traci.exceptions.FatalTraCIError:
                    # если мы получили fatal error — пробрасываем (это серьезно)
                    raise
                except Exception as e:
                    print(f"AccidentManager.step exception (ignored): {e}")

            # Если в сети больше не ожидается ТС — раннее завершение эпизода
            if traci.simulation.getMinExpectedNumber() == 0:
                if ENABLE_ACCIDENTS and accident_manager is not None:
                    try:
                        accident_manager.shutdown()
                    except Exception:
                        pass
                break
            # Читаем фазы один раз после шага
            cur_phase_idx = {tls_id: traci.trafficlight.getPhase(
                tls_id) for tls_id in tls_ids}

            # Глобальная награда (как раньше; при необходимости можно считать реже)
            global_reward = q_learning.calculate_global_reward(
                tls_ids,
                controlled_edges_dict,
                unique_edges_count,
                metrics=metrics_cache,
            )

            # Обновления по каждому светофору
            for tls_id in tls_ids:
                # Локальная награда: обязательно используем controlled_edges_dict[tls_id]
                local_reward = q_learning.calculate_local_reward(
                    controlled_edges_dict[tls_id],
                    metrics=metrics_cache,
                    use_accident_penalty=False,
                    accident_weight=0.35,
                    accident_provider=lambda edges: accident_manager.get_edge_impacts(
                        edges)
                    if (ENABLE_ACCIDENTS and accident_manager is not None)
                    else {}
                )

                total_reward = q_learning.calculate_total_reward(
                    local_reward, global_reward)
                total_reward_episode[tls_id] += total_reward

                # Текущее состояние
                current_state = q_learning.create_state_for_tls(
                    tls_id, controlled_edges_dict[tls_id]
                )

                # Q-обновление
                if last_states[tls_id] is not None and last_actions[tls_id] is not None:
                    agents[tls_id].update_q_table(
                        last_states[tls_id],
                        last_actions[tls_id],
                        total_reward,
                        current_state
                    )

                # Решение о действии только при смене фазы (минимум TraCI-вызовов)
                if cur_phase_idx[tls_id] != prev_phase_idx[tls_id]:
                    chosen_action_value = agents[tls_id].choose_action(
                        current_state)
                    sumo_utils.set_phase_duration_for_new_phase(
                        tls_id, chosen_action_value)
                    last_states[tls_id] = current_state
                    last_actions[tls_id] = chosen_action_value

            # Готовимся к следующему шагу: обновляем "предыдущие" фазы
            prev_phase_idx = cur_phase_idx

        # Декей эпсилона и сохранение Q-таблиц по окончанию эпизода
        for tls_id in tls_ids:
            agents[tls_id].decay_epsilon()
            agents[tls_id].save_q_table(os.path.join(
                output_base_dir, f"q_table_{tls_id}.npy"))

        # Корректное завершение менеджера аварий
        try:
            if ENABLE_ACCIDENTS and accident_manager is not None:
                accident_manager.shutdown()
        except Exception:
            pass

except traci.exceptions.TraCIException as e:
    print(f"TraCI error: {e}")

finally:
    try:
        traci.close()
    except traci.exceptions.FatalTraCIError:
        pass
    except Exception as e:
        print(f"Error closing TraCI connection: {e}")
    print("Q-learning process finished.")
