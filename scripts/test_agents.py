from utils.test_utils import *
from tqdm import tqdm
import csv
from utils import q_learning, sumo_utils
import libsumo as traci
import os
import random
import sys
import numpy as np
from pathlib import Path
from utils.accident_utils import AccidentManager

USING_LIBSUMO = True
# ----------------- Параметры -----------------
STEP_INTERVAL = 10             # собирать метрики каждые 10 шагов
MAX_SIMULATION_STEPS = 3600

# SUMO
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
sumoBinary = "sumo"  # при необходимости укажите полный путь к sumo-gui.exe
PROJECT_DIR = Path(__file__).resolve().parent.parent
relative_cfg = Path("sumo_config") / "2025-09-20-14-52-18" / "osm.sumocfg"
candidate_cfg = (PROJECT_DIR / relative_cfg).resolve()
sumoConfig = str(candidate_cfg)
sumoCmd = [sumoBinary, "-c", sumoConfig, "--seed", "42"]
sumoCmd.append("--no-warnings")
sumoCmd.extend(["--verbose", "false"])
current_script_dir = os.path.dirname(os.path.abspath(__file__))
agents_folder_path = os.path.join(
    current_script_dir, '..', 'agents', 'total_reward_lr01_df099_epd0999_30_20_10_0_100eps_7200steps(l_reward_ 1.5 1 0.5 g_reward_ 1 0.7 0.3)')


# Выход
OUTPUT_DIR = "metrics/total_reward_lr01_df099_epd0999_30_20_10_0_100eps_7200steps(l_reward_ 1.5 1 0.5 g_reward_ 1 0.7 0.3)"
os.makedirs(OUTPUT_DIR, exist_ok=True)
network_csv_path = os.path.join(OUTPUT_DIR, "network_metrics.csv")
tls_csv_path = os.path.join(OUTPUT_DIR, "tls_metrics.csv")
network_fields = [
    "step", "time",
    "active_vehicles",
    "mean_speed_network",
    "total_queue_len",
    "total_waiting_time_snapshot"
]

tls_fields = [
    "step", "time", "tls_id", "phase_index",
    "tls_queue_len",
    "tls_waiting_time_snapshot",
    "tls_mean_speed"
]

print("Starting SUMO simulation and metrics sampling every",
      STEP_INTERVAL, "steps...")

network_f, network_writer = write_csv_header(network_csv_path, network_fields)
tls_f, tls_writer = write_csv_header(tls_csv_path, tls_fields)

actions = [+20, +10, 0, -10, -20]

# --- Параметры аварий ---

ENABLE_ACCIDENTS = True
ACCIDENT_MODE = "obstacle"  # "lane_block" или "obstacle"
ACCIDENT_PROB_PER_STEP = 0.005  # вероятность за шаг (на всю сеть)
ACCIDENT_MIN_DURATION = 100     # шаги
ACCIDENT_MAX_DURATION = 300     # шаги
ACCIDENT_MAX_CONCURRENT = 10     # одновременно активных аварий


agents = {}
controlled_edges_dict = {}

try:
    traci.start(sumoCmd)
    # TLS -> контролируемые полосы (без дубликатов)
    tls_ids = list(traci.trafficlight.getIDList())
    tls_to_lanes = {}
    for tls_id in tls_ids:
        lanes = dedup(traci.trafficlight.getControlledLanes(tls_id))
        tls_to_lanes[tls_id] = lanes
        controlled_edges_dict[tls_id] = sumo_utils.get_tls_controlled_edges(
            tls_id)
        states = q_learning.create_state_table(
            tls_id, controlled_edges_dict[tls_id])
        agents[tls_id] = q_learning.QLearningAgent(tls_id=tls_id,
                                                   states=states,
                                                   actions=actions,
                                                   learning_rate=0.00,
                                                   discount_factor=0.8,
                                                   epsilon=0.00,
                                                   epsilon_decay=1,
                                                   min_epsilon=0.00)
        agents[tls_id].load_q_table(
            os.path.join(agents_folder_path, f"q_table_{tls_id}.npy"))

    # Полосы всей сети для сетевых метрик
    all_lanes = list(traci.lane.getIDList())
    step = 0
    if ENABLE_ACCIDENTS:
        # Все полосы в сети (кроме внутренних ":")
        all_lanes = list(traci.lane.getIDList())
        # Используемые классы ТС — чтобы корректно закрывать полосу только для реально существующих классов
        try:
            vtypes = traci.vehicletype.getIDList()
            used_vclasses = set(traci.vehicletype.getVehicleClass(t)
                                for t in vtypes)
        except Exception:
            used_vclasses = set()
        rng = random.Random(42)  # воспроизводимо
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
    for step in tqdm(range(MAX_SIMULATION_STEPS)):
        last_phase_idx = {tls_id: traci.trafficlight.getPhase(
            tls_id) for tls_id in tls_ids}
        traci.simulationStep()
        sim_time = traci.simulation.getTime()

        # === ТИК МЕНЕДЖЕРА АВАРИЙ ===
        if ENABLE_ACCIDENTS and accident_manager is not None:
            accident_manager.step(step)

        # События шага
        departed_ids = traci.simulation.getDepartedIDList()
        arrived_ids = traci.simulation.getArrivedIDList()
        # Сбор метрик только на выборочных шагах

        if step % STEP_INTERVAL == 0:
            veh_ids = traci.vehicle.getIDList()
            active_vehicles = len(veh_ids)

            # Средняя скорость по всем ТС (на моменте выборки)
            mean_speed_network = 0.0
            if active_vehicles > 0:
                sum_speed = 0.0
                for vid in veh_ids:
                    sum_speed += traci.vehicle.getSpeed(vid)
                mean_speed_network = sum_speed / active_vehicles

            # Очередь и ожидание по сети (снимок)
            total_queue_len = 0
            total_waiting_time_snapshot = 0.0
            for lid in all_lanes:
                total_queue_len += traci.lane.getLastStepHaltingNumber(lid)
                total_waiting_time_snapshot += traci.lane.getWaitingTime(lid)

            # Запись сетевых метрик
            network_writer.writerow({
                "step": step,
                "time": sim_time,
                "active_vehicles": active_vehicles,
                "mean_speed_network": mean_speed_network,
                "total_queue_len": total_queue_len,
                "total_waiting_time_snapshot": total_waiting_time_snapshot
            })
            # Метрики по каждому светофору

            for tls_id in tls_ids:
                current_state = q_learning.create_state_for_tls(
                    tls_id, controlled_edges_dict[tls_id])
                cur_phase = traci.trafficlight.getPhase(tls_id)
                if cur_phase != last_phase_idx[tls_id]:
                    chosen_action_value = agents[tls_id].choose_action(
                        current_state)
                    sumo_utils.set_phase_duration_by_action(
                        tls_id, chosen_action_value)
                lanes = tls_to_lanes[tls_id]
                phase_index = traci.trafficlight.getPhase(tls_id)
                tls_queue_len = sum_halting_on_lanes(lanes)
                tls_waiting = sum_waiting_time_on_lanes(lanes)
                tls_mean_speed = weighted_mean_speed_on_lanes(lanes)
                tls_writer.writerow({
                    "step": step,
                    "time": sim_time,
                    "tls_id": tls_id,
                    "phase_index": phase_index,
                    "tls_queue_len": tls_queue_len,
                    "tls_waiting_time_snapshot": tls_waiting,
                    "tls_mean_speed": tls_mean_speed
                })
        # Раннее завершение, если трафика больше нет
        if traci.simulation.getMinExpectedNumber() == 0 and step > 1:
            print(
                f"Simulation ended early at step {step} due to no more vehicles.")
            break
    traci.close()

except traci.exceptions.TraCIException as e:
    print(f"TraCI error: {e}")
finally:
    try:
        traci.close()
    except:
        pass
    network_f.close()
    tls_f.close()
    print("Sampling finished. CSV saved to:", OUTPUT_DIR)
