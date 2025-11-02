from utils.test_utils import *
from utils.accident_utils import AccidentManager
from tqdm import tqdm
import csv
from utils import q_learning, sumo_utils
import traci
import os
import random
import sys
import numpy as np
from pathlib import Path

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
sumoBinary = "sumo-gui"  # при необходимости укажите полный путь к sumo-gui.exe
PROJECT_DIR = Path(__file__).resolve().parent.parent
relative_cfg = Path("sumo_config") / "2025-09-20-14-52-18" / "osm.sumocfg"
candidate_cfg = (PROJECT_DIR / relative_cfg).resolve()
sumoConfig = str(candidate_cfg)
sumoCmd = [sumoBinary, "-c", sumoConfig, "--seed",
           "42", "--no-warnings", "--verbose", "false"]
current_script_dir = os.path.dirname(os.path.abspath(__file__))
agents_folder_path = os.path.join(
    current_script_dir, '..', 'agents', 'total_reward_lr01_df099_epd0999_every10s')

print("Starting SUMO simulation and metrics sampling every",
      STEP_INTERVAL, "steps...")

actions = [+20, +10, 0, -10, -20]

# --- Параметры аварий ---

ENABLE_ACCIDENTS = True
ACCIDENT_MODE = "obstacle"  # "lane_block" или "obstacle"
ACCIDENT_PROB_PER_STEP = 0.9  # вероятность за шаг (на всю сеть)
ACCIDENT_MIN_DURATION = 100     # шаги
ACCIDENT_MAX_DURATION = 300     # шаги
ACCIDENT_MAX_CONCURRENT = 3     # одновременно активных аварий


agents = {}
controlled_edges_dict = {}

try:
    traci.start(sumoCmd)
    tls_ids = list(traci.trafficlight.getIDList())
    tls_to_lanes = {}
    for tls_id in tls_ids:
        lanes = dedup(traci.trafficlight.getControlledLanes(tls_id))
        tls_to_lanes[tls_id] = lanes
        controlled_edges_dict[tls_id] = sumo_utils.get_tls_controlled_edges(
            tls_id)
        states = q_learning.create_state_table(
            tls_id, controlled_edges_dict[tls_id])
        agents[tls_id] = q_learning.QLearningAgent(
            tls_id=tls_id,
            states=states,
            actions=actions,
            learning_rate=0.00,
            discount_factor=0.8,
            epsilon=0.00,
            epsilon_decay=1,
            min_epsilon=0.00
        )
        agents[tls_id].load_q_table(
            os.path.join(agents_folder_path, f"q_table_{tls_id}.npy")
        )
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
        # Двигаем симуляцию на 1 шаг
        traci.simulationStep()
        sim_time = traci.simulation.getTime()

        # === ТИК МЕНЕДЖЕРА АВАРИЙ ===
        if ENABLE_ACCIDENTS and accident_manager is not None:
            accident_manager.step(step)

        # Пытаемся (случайно и воспроизводимо) заспавнить новую аварию
        if traci.simulation.getMinExpectedNumber() == 0 and step > 1:
            print(
                f"Simulation ended early at step {step} due to no more vehicles.")
            break
    traci.close()

except traci.exceptions.TraCIException as e:
    print(f"TraCI error: {e}")
finally:
    try:
        if ENABLE_ACCIDENTS and accident_manager is not None:
            accident_manager.shutdown()
    except Exception:
        pass
    try:
        traci.close()
    except:
        pass
