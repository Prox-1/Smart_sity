import os
import sys
import csv
import random
import numpy as np
import traci
from tqdm import tqdm
from pathlib import Path
from utils.test_utils import *

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
# путь для
PROJECT_DIR = Path(__file__).resolve().parent.parent
relative_cfg = Path("sumo_config") / "2025-09-20-14-52-18" / "osm.sumocfg"
candidate_cfg = (PROJECT_DIR / relative_cfg).resolve()
sumoConfig = str(candidate_cfg)
sumoCmd = [sumoBinary, "-c", sumoConfig, "--seed", "42"]
sumoCmd.append("--no-warnings")
sumoCmd.extend(["--verbose", "false"])

# Выход
OUTPUT_DIR = "metrics/without_agents"
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

try:
    traci.start(sumoCmd)
    # TLS -> контролируемые полосы (без дубликатов)
    tls_ids = list(traci.trafficlight.getIDList())
    tls_to_lanes = {}
    for tls_id in tls_ids:
        lanes = dedup(traci.trafficlight.getControlledLanes(tls_id))
        tls_to_lanes[tls_id] = lanes
    # Полосы всей сети для сетевых метрик
    all_lanes = list(traci.lane.getIDList())
    step = 0

    for step in tqdm(range(MAX_SIMULATION_STEPS)):
        traci.simulationStep()
        sim_time = traci.simulation.getTime()

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
