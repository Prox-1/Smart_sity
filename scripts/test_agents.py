from utils import q_learning, sumo_utils
import traci
import os
import sys

# Установка служебного имени SUMO_HOME
if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = r"C:\Program Files (x86)\Eclipse\Sumo"
if 'SUMO_HOME' in os.environ:
    tools_path = os.path.join(os.environ['SUMO_HOME'], 'tools')
    if tools_path not in sys.path:
        sys.path.append(tools_path)
else:
    sys.exit("Environment variable 'SUMO_HOME' is not set. Please set it to your SUMO installation directory.")

sumoBinary = "sumo-gui"  # sumo-gui
sumoConfig = r"C:\Program Files (x86)\Eclipse\Sumo\tools\2025-09-20-14-52-18\osm.sumocfg"
sumoCmd = [sumoBinary, "-c", sumoConfig]

actions = [+5, 0, -5]

# --- Параметры обучения Q-learning ---
NUM_EPISODES = 50        # Количество эпизодов обучения
# Максимальное количество шагов симуляции в одном эпизоде (например, 1 час)
MAX_SIMULATION_STEPS = 3600
DECISION_INTERVAL = 5     # Агент принимает решение каждые N секунд симуляции
# Запуск SUMO
print("Starting SUMO simulation and data extraction...")

current_script_dir = os.path.dirname(os.path.abspath(__file__))
agents_folder_path = os.path.join(
    current_script_dir, '..', 'agents', 'every_5s_learning')

agents = {}

try:
    traci.start(sumoCmd)
    tls_ids = traci.trafficlight.getIDList()
    for tls_id in tls_ids:
        states, controlled_edges = q_learning.create_state_table(
            tls_id, True)
        agents[tls_id] = q_learning.QLearningAgent(tls_id=tls_id,
                                                   states=states,
                                                   actions=actions,
                                                   learning_rate=0.1,
                                                   discount_factor=0.9,
                                                   epsilon=1.0,
                                                   epsilon_decay=0.995,
                                                   min_epsilon=0.01)

        agents[tls_id].load_q_table(
            os.path.join(agents_folder_path, f"q_table_{tls_id}.npy"))

    current_step = 0
    while current_step < MAX_SIMULATION_STEPS:
        traci.simulationStep()
        current_time = traci.simulation.getTime()
        for tls_id in tls_ids:
            controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
            controlled_edges = set()
            for lane_id in controlled_lanes:
                controlled_edges.add(traci.lane.getEdgeID(lane_id))
            current_state = q_learning.create_state_for_tls(
                tls_id, controlled_edges)
            if int(current_time) % DECISION_INTERVAL == 0:
                chosen_action_value = agents[tls_id].choose_action(
                    current_state)
                sumo_utils.set_phase_duration_by_action(
                    tls_id, chosen_action_value)
        current_step += 1
        if traci.simulation.getMinExpectedNumber() == 0 and current_step > 1:
            print(
                f"Simulation ended early at step {current_step} due to no more vehicles.")
            break
    traci.close()

except traci.exceptions.TraCIException as e:
    print(f"TraCI error: {e}")
finally:
    try:
        traci.close()
    except traci.exceptions.FatalTraCIError:
        pass
    except Exception as e:
        print(f"Error closing TraCI connection: {e}")
    print("Test process finished.")
