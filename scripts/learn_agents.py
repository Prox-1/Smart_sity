from threading import local
from utils import q_learning, sumo_utils
import traci
import os
import sys
from tqdm import tqdm

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
sumoConfig = r"C:\Program Files (x86)\Eclipse\Sumo\tools\2025-09-20-14-52-18\osm.sumocfg"
sumoCmd = [sumoBinary, "-c", sumoConfig]
sumoCmd.append("--no-warnings")
sumoCmd.extend(["--verbose", "false"])
script_dir = os.path.dirname(os.path.abspath(__file__))
output_base_dir = os.path.join(
    script_dir, "..", "agents", "total_reward_lr01_df099_epd0999_every10s")
os.makedirs(output_base_dir, exist_ok=True)

actions = [+10, 0, -10]

# --- Параметры обучения Q-learning ---
NUM_EPISODES = 100      # Количество эпизодов обучения
# Максимальное количество шагов симуляции в одном эпизоде (например, 1 час)
MAX_SIMULATION_STEPS = 7200
DECISION_INTERVAL = 10    # Агент принимает решение каждые N секунд симуляции
# Запуск SUMO
print("Starting SUMO simulation and data extraction...")

agents = {}

try:
    traci.start(sumoCmd)
    tls_ids = traci.trafficlight.getIDList()
    controlled_edges_dict = {}
    count_of_all_edges = 0
    for tls_id in tls_ids:
        controlled_edges = sumo_utils.get_tls_controlled_edges(tls_id)
        states = q_learning.create_state_table(
            tls_id, controlled_edges)
        agents[tls_id] = q_learning.QLearningAgent(tls_id=tls_id,
                                                   states=states,
                                                   actions=actions,
                                                   learning_rate=0.1,
                                                   discount_factor=0.99,
                                                   epsilon=1.0,
                                                   epsilon_decay=0.999,
                                                   min_epsilon=0.01)
        controlled_edges_dict[tls_id] = controlled_edges
        count_of_all_edges += len(controlled_edges)
    traci.close()
    for episode in tqdm(range(NUM_EPISODES)):
        traci.start(sumoCmd)
        total_reward_episode = {tls_id: 0.0 for tls_id in tls_ids}
        last_states = {tls_id: None for tls_id in tls_ids}
        last_actions = {tls_id: None for tls_id in tls_ids}

        for tls_id in tls_ids:
            last_states[tls_id] = q_learning.create_state_for_tls(
                tls_id, controlled_edges_dict[tls_id])

        for current_step in tqdm(range(MAX_SIMULATION_STEPS)):
            traci.simulationStep()
            if traci.simulation.getMinExpectedNumber() == 0 and current_step >= 0:
                continue
            current_time = traci.simulation.getTime()
            global_reward = q_learning.calculate_global_reward(
                tls_ids, controlled_edges_dict, count_of_all_edges)
            for tls_id in tls_ids:
                local_reward = q_learning.calculate_local_reward(
                    controlled_edges_dict[tls_id], count_of_all_edges)
                total_reward = q_learning.calculate_total_reward(
                    local_reward, global_reward)
                total_reward_episode[tls_id] += total_reward
                current_state = q_learning.create_state_for_tls(
                    tls_id, controlled_edges_dict[tls_id])
                if last_states[tls_id] is not None and last_actions[tls_id] is not None:
                    agents[tls_id].update_q_table(
                        last_states[tls_id],
                        last_actions[tls_id],
                        total_reward,
                        current_state
                    )
                if int(current_time) % DECISION_INTERVAL == 0:
                    chosen_action_value = agents[tls_id].choose_action(
                        current_state)
                    sumo_utils.set_phase_duration_by_action(
                        tls_id, chosen_action_value)
                    last_states[tls_id] = current_state
                    last_actions[tls_id] = chosen_action_value
        for tls_id in tls_ids:

            agents[tls_id].decay_epsilon()
            agents[tls_id].save_q_table(os.path.join(
                output_base_dir, f"q_table_{tls_id}.npy"))
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
    print("Q-learning process finished.")
