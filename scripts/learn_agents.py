from threading import local
from utils import q_learning, sumo_utils
from utils.accident_utils import AccidentManager
import random
import traci
import os
import sys
from tqdm import tqdm
from pathlib import Path

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
sumoCmd = [sumoBinary, "-c", sumoConfig]
sumoCmd.append("--no-warnings")
sumoCmd.extend(["--verbose", "false"])
script_dir = os.path.dirname(os.path.abspath(__file__))
output_base_dir = os.path.join(
    script_dir, "..", "agents", "total_reward_lr01_df099_epd0999_with_accident_in_reward")
os.makedirs(output_base_dir, exist_ok=True)

actions = [+20, +10, 0, -10, -20]

# --- Параметры аварий ---

ENABLE_ACCIDENTS = True
ACCIDENT_MODE = "obstacle"  # "lane_block" или "obstacle"
ACCIDENT_PROB_PER_STEP = 0.005  # вероятность за шаг (на всю сеть)
ACCIDENT_MIN_DURATION = 100     # шаги
ACCIDENT_MAX_DURATION = 300     # шаги
ACCIDENT_MAX_CONCURRENT = 10     # одновременно активных аварий

# --- Параметры обучения Q-learning ---
NUM_EPISODES = 100      # Количество эпизодов обучения
# Максимальное количество шагов симуляции в одном эпизоде (например, 1 час)
MAX_SIMULATION_STEPS = 7200
# Запуск SUMO
print("Starting SUMO simulation and data extraction...")

agents = {}

try:
    traci.start(sumoCmd)
    tls_ids = traci.trafficlight.getIDList()
    controlled_edges_dict = {}
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
    unique_edges_count = len(set().union(*controlled_edges_dict.values()))

    traci.close()
    for episode in tqdm(range(NUM_EPISODES)):
        traci.start(sumoCmd)

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
            rng = random.Random()  # воспроизводимо
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

        total_reward_episode = {tls_id: 0.0 for tls_id in tls_ids}
        last_states = {tls_id: None for tls_id in tls_ids}
        last_actions = {tls_id: None for tls_id in tls_ids}

        for tls_id in tls_ids:
            last_states[tls_id] = q_learning.create_state_for_tls(
                tls_id, controlled_edges_dict[tls_id])

        for current_step in range(MAX_SIMULATION_STEPS):
            last_phase_idx = {tls_id: traci.trafficlight.getPhase(
                tls_id) for tls_id in tls_ids}
            traci.simulationStep()
            # === ТИК МЕНЕДЖЕРА АВАРИЙ ===
            if ENABLE_ACCIDENTS and accident_manager is not None:
                accident_manager.step(current_step)
            if traci.simulation.getMinExpectedNumber() == 0 and current_step >= 0:
                continue
            current_time = traci.simulation.getTime()
            global_reward = q_learning.calculate_global_reward(
                tls_ids, controlled_edges_dict, unique_edges_count)
            for tls_id in tls_ids:
                local_reward = q_learning.calculate_local_reward(
                    controlled_edges_dict[tls_id],
                    use_accident_penalty=True,
                    accident_weight=0.35,
                    accident_provider=lambda edges: accident_manager.get_edge_impacts(
                        edges),
                )
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
                cur_phase = traci.trafficlight.getPhase(tls_id)
                if cur_phase != last_phase_idx[tls_id]:
                    chosen_action_value = agents[tls_id].choose_action(
                        current_state)
                    sumo_utils.set_phase_duration_for_new_phase(
                        tls_id, chosen_action_value)
                    last_states[tls_id] = current_state
                    last_actions[tls_id] = chosen_action_value
        for tls_id in tls_ids:

            agents[tls_id].decay_epsilon()
            agents[tls_id].save_q_table(os.path.join(
                output_base_dir, f"q_table_{tls_id}.npy"))
        try:
            if ENABLE_ACCIDENTS and accident_manager is not None:
                accident_manager.shutdown()
        except Exception:
            pass
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
