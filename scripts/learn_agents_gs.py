import os
import sys
import json
import random
import itertools
import glob
import shutil
import csv
import libsumo as traci
import test_agents
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

sumoBinary = "sumo"

PROJECT_DIR = Path(__file__).resolve().parent.parent
relative_cfg = Path("sumo_config") / "2025-09-20-14-52-18" / "osm.sumocfg"
candidate_cfg = (PROJECT_DIR / relative_cfg).resolve()
sumoConfig = str(candidate_cfg)

base_sumoCmd = [sumoBinary, "-c", sumoConfig, "--no-warnings",
           "--no-step-log", "true",
           "--verbose", "false"]

# --- Параметры обучения и симуляции по умолчанию (будут переопределяться гридом) ---
DEFAULTS = {
    # accidents
    "ENABLE_ACCIDENTS": True,
    "ACCIDENT_MODE": "obstacle",
    "ACCIDENT_PROB_PER_STEP": 0.01,
    "ACCIDENT_MIN_DURATION": 100,
    "ACCIDENT_MAX_DURATION": 300,
    "ACCIDENT_MAX_CONCURRENT": 20,
    # q-learning
    "NUM_EPISODES": 100,
    "MAX_SIMULATION_STEPS": 7200,
    "ACTIONS": [+30, +20, +10, 0, -10, -20, -30],
    "LEARNING_RATE": 0.1,
    "DISCOUNT_FACTOR": 0.99,
    "EPSILON": 1.0,
    "EPSILON_DECAY": 0.999,
    "MIN_EPSILON": 0.01,
    "USE_ACCIDENT_PENALTY": True,
    # weights
    "LOCAL_SPEED_WEIGHT": 1.5,
    "LOCAL_WTIME_WEIGHT": 1.2,
    "LOCAL_OCC_WEIGHT": 0.7,
    "GLOBAL_SPEED_WEIGHT": 1.0,
    "GLOBAL_WTIME_WEIGHT": 1.0,
    "GLOBAL_OCC_WEIGHT": 0.5,
    "WEIGHT_LOCAL": 0.5,
    "WEIGHT_GLOBAL": 0.5,
    # misc
    "FILE_NAME": "test_grid_search",
}

# --- GRID PARAMS: замените списки значениями, которые хотите перебрать ---
GRID_PARAMS = {
    "LEARNING_RATE": [0.05, 0.1, 0.2],
    "DISCOUNT_FACTOR": [0.95, 0.99],
    "EPSILON_DECAY": [0.999, 0.995],
    "WEIGHT_LOCAL": [0.3, 0.5, 0.7],
    "WEIGHT_GLOBAL": [0.7, 0.5, 0.3],
}

param_names = list(GRID_PARAMS.keys())
param_values = [GRID_PARAMS[n] for n in param_names]
combinations = list(itertools.product(*param_values))

script_dir = os.path.dirname(os.path.abspath(__file__))
agents_root = os.path.join(script_dir, "..", "agents")
os.makedirs(agents_root, exist_ok=True)


def run_experiment(run_id: int, run_params: dict):
    cfg = DEFAULTS.copy()
    cfg.update(run_params)

    FILE_NAME = f"{cfg['FILE_NAME']}_run_{run_id}"
    output_base_dir = os.path.join(agents_root, FILE_NAME)
    os.makedirs(output_base_dir, exist_ok=True)

    # Сохраняем конфигурацию для этого прогона
    out_file = Path(output_base_dir) / "training_config.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print(f"[Run {run_id}] Конфигурация сохранена в {out_file}")

    STATE_SNAPSHOT_PATH = os.path.join(output_base_dir, "initial_state.xml")

    sumoCmd = base_sumoCmd.copy()
    agents = {}
    total_reward_episode = {}
    try:
        traci.start(sumoCmd)

        tls_ids = traci.trafficlight.getIDList()

        controlled_edges_dict = {}
        for tls_id in tls_ids:
            controlled_edges = sumo_utils.get_tls_controlled_edges(tls_id)
            states = q_learning.create_state_table(tls_id, controlled_edges)
            agents[tls_id] = q_learning.QLearningAgent(
                tls_id=tls_id,
                states=states,
                actions=cfg["ACTIONS"],
                learning_rate=cfg["LEARNING_RATE"],
                discount_factor=cfg["DISCOUNT_FACTOR"],
                epsilon=cfg["EPSILON"],
                epsilon_decay=cfg["EPSILON_DECAY"],
                min_epsilon=cfg["MIN_EPSILON"]
            )
            controlled_edges_dict[tls_id] = controlled_edges

        all_lanes = list(traci.lane.getIDList())

        try:
            vtypes = traci.vehicletype.getIDList()
            used_vclasses = set(traci.vehicletype.getVehicleClass(t)
                                for t in vtypes)
        except Exception:
            used_vclasses = set()

        relevant_edges = set().union(*controlled_edges_dict.values()) if controlled_edges_dict else set()
        external_lanes = [l for l in all_lanes if not l.startswith(":")]

        metrics_cache = RewardMetricsCache(traci, relevant_edges, external_lanes,
                                       waiting_cache_enabled=True,
                                       waiting_cache_period=5,
                                       waiting_accumulated=False,
                                       waiting_among_waiting_only=True)
        metrics_cache.subscribe_all()

        unique_edges_count = len(relevant_edges)

        traci.simulation.saveState(STATE_SNAPSHOT_PATH)

        # Для аккумулирования суммарной награды по всем эпизодам
        sum_reward_all_episodes = {tls_id: 0.0 for tls_id in tls_ids}

        for episode in tqdm(range(cfg["NUM_EPISODES"]), desc=f"Run {run_id} Episodes"):
            traci.load(sumoCmd[1:])
            traci.simulation.step()

            accident_manager = None
            if cfg["ENABLE_ACCIDENTS"]:
                rng = random.Random(12345 + episode)
                accident_manager = AccidentManager(
                    all_lanes,
                    used_vclasses,
                    rng=rng,
                    mode=cfg["ACCIDENT_MODE"],
                    prob_per_step=cfg["ACCIDENT_PROB_PER_STEP"],
                    min_duration_steps=cfg["ACCIDENT_MIN_DURATION"],
                    max_duration_steps=cfg["ACCIDENT_MAX_DURATION"],
                    max_concurrent=cfg["ACCIDENT_MAX_CONCURRENT"],
                    min_margin_from_ends_m=10.0,
                    enable_markers=False,
                    marker_color=(255, 0, 0, 255),
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
                    tls_id, controlled_edges_dict[tls_id]
                )

            prev_phase_idx = {tls_id: traci.trafficlight.getPhase(
                tls_id) for tls_id in tls_ids}

            for current_step in range(cfg["MAX_SIMULATION_STEPS"]):
                try:
                    traci.simulationStep()
                except traci.exceptions.FatalTraCIError:
                    print(f"FatalTraCIError at simulation step {current_step}")
                    raise

                try:
                    metrics_cache.update_from_subscriptions()
                except Exception as e:
                    print(f"Metrics cache update failed, attempting resubscribe: {e}")
                    try:
                        metrics_cache.resubscribe()
                    except Exception as e2:
                        print(f"Resubscribe failed: {e2}")

                if cfg["ENABLE_ACCIDENTS"] and accident_manager is not None:
                    try:
                        accident_manager.step(current_step)
                    except traci.exceptions.FatalTraCIError:
                        raise
                    except Exception as e:
                        print(f"AccidentManager.step exception (ignored): {e}")

                if traci.simulation.getMinExpectedNumber() == 0:
                    if cfg["ENABLE_ACCIDENTS"] and accident_manager is not None:
                        try:
                            accident_manager.shutdown()
                        except Exception:
                            pass
                    break

                cur_phase_idx = {tls_id: traci.trafficlight.getPhase(
                    tls_id) for tls_id in tls_ids}

                global_reward = q_learning.calculate_global_reward(
                    tls_ids,
                    controlled_edges_dict,
                    unique_edges_count,
                    metrics=metrics_cache,
                    speed_weight=cfg["GLOBAL_SPEED_WEIGHT"],
                    wtime_weight=cfg["GLOBAL_WTIME_WEIGHT"],
                    occ_weight=cfg["GLOBAL_OCC_WEIGHT"],
                )

                for tls_id in tls_ids:
                    local_reward = q_learning.calculate_local_reward(
                        controlled_edges_dict[tls_id],
                        metrics=metrics_cache,
                        use_accident_penalty=cfg["USE_ACCIDENT_PENALTY"],
                        speed_weight=cfg["LOCAL_SPEED_WEIGHT"],
                        wtime_weight=cfg["LOCAL_WTIME_WEIGHT"],
                        occ_weight=cfg["LOCAL_OCC_WEIGHT"],
                        accident_weight=0.35,
                        accident_provider=lambda edges: accident_manager.get_edge_impacts(
                            edges)
                        if (cfg["ENABLE_ACCIDENTS"] and accident_manager is not None)
                        else {}
                    )

                    total_reward = q_learning.calculate_total_reward(
                        local_reward=local_reward,
                        global_reward=global_reward,
                        weight_local=cfg["WEIGHT_LOCAL"],
                        weight_global=cfg["WEIGHT_GLOBAL"]
                    )
                    total_reward_episode[tls_id] += total_reward

                    current_state = q_learning.create_state_for_tls(
                        tls_id, controlled_edges_dict[tls_id]
                    )

                    if last_states[tls_id] is not None and last_actions[tls_id] is not None:
                        agents[tls_id].update_q_table(
                            last_states[tls_id],
                            last_actions[tls_id],
                            total_reward,
                            current_state
                        )

                    if cur_phase_idx[tls_id] != prev_phase_idx[tls_id]:
                        chosen_action_value = agents[tls_id].choose_action(
                            current_state)
                        sumo_utils.set_phase_duration_for_new_phase(
                            tls_id, chosen_action_value)
                        last_states[tls_id] = current_state
                        last_actions[tls_id] = chosen_action_value

                prev_phase_idx = cur_phase_idx

            # после эпизода: decay и сохранение q-таблиц
            for tls_id in tls_ids:
                agents[tls_id].decay_epsilon()
                agents[tls_id].save_q_table(os.path.join(
                    output_base_dir, f"q_table_{tls_id}.npy"))

            try:
                if cfg["ENABLE_ACCIDENTS"] and accident_manager is not None:
                    accident_manager.shutdown()
            except Exception:
                pass

            # аккумулируем суммарную награду по всем TLS за этот эпизод
            for tls_id, val in total_reward_episode.items():
                sum_reward_all_episodes[tls_id] += val

            # сохраняем итоговую награду эпизода (на случай детального анализа)
            episode_rewards_path = Path(output_base_dir) / f"episode_{episode}_total_reward.json"
            try:
                with episode_rewards_path.open("w", encoding="utf-8") as f:
                    json.dump(total_reward_episode, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        # Сохранение суммарных наград за все эпизоды
        total_rewards_path = Path(output_base_dir) / "total_reward_all_episodes.json"
        with total_rewards_path.open("w", encoding="utf-8") as f:
            json.dump(sum_reward_all_episodes, f, ensure_ascii=False, indent=2)

        # Вычисление метрики для summary: средняя суммарная награда по TLS (можно заменить)
        try:
            metric_value = sum(sum_reward_all_episodes.values())
        except Exception:
            metric_value = None

        summary = {
            "run_id": run_id,
            "params": run_params,
            "metric_sum_reward": metric_value,
        }
        with open(os.path.join(output_base_dir, "summary.json"), "w", encoding="utf-8") as sf:
            json.dump(summary, sf, ensure_ascii=False, indent=2)

    except traci.exceptions.TraCIException as e:
        print(f"TraCI error: {e}")

    finally:
        try:
            traci.close()
        except traci.exceptions.FatalTraCIError:
            pass
        except Exception as e:
            print(f"Error closing TraCI connection: {e}")
        print(f"[Run {run_id}] Q-learning process finished.")

    try:
        test_agents.main(FILE_NAME)
    except Exception:
        pass


# Главный цикл по всем комбинациям
print(f"Total combinations to run: {len(combinations)}")
for idx, combo in enumerate(combinations, start=1):
    params = {name: value for name, value in zip(param_names, combo)}
    # Нормализация весов лок/глоб если заданы вместе
    if "WEIGHT_LOCAL" in params and "WEIGHT_GLOBAL" in params:
        s = params["WEIGHT_LOCAL"] + params["WEIGHT_GLOBAL"]
        if s != 0:
            params["WEIGHT_LOCAL"] = params["WEIGHT_LOCAL"] / s
            params["WEIGHT_GLOBAL"] = params["WEIGHT_GLOBAL"] / s
    run_experiment(idx, params)

# --- AGGREGATION & CLEANUP: собрать summary, выбрать лучший и удалить проигравшие ---
summary_files = sorted(glob.glob(os.path.join(agents_root, f"{DEFAULTS['FILE_NAME']}_run_*", "summary.json")))

if not summary_files:
    print("No summary.json files found; skipping aggregation.")
else:
    summaries = []
    for p in summary_files:
        try:
            with open(p, "r", encoding="utf-8") as f:
                summaries.append(json.load(f))
        except Exception as e:
            print(f"Failed to read {p}: {e}")

    # Попытка извлечь метрику из total_reward_all_episodes.json если summary не содержит метрики
    for s in summaries:
        if "metric_sum_reward" not in s or s["metric_sum_reward"] is None:
            run_id = s.get("run_id")
            run_dir = os.path.join(agents_root, f"{DEFAULTS['FILE_NAME']}_run_{run_id}")
            alt = os.path.join(run_dir, "total_reward_all_episodes.json")
            if os.path.exists(alt):
                try:
                    with open(alt, "r", encoding="utf-8") as f:
                        tr = json.load(f)
                        s["metric_sum_reward"] = sum(tr.values()) if isinstance(tr, dict) else float(tr)
                except Exception:
                    s["metric_sum_reward"] = None
            else:
                s["metric_sum_reward"] = None

    # Сохраняем CSV с метриками всех прогонов
    csv_path = os.path.join(agents_root, f"{DEFAULTS['FILE_NAME']}_grid_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=["run_id", "metric_sum_reward", "params"])
        writer.writeheader()
        for s in summaries:
            writer.writerow({
                "run_id": s.get("run_id"),
                "metric_sum_reward": s.get("metric_sum_reward"),
                "params": json.dumps(s.get("params", {}), ensure_ascii=False)
            })
    print(f"Grid summary saved to {csv_path}")

    # Выбрать лучший (максимум metric_sum_reward)
    valid = [s for s in summaries if s.get("metric_sum_reward") is not None]
    if not valid:
        print("No valid metric_sum_reward values found; nothing to select.")
    else:
        best = max(valid, key=lambda x: x["metric_sum_reward"])
        best_run_id = best["run_id"]
        print(f"Best run: {best_run_id} metric: {best['metric_sum_reward']}")
        # Удаляем все прочие директории run_*, оставляем только лучший
        all_run_dirs = sorted(glob.glob(os.path.join(agents_root, f"{DEFAULTS['FILE_NAME']}_run_*")))
        for d in all_run_dirs:
            if f"_run_{best_run_id}" in d:
                print(f"Keeping best run dir: {d}")
                continue
            try:
                shutil.rmtree(d)
                print(f"Removed directory: {d}")
            except Exception as e:
                print(f"Failed to remove {d}: {e}")
        print("Cleanup finished.")