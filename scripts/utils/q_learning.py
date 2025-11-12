from typing import Dict, Iterable, Optional
import os
import libsumo as traci
import itertools
import collections
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from utils import sumo_utils

USING_LIBSUMO = True

MAX_WAITING_TIME_PER_EDGE = 300
MAX_QUEUE_LENGTH_PER_EDGE = 50
MAX_VEHICLES_ARRIVED_PER_STEP = 50
MAX_SPEED = 70/3.6


def create_state_table(tls_id, controlled_edges):
    queue_categories = ['Low', 'Medium', 'High']
    phases = sumo_utils.get_all_tls_phases(tls_id)
    all_states = []
    combinations_of_queues = list(itertools.product(
        queue_categories, repeat=len(controlled_edges)))
    for phase in phases:
        for queue_combination in combinations_of_queues:
            state = (phase,) + queue_combination
            all_states.append(state)
    return all_states


def create_Q_table(states: list, actions: list = [+5, 0, -5]):
    Q_table = {}
    for state in states:
        Q_table[tuple(state)] = {action: 0.0 for action in actions}
    return Q_table


def data2queue_categories(controlled_edges):
    queue_cat = []
    for edge_id in controlled_edges:
        if traci.edge.getWaitingTime(edge_id) < 10:
            queue_cat.append('Low')
        elif traci.edge.getWaitingTime(edge_id) < 30:
            queue_cat.append('Medium')
        else:
            queue_cat.append('High')
    return queue_cat


def create_state_for_tls(tls_id, controlled_edges):
    """
    Создает дискретное состояние для Q-learning агента на основе данных SUMO для данного светофора.
    Возвращает кортеж (текущая_фаза, категория_очереди_дороги1, ..., категория_очереди_дорогиN).
    """
    current_phase_index = traci.trafficlight.getPhase(tls_id)
    queue_categories_on_edges = data2queue_categories(controlled_edges)
    return (current_phase_index,) + tuple(queue_categories_on_edges)


def get_metrics(controlled_edges):
    waiting_time = 0
    halting_number = 0
    for edge_id in controlled_edges:
        waiting_time += traci.edge.getWaitingTime(
            edge_id)  # type: ignore
        halting_number += traci.edge.getLastStepHaltingNumber(
            edge_id)  # type: ignore
    return waiting_time, halting_number


def calculate_local_reward(
    controlled_edges: Iterable[str],
    metrics: Optional["RewardMetricsCache"] = None,
    *,
    use_accident_penalty: bool = False,
    accident_weight: float = 0.35,
    accident_provider=None,
) -> float:
    """
    Если metrics не передан — используем ваш прежний (legacy) код.
    Если metrics передан — берём данные из кэша подписок без дополнительных TraCI-вызовов.
    """

    edges = list(controlled_edges)
    stats = [metrics.get_edge_stats(e) for e in edges]

    veh = sum(s["veh"] for s in stats)
    mean_waiting_time = sum(s["waiting_mean"] for s in stats) / len(stats) if stats else 0.0
    mean_speed = (sum(s["speed"] * s["veh"]
                  for s in stats) / veh) if veh > 0 else 0.0
    mean_occ = sum(s["occ"] for s in stats) / len(stats) if stats else 0.0

    # Нормализации/веса
    DESIRED_SPEED = 13.89  # ~50 км/ч 
    MAX_WAITING_TIME = 300
    speed_score = (mean_speed / DESIRED_SPEED) if DESIRED_SPEED > 0 else 0.0
    normalized_waiting_time = (mean_waiting_time / MAX_WAITING_TIME) if MAX_WAITING_TIME > 0 else 0.0


    reward = 1.5 * speed_score - 1.2 * normalized_waiting_time - 0.70 * mean_occ

    if use_accident_penalty and accident_provider is not None:
        impacts = accident_provider(edges) or {}
        total_impact = sum(float(impacts.get(e, 0.0)) for e in edges)
        reward -= accident_weight * total_impact

    return float(reward)


def calculate_global_reward(
    tls_ids: Iterable[str],
    controlled_edges_dict: Dict[str, Iterable[str]],
    unique_edges_count: int,
    metrics: Optional["RewardMetricsCache"] = None,
) -> float:
    """
    Аналогично: если metrics не передан — оставьте ваш прежний код.
    Если metrics передан — считаем глобальные метрики из кэша.
    """

    # === Быстрый путь на кэше ===

    all_edges = set().union(*[set(v) for v in controlled_edges_dict.values()])
    s = metrics.get_global_stats(all_edges)

    DESIRED_SPEED = 13.89
    MAX_GLOBAL_WAITING_TIME = 300 * len(all_edges)
    speed_score = (s["speed"] / DESIRED_SPEED) if DESIRED_SPEED > 0 else 0.0
    halting_per_edge = (s["halting"] / max(1, len(all_edges)))
    normalized_waiting_time = (s["sum_waiting_mean"] / MAX_GLOBAL_WAITING_TIME) if MAX_GLOBAL_WAITING_TIME > 0 else 0.0
    occ = s["occ"]

    # Глобальная метрика — примерная (подкорректируйте веса под свою постановку)
    reward = 1.0 * speed_score - 1.0 * normalized_waiting_time - 0.5 * occ
    return float(reward)


def calculate_total_reward(local_reward, global_reward, weight_local=0.5, weight_global=0.5):
    return weight_local*local_reward + weight_global*global_reward


class QLearningAgent:

    def __init__(self, tls_id, states, actions, learning_rate=0.1, discount_factor=0.9, epsilon=0.1, epsilon_decay=0.995, min_epsilon=0.01):
        self.tls_id = tls_id
        self.states = states
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = collections.defaultdict(
            lambda: {action: 0.0 for action in self.actions})
        for state_list in self.states:
            state_tuple = tuple(state_list)  # Преобразуем в кортеж
            self.q_table[state_tuple]

    def get_q_value(self, state, action):
        return self.q_table[state][action]

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.actions)
        else:
            q_values_for_state = self.q_table[state]
            max_q = -np.inf
            best_actions = []
            for action, q_val in q_values_for_state.items():
                if q_val > max_q:
                    max_q = q_val
                    best_actions = [action]
                elif q_val == max_q:
                    best_actions.append(action)
            return np.random.choice(best_actions)

    def update_q_table(self, state, action, reward, next_state):
        current_q = self.get_q_value(state, action)
        max_q_next = max(self.q_table[next_state].values())
        new_q = current_q + self.lr * \
            (reward + self.gamma * max_q_next - current_q)
        self.q_table[state][action] = new_q

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save_q_table(self, filename="q_table.npy"):
        """Сохраняет Q-таблицу в файл."""
        # defaultdict не сериализуется напрямую, преобразуем в обычный dict
        save_data = {k: dict(v) for k, v in self.q_table.items()}
        np.save(filename, save_data)
        # print(f"Q-table for {self.tls_id} saved to {filename}")

    def load_q_table(self, filename="q_table.npy"):
        """Загружает Q-таблицу из файла."""
        if os.path.exists(filename):
            loaded_data = np.load(filename, allow_pickle=True).item()
            # Преобразуем загруженный dict обратно в defaultdict
            self.q_table = collections.defaultdict(
                lambda: {action: 0.0 for action in self.actions}, loaded_data)
            # print(f"Q-table for {self.tls_id} loaded from {filename}")
        else:
            print(
                f"No Q-table file found at {filename}. Starting with fresh Q-table.")
