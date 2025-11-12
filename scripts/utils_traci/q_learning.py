import os
import traci
import itertools
import collections
import numpy as np

from typing import Dict, Iterable, Optional
from utils_traci import sumo_utils


MAX_WAITING_TIME_PER_EDGE = 300

MAX_QUEUE_LENGTH_PER_EDGE = 50

MAX_VEHICLES_ARRIVED_PER_STEP = 50

MAX_SPEED = 70/3.6


def create_state_table(tls_id, controlled_edges):
    """
    Создает список всех возможных дискретных состояний для конкретного светофора (TLS).

    Состояние задаётся как кортеж:
        (phase, queue_cat_edge1, queue_cat_edge2, ..., queue_cat_edgeN)

    Где:
        - phase: индекс текущей фазы светофора (целое число)
        - queue_cat_edgeX: категория длины очереди на соответствующем входном ребре ('Low', 'Medium', 'High')

    Параметры:
        tls_id: идентификатор светофора (строка), используется для получения всех фаз через sumo_utils.
        controlled_edges: список (или итерируемый) идентификаторов ребер, которые контролирует этот светофор.

    Возвращает:
        all_states: список кортежей — все возможные сочетания (phase + комбинации категорий очередей)
    """
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
    """
    Инициализирует Q-таблицу для заданного набора состояний и действий.

    Параметры:
        states: список состояний (каждое состояние — итерируемый или кортеж).
        actions: список возможных действий (по умолчанию [+5, 0, -5], предполагают изменение длительности фазы в секундах).

    Возвращает:
        Q_table: словарь, где ключ — кортеж состояния, значение — словарь {action: q_value}
    """
    Q_table = {}

    for state in states:
        Q_table[tuple(state)] = {action: 0.0 for action in actions}

    return Q_table


def data2queue_categories(controlled_edges):
    """
    Преобразует текущую суммарную (не усреднённую) информацию об ожидании на каждом ребре в категорию.

    Логика категорий:
        - 'Low'    : waiting time < 10 сек
        - 'Medium' : 10 <= waiting time < 30 сек
        - 'High'   : waiting time >= 30 сек

    Параметры:
        controlled_edges: итерируемый список идентификаторов ребер.

    Возвращает:
        queue_cat: список категорий для каждого ребра в том же порядке, что и controlled_edges.
    """
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

    Параметры:
        tls_id: идентификатор светофора.
        controlled_edges: список идентификаторов ребер, контролируемых светофором.

    Использует:
        - traci.trafficlight.getPhase для получения текущей фазы.
        - data2queue_categories для получения категорий очередей.

    Возвращает:
        tuple: текущее дискретное состояние (phase, queue_cat_1, ..., queue_cat_N).
    """
    current_phase_index = traci.trafficlight.getPhase(tls_id)
    queue_categories_on_edges = data2queue_categories(controlled_edges)
    return (current_phase_index,) + tuple(queue_categories_on_edges)


def get_metrics(controlled_edges):
    """
    Собирает простые агрегированные метрики по списку ребер.

    Параметры:
        controlled_edges: итерируемый список идентификаторов ребер.

    Возвращает:
        waiting_time: суммарное время ожидания по всем ребрам (в секундах).
        halting_number: суммарное количество остановленных/тормозящих ТС за последний шаг.
    """
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
    Вычисляет локальную (для одного агента/светофора) награду на основе статистик по ребрам.

    Поддерживает два режима:
        - legacy (если metrics == None): не использовать кэш — (в текущей реализации legacy-ветвь не прописана,
          но оставлен интерфейс).
        - metrics-путь: если объект metrics передан, данные берутся из его API (metrics.get_edge_stats и т.п.),
          что минимизирует число TraCI вызовов.

    Основные метрики, используемые в награде:
        - mean_speed (вес положительный — чем выше, тем лучше)
        - mean_waiting_time (нормализованный — отрицательная составляющая)
        - mean_occ (occupancy) — отрицательная составляющая

    Параметры:
        controlled_edges: список ребер, контролируемых агентом.
        metrics: объект-кеш с методом get_edge_stats(edge_id) -> dict с полями:
                 {"veh": число_тс, "waiting_mean": среднее_время_ожидания, "speed": средняя_скорость, "occ": occupancy}
        use_accident_penalty: флаг использования штрафа за ДТП.
        accident_weight: множитель штрафа за суммарное влияние аварий на ребра.
        accident_provider: вызываемый объект, принимающий список ребер и возвращающий dict {edge_id: impact_value}.

    Возвращает:
        reward: вещественное число — итоговая локальная награда.
    """
    edges = list(controlled_edges)
    stats = [metrics.get_edge_stats(e) for e in edges]

    # Число машин на контролируемых ребрах
    veh = sum(s["veh"] for s in stats)

    # Среднее время ожидания по ребрам (среднее от mean для каждого ребра)
    mean_waiting_time = sum(s["waiting_mean"] for s in stats) / len(stats) if stats else 0.0

    # Взвешенная средняя скорость (по числу ед. транспорта)
    mean_speed = (sum(s["speed"] * s["veh"]
                  for s in stats) / veh) if veh > 0 else 0.0

    # Средняя загрузка (occupancy)
    mean_occ = sum(s["occ"] for s in stats) / len(stats) if stats else 0.0

    # Нормализации/референсные значения
    DESIRED_SPEED = 13.89  # ~50 км/ч
    MAX_WAITING_TIME = 300

    speed_score = (mean_speed / DESIRED_SPEED) if DESIRED_SPEED > 0 else 0.0
    normalized_waiting_time = (mean_waiting_time / MAX_WAITING_TIME) if MAX_WAITING_TIME > 0 else 0.0

    # Скомбинированная награда: положительная за скорость, отрицательные за ожидание и загрузку
    reward = 1.5 * speed_score - 1.2 * normalized_waiting_time - 0.70 * mean_occ

    # При необходимости учитываем штрафы за аварии
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
    Вычисляет глобальную награду для набора светофоров (агентов), используя агрегированные метрики.

    Параметры:
        tls_ids: итерируемый список идентификаторов светофоров (не обязательно используется внутри).
        controlled_edges_dict: словарь {tls_id: iterable_of_edges} для всех агентов.
        unique_edges_count: общее число уникальных ребер (может использоваться для нормализации — в текущей реализации не использован напрямую).
        metrics: объект-кеш с методом get_global_stats(edges_set) -> dict, ожидаются поля:
                 {"speed": avg_speed, "halting": total_halting, "sum_waiting_mean": суммарное среднее ожидание, "occ": mean_occupancy}

    Возвращает:
        reward: вещественное число — глобальная награда, комбинирующая скорость, ожидание и occupancy.
    """
    # Собираем все уникальные ребра, контролируемые агентами
    all_edges = set().union(*[set(v) for v in controlled_edges_dict.values()])

    # Берём агрегированные статистики из кеша
    s = metrics.get_global_stats(all_edges)

    DESIRED_SPEED = 13.89
    MAX_GLOBAL_WAITING_TIME = 300 * len(all_edges)

    speed_score = (s["speed"] / DESIRED_SPEED) if DESIRED_SPEED > 0 else 0.0
    halting_per_edge = (s["halting"] / max(1, len(all_edges)))
    normalized_waiting_time = (s["sum_waiting_mean"] / MAX_GLOBAL_WAITING_TIME) if MAX_GLOBAL_WAITING_TIME > 0 else 0.0
    occ = s["occ"]

    # Комбинация метрик для глобальной награды — веса можно подстроить под задачу
    reward = 1.0 * speed_score - 1.0 * normalized_waiting_time - 0.5 * occ

    return float(reward)


def calculate_total_reward(local_reward, global_reward, weight_local=0.5, weight_global=0.5):
    """
    Комбинирует локальную и глобальную награды в одну суммарную метрику.

    Параметры:
        local_reward: числовая локальная награда агента.
        global_reward: числовая глобальная награда системы.
        weight_local: вес локальной награды в итоговой сумме.
        weight_global: вес глобальной награды.

    Возвращает:
        комбинированную награду (float).
    """
    return weight_local*local_reward + weight_global*global_reward


class QLearningAgent:
    """
    Простой Q-learning агент для управления одним светофором (TLS).

    Атрибуты:
        tls_id: идентификатор светофора (строка).
        states: список всех допустимых состояний (итерируемый набор).
        actions: список доступных действий (например изменение длительности фазы).
        lr: learning rate (alpha).
        gamma: discount factor.
        epsilon: вероятность случайной (exploration) политики в eps-greedy.
        epsilon_decay: множитель, применяемый к epsilon после каждого эпизода/шага вызова decay_epsilon.
        min_epsilon: минимально допустимое значение epsilon.
        q_table: defaultdict, где ключ — состояние (tuple), значение — dict {action: q_value}.

    Методы:
        get_q_value(state, action) -> float
        choose_action(state) -> action
        update_q_table(state, action, reward, next_state) -> None
        decay_epsilon() -> None
        save_q_table(filename) -> None
        load_q_table(filename) -> None
    """

    def __init__(self, tls_id, states, actions, learning_rate=0.1, discount_factor=0.9, epsilon=0.1, epsilon_decay=0.995, min_epsilon=0.01):
        """
        Инициализация агента. Заполняет Q-таблицу нулевыми значениями для всех пар (state, action).

        Параметры:
            tls_id: идентификатор светофора.
            states: список/итерируемый набор всех состояний.
            actions: список возможных действий.
            learning_rate: коэффициент обучения.
            discount_factor: discount factor (gamma).
            epsilon: стартовое значение epsilon для eps-greedy.
            epsilon_decay: множитель для уменьшения epsilon.
            min_epsilon: минимальное значение epsilon.
        """
        self.tls_id = tls_id
        self.states = states
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Используем defaultdict для удобства: при обращении к несуществующему состоянию будет создан словарь действий со значениями 0.0
        self.q_table = collections.defaultdict(
            lambda: {action: 0.0 for action in self.actions})

        # Явно инициализируем записи для известных состояний
        for state_list in self.states:
            state_tuple = tuple(state_list)
            # При обращении к ключу сработает lambda и создаст словарь значений действий
            self.q_table[state_tuple]

    def get_q_value(self, state, action):
        """
        Возвращает Q-значение для пары (state, action).
        Ожидается, что state — уже кортеж/ключ, соответствующий ключам q_table.
        """
        return self.q_table[state][action]

    def choose_action(self, state):
        """
        Выбирает действие по eps-greedy политике:
            - с вероятностью epsilon выбирается случайное действие
            - иначе выбирается действие с максимальным Q (при равенстве — случайно среди лучших)

        Параметр:
            state: текущее состояние (кортеж)

        Возвращает:
            выбранное действие из self.actions
        """
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.actions)
        else:
            q_values_for_state = self.q_table[state]
            max_q = -np.inf
            best_actions = []

            # Выбираем все действия, имеющие максимальное q значение (для случайного выбора между ними)
            for action, q_val in q_values_for_state.items():
                if q_val > max_q:
                    max_q = q_val
                    best_actions = [action]
                elif q_val == max_q:
                    best_actions.append(action)

            return np.random.choice(best_actions)

    def update_q_table(self, state, action, reward, next_state):
        """
        Обновляет Q-таблицу по правилу Q-learning:
            Q(s,a) = Q(s,a) + lr * (r + gamma * max_a' Q(s', a') - Q(s,a))

        Параметры:
            state: текущее состояние (tuple)
            action: выполненное действие
            reward: полученная награда (float)
            next_state: следующее состояние (tuple)
        """
        current_q = self.get_q_value(state, action)
        max_q_next = max(self.q_table[next_state].values())
        new_q = current_q + self.lr * \
            (reward + self.gamma * max_q_next - current_q)
        self.q_table[state][action] = new_q

    def decay_epsilon(self):
        """
        Уменьшает epsilon с учётом epsilon_decay, но не ниже min_epsilon.
        Вызывать, например, по окончании эпизода.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save_q_table(self, filename="q_table.npy"):
        """
        Сохраняет текущую Q-таблицу в файл numpy (.npy).
        defaultdict не сериализуется напрямую, поэтому сначала преобразуем к обычному dict.

        Параметры:
            filename: путь к файлу для сохранения.
        """
        save_data = {k: dict(v) for k, v in self.q_table.items()}
        np.save(filename, save_data)
        # print(f"Q-table for {self.tls_id} saved to {filename}")

    def load_q_table(self, filename="q_table.npy"):
        """
        Загружает Q-таблицу из файла .npy, если файл существует.
        Восстанавливает структуру как defaultdict, чтобы поведение осталось прежним.

        Параметры:
            filename: путь к файлу для загрузки.
        """
        if os.path.exists(filename):
            loaded_data = np.load(filename, allow_pickle=True).item()
            # Преобразуем загруженный dict обратно в defaultdict с default-словари действий
            self.q_table = collections.defaultdict(
                lambda: {action: 0.0 for action in self.actions}, loaded_data)
            # print(f"Q-table for {self.tls_id} loaded from {filename}")
        else:
            print(
                f"No Q-table file found at {filename}. Starting with fresh Q-table.")