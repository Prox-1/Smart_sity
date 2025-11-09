# utils/q_learning.py
from typing import Dict, Iterable, Optional
from dataclasses import dataclass

# Если у вас уже есть импорты traci/constants — оставьте как есть
from traci import constants as tc

# Импортируем тип кэша (чтобы была подсказка в IDE); можно обойтись Optional[Any]
try:
    from utils.metrics_cache import RewardMetricsCache
except Exception:
    RewardMetricsCache = None  # type: ignore


# --- Пример: добавляем параметр metrics с дефолтом ---
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

    if metrics is None:
        # === Legacy-путь: оставьте ваш существующий код здесь ===
        # return <ваш_старый_расчёт_награды>(controlled_edges, ...)
        raise RuntimeError(
            "calculate_local_reward legacy path is not filled in.")
    # === Быстрый путь на кэше ===

    edges = list(controlled_edges)
    stats = [metrics.get_edge_stats(e) for e in edges]

    veh = sum(s["veh"] for s in stats)
    halting = sum(s["halting"] for s in stats)
    mean_speed = (sum(s["speed"] * s["veh"]
                  for s in stats) / veh) if veh > 0 else 0.0
    mean_occ = sum(s["occ"] for s in stats) / len(stats) if stats else 0.0

    # Нормализации/веса — подберите под вашу методику
    DESIRED_SPEED = 13.89  # ~50 км/ч (пример)
    speed_score = (mean_speed / DESIRED_SPEED) if DESIRED_SPEED > 0 else 0.0

    reward = 1.5 * speed_score - 0.10 * halting - 0.50 * mean_occ

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

    if metrics is None:
        # === Legacy-путь: оставьте ваш существующий код здесь ===
        # return <ваш_старый_глобальный_расчёт>(...)
        raise RuntimeError(
            "calculate_global_reward legacy path is not filled in.")
    # === Быстрый путь на кэше ===

    all_edges = set().union(*[set(v) for v in controlled_edges_dict.values()])
    s = metrics.get_global_stats(all_edges)

    DESIRED_SPEED = 13.89
    speed_score = (s["speed"] / DESIRED_SPEED) if DESIRED_SPEED > 0 else 0.0
    halting_per_edge = (s["halting"] / max(1, len(all_edges)))
    occ = s["occ"]

    # Глобальная метрика — примерная (подкорректируйте веса под свою постановку)
    reward = 2.0 * speed_score - 0.05 * halting_per_edge - 0.30 * occ
    return float(reward)
