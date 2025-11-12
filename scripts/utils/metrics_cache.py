# utils/metrics_cache.py
from collections import defaultdict
from typing import Dict, List, Iterable, Optional, Union

# Важно: import constants должен быть успешным, но часть имён может отсутствовать
from traci import constants as tc

Numeric = Union[int, float]


def edge_from_lane(lane_id: str) -> str:
    """
    Надёжно получаем edgeID по laneID:
      "edge_foo_2" -> "edge_foo"
    Внутренние полосы начинаются с ":" и фильтруются снаружи.
    """
    if "_" in lane_id:
        return lane_id.rsplit("_", 1)[0]
    return lane_id


def unsubscribe_all_safe(traci_module) -> None:
    """
    Безопасно отписываемся от всех возможных доменов/ID.
    Заглушаем исключения — функция должна быть полностью идемпотентной и не падать.
    Вызовите перед traci.simulation.loadState(...) чтобы избежать невалидных подписок.
    """
    try:
        # По возможности используем существующие методы getIDList()
        domains = [
            ("vehicle", "getIDList", "unsubscribe"),
            ("person", "getIDList", "unsubscribe"),
            ("trafficlight", "getIDList", "unsubscribe"),
            ("edge", "getIDList", "unsubscribe"),
            ("lane", "getIDList", "unsubscribe"),
            ("poi", "getIDList", "unsubscribe"),
            ("polygon", "getIDList", "unsubscribe"),
            # на случай, если используется
            ("busstop", "getIDList", "unsubscribe")
        ]

        for domain_name, id_getter_name, unsub_name in domains:
            domain = getattr(traci_module, domain_name, None)
            if domain is None:
                continue
            id_getter = getattr(domain, id_getter_name, None)
            unsub = getattr(domain, unsub_name, None)
            if id_getter is None or unsub is None:
                continue
            try:
                ids = list(id_getter() or [])
            except Exception:
                ids = []
            for _id in ids:
                try:
                    unsub(_id)
                except Exception:
                    # глушим отдельные ошибки, продолжаем дальше
                    continue
    except Exception:
        # Защита на случай неожиданной ошибки в логике отписки
        return


class RewardMetricsCache:
    """
    Кэш метрик наград с использованием подписок по полосам.
    Автоматически подбирает доступные переменные для вашей версии TraCI
    и делает безопасные fallback'и на прямые геттеры при отсутствии констант.
    """

    def __init__(self, traci_module, edges: Iterable[str], all_lanes: Iterable[str],
                waiting_cache_enabled: bool = True,        # включить кэш среднего waiting time
                waiting_cache_period: int = 5,             # пересчитывать каждые N шагов
                waiting_accumulated: bool = False,         # использовать накопленный waiting?
                waiting_among_waiting_only: bool = True,   # среднее только по тем, у кого wt>0

    ) -> None:
        self.traci = traci_module
        self.edges: set[str] = set(edges)

        # Поддержка разных версий TraCI: определяем доступные varID
        self._veh_var_id = getattr(tc, "LAST_STEP_VEHICLE_NUMBER", None)
        self._spd_var_id = getattr(tc, "LAST_STEP_MEAN_SPEED", None)

        # OCCUPANCY бывает отсутствует в старых версиях
        self._occ_var_id = getattr(tc, "LAST_STEP_OCCUPANCY", None)
        self._has_direct_occ = hasattr(self.traci.lane, "getLastStepOccupancy")

        # HALTING varID может отсутствовать — используем прямой getter, если есть
        halt_candidates = [
            "LAST_STEP_HALTING_NUMBER",                  # современное имя
            "LAST_STEP_VEHICLE_HALTING_NUMBER",          # редкие сборки
            "LANE_LAST_STEP_HALTING_NUMBER",             # экзотика
        ]
        self._halt_var_id = None
        for name in halt_candidates:
            if hasattr(tc, name):
                self._halt_var_id = getattr(tc, name)
                break
        self._has_direct_halt = hasattr(
            self.traci.lane, "getLastStepHaltingNumber")

        # Список переменных для подписки формируем динамически
        self._lane_vars: List[int] = []
        for var_id in (self._veh_var_id, self._spd_var_id, self._occ_var_id, self._halt_var_id):
            if var_id is not None:
                self._lane_vars.append(var_id)

        # Фильтруем только внешние полосы и только те, что принадлежат нужным рёбрам
        self.edge_lanes: Dict[str, List[str]] = defaultdict(list)
        for lane in all_lanes:
            if lane.startswith(":"):
                continue
            e = edge_from_lane(lane)
            if e in self.edges:
                self.edge_lanes[e].append(lane)

        self._lane_stats: Dict[str, Dict[str, Numeric]] = {}
        self._edge_stats: Dict[str, Dict[str, Numeric]] = {}
        self._subscribed = False

        self._waiting_cache_enabled = bool(waiting_cache_enabled)
        self._waiting_cache_period = max(1, int(waiting_cache_period))
        self._waiting_accumulated = bool(waiting_accumulated)
        self._waiting_among_waiting_only = bool(waiting_among_waiting_only)
        self._waiting_cache: Dict[str, float] = {}
        self._step_counter = 0
    # Подписки

    def _compute_edge_waiting_mean_now(self, edge_id: str) -> float:
        """
        Немедленно вычисляет средний waiting time по ребру edge_id (в секундах)
        по текущему шагу симуляции (по всем полосам ребра).
        """
        lanes = self.edge_lanes.get(edge_id, [])
        total = 0.0
        cnt = 0

        for lane in lanes:
            try:
                veh_ids = list(self.traci.lane.getLastStepVehicleIDs(lane) or [])
            except Exception:
                veh_ids = []

            for vid in veh_ids:
                wt = 0.0
                try:
                    if self._waiting_accumulated:
                        wt = float(self.traci.vehicle.getAccumulatedWaitingTime(vid))
                    else:
                        wt = float(self.traci.vehicle.getWaitingTime(vid))
                except Exception:
                    # альтернативный геттер на случай частичной совместимости
                    try:
                        if self._waiting_accumulated:
                            wt = float(self.traci.vehicle.getWaitingTime(vid))
                        else:
                            wt = float(self.traci.vehicle.getAccumulatedWaitingTime(vid))
                    except Exception:
                        wt = 0.0

                if self._waiting_among_waiting_only:
                    if wt > 0.0:
                        total += wt
                        cnt += 1
                else:
                    total += wt
                    cnt += 1

        return (total / cnt) if cnt else 0.0

    def refresh_waiting_cache(self, edges: Optional[Iterable[str]] = None) -> None:
        """
        Принудительно пересчитать и обновить кэш среднего waiting по переданным рёбрам.
        Если edges=None — пересчитываем для всех известных рёбер.
        """
        if not self._waiting_cache_enabled:
            return
        if edges is None:
            edges = self.edge_lanes.keys()
        for e in edges:
            try:
                self._waiting_cache[e] = self._compute_edge_waiting_mean_now(e)
            except Exception:
                # глушим любые ошибки, оставляя предыдущее значение кэша
                continue

    def get_edge_waiting_mean(self, edge_id: str) -> float:
        """
        Возвращает закэшированное среднее waiting time по ребру.
        Если в кэше нет — вычисляет лениво и сохраняет.
        """
        if not self._waiting_cache_enabled:
            # если кэш выключен — считаем на лету
            return self._compute_edge_waiting_mean_now(edge_id)
        if edge_id not in self._waiting_cache:
            self._waiting_cache[edge_id] = self._compute_edge_waiting_mean_now(edge_id)
        return float(self._waiting_cache.get(edge_id, 0.0))

    def subscribe_all(self) -> None:
        if self._subscribed:
            return
        # Если по какой-то причине список пуст (крайне маловероятно) — ничего не делаем
        if not self._lane_vars:
            self._subscribed = True
            return
        for lanes in self.edge_lanes.values():
            for lane in lanes:
                self.traci.lane.subscribe(lane, self._lane_vars)
        self._subscribed = True

    def resubscribe(self) -> None:
        """
        После simulation.loadState() подписки сбрасываются — перевешиваем.
        """
        self._subscribed = False
        self.subscribe_all()

    # Обновление кэша

    def _clear_step_cache(self) -> None:
        self._lane_stats.clear()
        self._edge_stats.clear()

    def update_from_subscriptions(self) -> None:
        """
        Вызывать строго ПОСЛЕ traci.simulationStep().
        """
        if not self._subscribed:
            self.subscribe_all()
        
        self._step_counter += 1

        self._clear_step_cache()

        all_lane_results = self.traci.lane.getAllSubscriptionResults() or {}

        for lane_id, res in all_lane_results.items():
            # Базовые величины по умолчанию
            veh = 0
            spd = 0.0
            occ = 0.0
            halt = 0

            # Из подписки
            if self._veh_var_id is not None:
                veh = int(res.get(self._veh_var_id, 0))
            if self._spd_var_id is not None:
                spd = float(res.get(self._spd_var_id, 0.0))

            # Occupancy: из подписки либо прямым геттером, если varID недоступен
            if self._occ_var_id is not None:
                occ = float(res.get(self._occ_var_id, 0.0))
            elif self._has_direct_occ:
                try:
                    occ = float(self.traci.lane.getLastStepOccupancy(lane_id))
                except Exception:
                    occ = 0.0

            # Halting: из подписки либо прямым геттером
            if self._halt_var_id is not None:
                halt = int(res.get(self._halt_var_id, 0))
            elif self._has_direct_halt:
                try:
                    halt = int(
                        self.traci.lane.getLastStepHaltingNumber(lane_id))
                except Exception:
                    halt = 0
            else:
                # Самый дешёвый эвристический fallback (если ничего нет):
                # считаем, что если средняя скорость почти нулевая, то все "veh" — halting.
                halt = veh if spd < 0.1 else 0

            self._lane_stats[lane_id] = {
                "veh": veh,
                "speed": spd,
                "halting": halt,
                "occ": occ,
            }

        if self._waiting_cache_enabled and (self._step_counter % self._waiting_cache_period == 0):
            try:
                self.refresh_waiting_cache()
            except Exception:
                pass  # безопасно игнорируем

        # edge-агрегаты
        for edge, lanes in self.edge_lanes.items():
            veh_sum = 0
            halt_sum = 0
            speed_num = 0.0
            speed_den = 0
            occ_sum = 0.0
            occ_cnt = 0

            for lane in lanes:
                s = self._lane_stats.get(lane)
                if not s:
                    continue
                veh_sum += int(s["veh"])
                halt_sum += int(s["halting"])
                if s["veh"] > 0:
                    speed_num += float(s["speed"]) * int(s["veh"])
                    speed_den += int(s["veh"])
                occ_sum += float(s["occ"])
                occ_cnt += 1

            self._edge_stats[edge] = {
                "veh": veh_sum,
                "halting": halt_sum,
                "speed": (speed_den and (speed_num / speed_den)) or 0.0,
                "occ": (occ_cnt and (occ_sum / occ_cnt)) or 0.0,
                "waiting_mean": float(self._waiting_cache.get(edge, 0.0)) if self._waiting_cache_enabled
                                 else self._compute_edge_waiting_mean_now(edge),
            }

    # Доступ к метрикам

    def get_edge_stats(self, edge_id: str) -> Dict[str, Numeric]:
        return self._edge_stats.get(edge_id, {
            "veh": 0, "halting": 0, "speed": 0.0, "occ": 0.0, "waiting_mean": 0.0
        })

    def get_global_stats(self, edges: Optional[Iterable[str]] = None) -> Dict[str, Numeric]:
        if edges is None:
            edges = self._edge_stats.keys()

        veh_sum = 0
        halt_sum = 0
        speed_num = 0.0
        speed_den = 0
        occ_sum = 0.0
        occ_cnt = 0
        sum_waiting_mean = 0

        for e in edges:
            s = self._edge_stats.get(e)
            if not s:
                continue
            veh_sum += int(s["veh"])
            halt_sum += int(s["halting"])
            if s["veh"] > 0:
                speed_num += float(s["speed"]) * int(s["veh"])
                speed_den += int(s["veh"])
            sum_waiting_mean += float(s["waiting_mean"])
            occ_sum += float(s["occ"])
            occ_cnt += 1

        return {
            "veh": veh_sum,
            "halting": halt_sum,
            "speed": (speed_den and (speed_num / speed_den)) or 0.0,
            "occ": (occ_cnt and (occ_sum / occ_cnt)) or 0.0,
            "sum_waiting_mean": sum_waiting_mean,
        }
