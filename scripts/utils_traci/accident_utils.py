from __future__ import annotations
from typing import Iterable

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import random

import traci

@dataclass
class Accident:
    """
    Описание одной «аварии» на полосе.

    Поля:
    - lane_id: идентификатор полосы, на которой произошла авария.
    - start_step, end_step: номера шагов симуляции начала и конца аварии.
    - prev_max_speed: сохранённый maxSpeed полосы до применения lane_block (для восстановления).
    - prev_allowed: сохранённый список allowed-классов полосы до изменения (None означает «все разрешены»).
    - obstacle_*: данные, связанные с obstacle-режимом (id спавненного транспортного средства, ребро, позиция, индекс полосы).
    - marker_*: данные GUI-маркера (poi id и координаты), используются для визуализации в SUMO-GUI.
    """
    lane_id: str
    start_step: int
    end_step: int
    # сохранённое состояние (для lane_block)
    prev_max_speed: float
    prev_allowed: Optional[List[str]]  # None = "все классы были разрешены"
    # obstacle-режим
    obstacle_veh_id: Optional[str] = None
    obstacle_edge_id: Optional[str] = None
    obstacle_pos: Optional[float] = None
    obstacle_lane_index: Optional[int] = None
    # GUI-маркер
    marker_poi_id: Optional[str] = None
    marker_x: Optional[float] = None
    marker_y: Optional[float] = None


class AccidentManager:
    """
    Менеджер псевдослучайных аварий/препятствий для SUMO.

    Основные возможности:
    - два режима работы:
      * "lane_block": имитируем закрытие полосы (setMaxSpeed=0 + запрет vClass).
      * "obstacle": создаём неподвижный транспорт (vehicle) и ставим ему стоп на полосе.
      При невозможности спавна препятствия — есть фолбэк к lane_block.
    - отметки (POI) для визуализации в GUI.
    - безопасные проверки существования lane/edge/vehicle/poi и устойчивые операции (try/except вокруг вызовов traci).
    - подсчёт «влияния» аварий на заданные ребра (доля поражённых полос).
    - лимит одновременных активных аварий, параметры длительности, вероятности спавна и т.д.

    Примечания по надёжности:
    - большинство операций с traci защищены try/except, чтобы код продолжал работать при отсутствии некоторых API или в случае ошибок.
    - кешируются маршруты и vtype для ускорения и предотвращения повторного создания.
    """

    def __init__(
        self,
        lane_ids: List[str],
        used_vclasses: Set[str],
        *,
        rng: random.Random,
        mode: str = "lane_block",
        prob_per_step: float = 0.002,
        min_duration_steps: int = 100,
        max_duration_steps: int = 300,
        max_concurrent: int = 3,
        min_margin_from_ends_m: float = 10.0,
        # Параметры маркера
        enable_markers: bool = True,
        marker_color: Tuple[int, int, int, int] = (255, 0, 0, 255),
        marker_layer: int = 10,
        marker_size: Tuple[int, int] = (6, 6),
        marker_type: str = "ACCIDENT",
        marker_label: str = "ДТП",
    ):
        """
        Инициализация менеджера.

        Аргументы:
        - lane_ids: список candidate lane_id — из них выбираются места для аварий.
          В конструкторе мы фильтруем внутренние/несуществующие полосы (проверка traci.lane.getEdgeID).
        - used_vclasses: набор vClass, которые будут запрещены на полосе при lane_block (если не пуст).
        - rng: объект random.Random для детерминированности/подмены seed.
        - mode: "lane_block" или "obstacle" — режим по умолчанию.
        - prob_per_step: вероятность создания аварии на шаге (если есть свободные полосы).
        - min/max_duration_steps, max_concurrent: управление длительностью и количеством активных аварий.
        - min_margin_from_ends_m: минимальный отступ от концов полосы при выборе позиции маркера/препятствия.
        - enable_markers и связанные параметры: внешний вид и поведение POI-маркеров.
        """
        # Обёртки для безопасности: отфильтруем полосы, на которых треги не падают
        self.lane_candidates = []
        for l in lane_ids:
            try:
                # если lane не существует — getEdgeID бросит; пропускаем
                e = traci.lane.getEdgeID(l)
                if not e.startswith(":"):
                    # исключаем внутренние ребра, идентификаторы которых обычно начинаются с ":"
                    self.lane_candidates.append(l)
            except Exception:
                # пропустим проблемные полосы
                continue

        # настройки и параметры
        self.used_vclasses = set(used_vclasses)
        self.rng = rng
        self.mode = mode
        self.prob = prob_per_step
        self.min_dur = min_duration_steps
        self.max_dur = max_duration_steps
        self.max_concurrent = max_concurrent
        self.min_margin = min_margin_from_ends_m

        # параметры маркера
        self.enable_markers = enable_markers
        self.marker_color = marker_color
        self.marker_layer = marker_layer
        self.marker_w, self.marker_h = marker_size
        self.marker_type = marker_type
        self.marker_label = marker_label

        # активные аварии: mapping lane_id -> Accident
        self.active: Dict[str, Accident] = {}
        # кэш: route для каждого ребра и vtype для каждого vClass
        self._edge_route_id: Dict[str, str] = {}
        self._vclass_vtype: Dict[str, str] = {}

        # Порядок предпочтения vClass при подборе, если lane.getAllowed пуст
        self._vclass_preference = [
            "passenger", "bus", "delivery", "authority", "taxi",
            "motorcycle", "evehicle", "emergency", "truck", "trailer",
            "coach", "tram", "rail_urban"
        ]
        # вспомогательные счётчики и контейнеры
        self._veh_seq = 0
        self.episode = -1
        self.counter = 0
        self.spawned_ids = set()
        self.pending_ops = []
        self.cooldown_until = 0

    # ---------- Безопасные проверки ----------
    def _veh_exists(self, vid: str) -> bool:
        """
        Возвращает True, если транспорт с id vid существует в симуляции.
        Оборачиваем в try/except, чтобы при отказе traci метод возвращал False.
        """
        try:
            return vid in set(traci.vehicle.getIDList())
        except Exception:
            return False

    def _lane_exists(self, lane_id: str) -> bool:
        """
        Проверяет существование полосы через traci.lane.getLength.
        Возвращает False при исключении.
        """
        try:
            _ = traci.lane.getLength(lane_id)
            return True
        except Exception:
            return False

    def _edge_exists(self, edge_id: str) -> bool:
        """
        Проверяет существование ребра (edge) через traci.edge.getLaneNumber.
        """
        try:
            _ = traci.edge.getLaneNumber(edge_id)
            return True
        except Exception:
            return False

    def _poi_exists(self, poi_id: str) -> bool:
        """
        Проверяет наличие POI по списку идентификаторов.
        """
        try:
            return poi_id in traci.poi.getIDList()
        except Exception:
            return False

    def _safe_sim_time_int(self) -> int:
        """
        Возвращает текущее время симуляции как int, в случае ошибки — 0.
        Полезно для генерации уникальных id.
        """
        try:
            t = traci.simulation.getTime()
            return int(t)
        except Exception:
            return 0

    # ---------- Вспомогательные ----------
    def _lane_index_from_id(self, lane_id: str) -> int:
        """
        Извлекает индекс полосы из lane_id вида '<edge>_<index>'.
        Если парсинг не удался — возвращаем 0.
        """
        try:
            return int(lane_id.rsplit("_", 1)[1])
        except Exception:
            return 0

    def _safe_pos_on_lane(self, lane_id: str) -> float:
        """
        Вычисляет безопасную позицию (в метрах) вдоль полосы для установки маркера или спавна препятствия.
        Подстраивается под длину полосы, использует min_margin и случайный выбор внутри допустимого диапазона.
        На случай ошибок возвращает разумные дефолтные значения.
        """
        try:
            length = float(traci.lane.getLength(lane_id))
        except Exception:
            # если не можем получить длину — вернём 0.5 как безопасный дефолт
            return 0.5
        if length <= 1.0:
            return max(0.0, length * 0.5)
        # adaptive margin: минимум 0.5м и максимум min_margin или 20% от длины
        margin = max(0.5, min(self.min_margin, length * 0.2))
        min_pos = margin
        max_pos = max(min_pos + 0.1, length - margin)
        pos = float(self.rng.uniform(min_pos, max_pos))
        # Жёстко гарантируем границы, чтобы не выйти за пределы полосы
        pos = min(max(0.1, pos), max(0.2, length - 0.2))
        return pos

    def _store_prev_state(self, lane_id: str) -> Tuple[float, Optional[List[str]]]:
        """
        Сохраняет предыдущее состояние полосы: maxSpeed и allowed-классы.
        Возвращает кортеж (prev_speed, prev_allowed), где prev_allowed == None трактуется как «все разрешены».
        В случае ошибок возвращает запасные значения.
        """
        prev_speed = 0.0
        prev_allowed = None
        try:
            prev_speed = traci.lane.getMaxSpeed(lane_id)
        except Exception:
            prev_speed = 0.0
        try:
            allowed = traci.lane.getAllowed(lane_id)  # [] => "все разрешены"
            prev_allowed = None if len(allowed) == 0 else allowed
        except Exception:
            prev_allowed = None
        return prev_speed, prev_allowed

    def _restore_lane_state(self, acc: Accident) -> None:
        """
        Восстанавливает сохранённые параметры полосы из Accident.prev_*.
        Игнорирует ошибки (только попытка восстановления).
        """
        if not acc:
            return
        if not self._lane_exists(acc.lane_id):
            return
        try:
            traci.lane.setMaxSpeed(acc.lane_id, acc.prev_max_speed)
        except Exception:
            pass
        try:
            if acc.prev_allowed is None:
                # пустой список означает «все разрешены»
                traci.lane.setAllowed(acc.lane_id, [])
            else:
                traci.lane.setAllowed(acc.lane_id, acc.prev_allowed)
        except Exception:
            pass

    def _ensure_edge_route(self, edge_id: str) -> str:
        """
        Гарантирует существование простого маршрута, состоящего из одного ребра (edge_id).
        - Кеширует имя маршрута в self._edge_route_id.
        - Если traci.route.add выбрасывает (например, route уже есть) — это безопасно игнорируется.
        Возвращает id маршрута (в любом случае).
        """
        if not edge_id:
            return ""
        rid = self._edge_route_id.get(edge_id)
        if rid is not None:
            return rid
        rid = f"__acc_route__{edge_id}"
        try:
            # если route уже есть — add бросит; ловим
            existing = []
            try:
                existing = traci.route.getIDList()
            except Exception:
                existing = []
            if rid not in existing:
                traci.route.add(rid, [edge_id])
        except Exception:
            # в случае ошибки — всё равно кешируем имя маршрута, чтобы не пытаться снова и снова
            pass
        self._edge_route_id[edge_id] = rid
        return rid

    def _get_allowed_vclass_for_lane(self, lane_id: str) -> Optional[str]:
        """
        Подбирает vClass, разрешённый на конкретной полосе:
        - Если lane.getAllowed() возвращает непустой список — берём первый элемент.
        - Иначе получаем список disallowed и выбираем первый vClass из _vclass_preference, которого нет в disallowed.
        - Если ничего не найдено — возвращаем None.
        """
        try:
            allowed = traci.lane.getAllowed(lane_id)
        except Exception:
            allowed = []
        if allowed:
            return allowed[0]

        try:
            disallowed = set(traci.lane.getDisallowed(lane_id))
        except Exception:
            disallowed = set()

        for vc in self._vclass_preference:
            if vc not in disallowed:
                return vc
        return None

    def _ensure_obstacle_vtype(self, vclass: str) -> str:
        """
        Обеспечивает наличие vehicletype для заданного vClass.
        - Если в кеше уже есть vt — возвращаем его.
        - Иначе пытаемся создать новый vtype как копию DEFAULT_VEHTYPE (если доступно) или добавить новый.
        - Устанавливаем vehicleClass, color, length, width — если traci поддерживает эти операции.
        - Всегда кешируем итоговый идентификатор vtype и возвращаем его.
        """
        vt = self._vclass_vtype.get(vclass)
        if vt:
            return vt
        vt = f"__acc_vtype__{vclass}"
        try:
            base_types = set()
            try:
                base_types = set(traci.vehicletype.getIDList())
            except Exception:
                base_types = set()
            if "DEFAULT_VEHTYPE" in base_types:
                try:
                    traci.vehicletype.copy("DEFAULT_VEHTYPE", vt)
                except Exception:
                    # Возможно уже создан или не поддерживается copy
                    pass
            else:
                try:
                    # Если add не поддерживает параметры — просто пробуем создать
                    traci.vehicletype.add(vt)
                except Exception:
                    pass

            try:
                traci.vehicletype.setVehicleClass(vt, vclass)
            except Exception:
                pass

            try:
                # визуальные параметры для удобства отладки/визуализации
                traci.vehicletype.setColor(vt, (255, 128, 0, 255))
                traci.vehicletype.setLength(vt, 4.5)
                traci.vehicletype.setWidth(vt, 2.0)
            except Exception:
                pass
        finally:
            # кешируем в любом случае (даже если создание прошло не полностью)
            self._vclass_vtype[vclass] = vt
            return vt

    def get_edge_impacts(
        self,
        edge_ids: Iterable[str],
        *,
        severity_lane_block: float = 1.0,
        severity_obstacle: float = 0.7,
    ) -> Dict[str, float]:
        """
        Возвращает словарь {edge_id: impact в [0,1]} — оценка влияния активных аварий на переданные ребра.
        - impact моделируется как отношение количества поражённых полос к общему числу полос ребра.
        - Результат масштабируется коэффициентом severity в зависимости от режима (lane_block суровее).
        - Игнорируются ребра, не входящие в edge_ids.
        - В случае ошибок получения lane/edge информации используется запасное значение (1 полоса).
        """
        edge_ids = list(edge_ids)
        impacts: Dict[str, float] = {e: 0.0 for e in edge_ids}
        if not self.active or not edge_ids:
            return impacts

        # собираем индексы поражённых полос для каждого ребра
        affected_lane_indices_by_edge: Dict[str, Set[int]] = {
            e: set() for e in edge_ids}

        for acc in self.active.values():
            try:
                e = traci.lane.getEdgeID(acc.lane_id)
            except Exception:
                continue
            if e not in impacts:
                continue
            li = self._lane_index_from_id(acc.lane_id)
            affected_lane_indices_by_edge[e].add(li)

        severity = severity_lane_block if self.mode == "lane_block" else severity_obstacle

        for e in edge_ids:
            try:
                total_lanes = max(1, int(traci.edge.getLaneNumber(e)))
            except Exception:
                total_lanes = 1
            affected = len(affected_lane_indices_by_edge[e])
            frac = min(1.0, affected / float(total_lanes))
            impacts[e] = float(frac * max(0.0, min(1.0, severity)))

        return impacts

    # ---------- lane_block режим ----------
    def _apply_lane_block(self, lane_id: str) -> None:
        """
        Применяет lane_block на заданной полосе:
        - ставит maxSpeed=0
        - если self.used_vclasses не пуст — устанавливает disallowed в список этих классов
        Все операции — в try/except, чтобы не падать при отсутствии API или ошибках.
        """
        if not self._lane_exists(lane_id):
            return
        try:
            traci.lane.setMaxSpeed(lane_id, 0.0)
        except Exception:
            pass
        if self.used_vclasses:
            try:
                traci.lane.setDisallowed(lane_id, list(self.used_vclasses))
            except Exception:
                pass

    # ---------- obstacle режим ----------
    def _spawn_obstacle(self, lane_id: str, duration: int, pos_override: Optional[float] = None) -> Tuple[str, str, float, int]:
        """
        Создаёт неподвижное транспортное средство на полосе, ставит ему stop.
        Возвращает кортеж (veh_id, edge_id, pos, lane_index).

        Поведение и особенности:
        - Вычисляет edge_id и порядковый индекс полосы.
        - Выбирает безопасную позицию на полосе (_safe_pos_on_lane) если pos_override не задан.
        - Подбирает vClass допустимый для полосы через _get_allowed_vclass_for_lane.
        - Создаёт/копирует vehicletype через _ensure_obstacle_vtype.
        - Обёрнуто в try/except: при проблемах могут быть применены fallback-операции
          (установка скорости 0 и speedMode=0).
        - Генерация id транспорта использует текущее время симуляции и внутренний счётчик,
          чтобы минимизировать вероятность конфликтов id.
        - На выходе гарантируем корректные индекс и позицию в пределах полосы (границы).
        """
        edge_id = traci.lane.getEdgeID(lane_id)
        lane_index = self._lane_index_from_id(lane_id)
        length = float(traci.lane.getLength(lane_id))
        pos = self._safe_pos_on_lane(
            lane_id) if pos_override is None else float(pos_override)
        pos = min(max(0.1, pos), max(0.2, length - 0.2))
        vclass = self._get_allowed_vclass_for_lane(lane_id)
        if not vclass:
            # если нет подходящего vClass — сигнализируем ошибку вызывающему коду
            raise RuntimeError("No suitable vClass for this lane")
        vtype_id = self._ensure_obstacle_vtype(vclass)
        route_id = self._ensure_edge_route(edge_id)
        self._veh_seq += 1
        veh_id = f"__acc_{lane_id.replace('#', '_').replace(':', '_')}_{int(traci.simulation.getTime())}_{self._veh_seq}"
        # Add vehicle — при ошибке exception будет проброшен наружу
        traci.vehicle.add(veh_id, route_id, typeID=vtype_id)
        # Validate laneIndex bounds: если получили nlanes — ограничиваем индекс
        try:
            nlanes = int(traci.edge.getLaneNumber(edge_id))
            lane_index = max(0, min(lane_index, max(0, nlanes - 1)))
        except Exception:
            lane_index = max(0, lane_index)
        try:
            # перемещаем транспорт на конкретную позицию/полосу
            traci.vehicle.moveTo(veh_id, lane_id, pos)
        except Exception:
            pass
        try:
            # ставим стоп у ребра (edgeID) с указанием позиции и длительности
            traci.vehicle.setStop(
                veh_id,
                edgeID=edge_id,
                pos=pos,
                laneIndex=lane_index,
                duration=max(1, int(duration)),
                flags=0
            )
        except Exception:
            # фолбэк: если setStop недоступен или упал — принудительно обнулим скорость
            try:
                traci.vehicle.setSpeedMode(veh_id, 0)
                traci.vehicle.setSpeed(veh_id, 0.0)
            except Exception:
                pass
        try:
            traci.vehicle.setSpeedMode(veh_id, 0)
            traci.vehicle.setSpeed(veh_id, 0.0)
        except Exception:
            pass
        return veh_id, edge_id, pos, lane_index

    def _despawn_obstacle(self, veh_id: str) -> None:
        """
        Удаляет спавненное препятствие (vehicle.remove). Игнорирует ошибки.
        """
        try:
            if self._veh_exists(veh_id):
                traci.vehicle.remove(veh_id)
        except Exception:
            pass

    # ---------- GUI POI-маркеры ----------
    def _add_marker(self, lane_id: str, edge_id: str, pos: float) -> Tuple[str, float, float]:
        """
        Добавляет POI-маркер в GUI по координатам, полученным через convert2D.
        Возвращает (poi_id, x, y).
        - Если enable_markers == False — возвращаем пустой id и NaN координаты.
        - В случае ошибок convert2D пытаемся получить shape полосы и взять её среднюю точку.
        - Добавление poi делается с полным набором параметров; при ошибке используется упрощённый fallback.
        - Ставит параметр name/label, если traci поддерживает setParameter.
        """
        if not self.enable_markers:
            return "", float("nan"), float("nan")

        lane_index = self._lane_index_from_id(lane_id)

        x = y = 0.0
        try:
            # convert2D может бросать, особенно в headless-режиме или в разных версиях traci
            x, y = traci.simulation.convert2D(edge_id, pos, lane_index)
        except Exception:
            try:
                # если convert2D недоступен — попробуем взять среднюю точку shape полосы
                shape = traci.lane.getShape(lane_id)
                if shape:
                    mid = len(shape) // 2
                    x, y = shape[mid]
            except Exception:
                # оставим x,y = 0.0 и продолжим
                pass

        poi_id = f"__acc_poi__{edge_id.replace('#', '_').replace(':', '_')}_{lane_index}_{self._safe_sim_time_int()}"

        try:
            # основная попытка с полными параметрами
            traci.poi.add(
                poi_id, x, y,
                color=self.marker_color,
                layer=self.marker_layer,
                type=self.marker_type,
                width=self.marker_w,
                height=self.marker_h
            )
        except Exception:
            # fallback: попытаемся добавить минимальную информацию
            try:
                traci.poi.add(poi_id, x, y, self.marker_color)
            except Exception:
                # если и это не работает — вернём пустой id
                return "", x, y
            try:
                traci.poi.setColor(poi_id, self.marker_color)
            except Exception:
                pass
            try:
                traci.poi.setType(poi_id, self.marker_type)
            except Exception:
                pass

        try:
            traci.poi.setParameter(poi_id, "name", self.marker_label)
        except Exception:
            # не критично, продолжим без установки имени
            pass

        return poi_id, x, y

    def _remove_marker(self, poi_id: Optional[str]) -> None:
        """
        Удаляет POI по идентификатору, если он существует.
        - Проверяет список traci.poi.getIDList() перед удалением, чтобы избежать исключения.
        - Игнорирует ошибки удаления.
        """
        if not poi_id:
            return
        try:
            # traci.poi.getIDList может бросить; в таком случае считаем, что POI не существует
            try:
                ids = traci.poi.getIDList()
            except Exception:
                ids = []
            if poi_id in ids:
                traci.poi.remove(poi_id)
        except Exception:
            pass

    # ---------- Публичные методы ----------
    def create_accident_at(
        self,
        lane_id: str,
        *,
        duration_steps: Optional[int] = None,
        pos_m: Optional[float] = None,
        mode: Optional[str] = None,
        ignore_max_concurrent: bool = False
    ) -> Optional[Accident]:
        """
        Создаёт аварийную запись вручную на заданной полосе lane_id.

        Поведение:
        - Проверяет существование полосы и исключает внутренние ребра (те, что начинаются с ':').
        - Учитывает ограничение max_concurrent, если ignore_max_concurrent=False.
        - Сохраняет предыдущее состояние полосы (speed/allowed).
        - В зависимости от режима (mode параметр или self.mode) применяет lane_block или пытается спавнить obstacle.
        - Добавляет GUI-маркер (если включён).
        - Возвращает объект Accident при успешном создании, иначе None.

        Аргументы:
        - duration_steps: явная длительность в шагах (если None — случай между min_dur и max_dur).
        - pos_m: позиция в метрах для маркера/спавна (если None — выбирается автоматически).
        - mode: переопределяет режим на один вызов ("lane_block" или "obstacle").
        - ignore_max_concurrent: если True — игнорировать ограничение по числу одновременных аварий.
        """
        # Не спавним на внутренних ребрах
        try:
            if not self._lane_exists(lane_id):
                return None
            if traci.lane.getEdgeID(lane_id).startswith(":"):
                return None
        except Exception:
            return None

        if (not ignore_max_concurrent) and len(self.active) >= self.max_concurrent:
            return None
        if lane_id in self.active:
            # уже есть авария на этой полосе
            return None

        dur = int(duration_steps) if duration_steps is not None else int(
            self.rng.randint(self.min_dur, self.max_dur))
        prev_speed, prev_allowed = self._store_prev_state(lane_id)
        step_idx = self._safe_sim_time_int()
        new_acc = Accident(
            lane_id=lane_id,
            start_step=step_idx,
            end_step=step_idx + dur,
            prev_max_speed=prev_speed,
            prev_allowed=prev_allowed,
        )

        use_mode = (mode or self.mode).lower()
        try:
            edge_id = traci.lane.getEdgeID(lane_id)
        except Exception:
            edge_id = ""

        if use_mode == "lane_block":
            # применяем блокировку полосы
            self._apply_lane_block(lane_id)
            pos_for_marker = self._safe_pos_on_lane(
                lane_id) if pos_m is None else float(pos_m)
            poi_id, x, y = self._add_marker(lane_id, edge_id, pos_for_marker)
            new_acc.marker_poi_id, new_acc.marker_x, new_acc.marker_y = poi_id, x, y

        elif use_mode == "obstacle":
            # пытаемся заспавнить препятствие; при неудаче — откат к lane_block
            try:
                veh_id, e_id, pos, l_idx = self._spawn_obstacle(
                    lane_id, dur, pos_override=pos_m)
                new_acc.obstacle_veh_id = veh_id
                new_acc.obstacle_edge_id = e_id
                new_acc.obstacle_pos = pos
                new_acc.obstacle_lane_index = l_idx
                poi_id, x, y = self._add_marker(lane_id, e_id, pos)
                new_acc.marker_poi_id, new_acc.marker_x, new_acc.marker_y = poi_id, x, y
            except Exception:
                # Фолбэк к lane_block
                self._apply_lane_block(lane_id)
                pos_for_marker = self._safe_pos_on_lane(
                    lane_id) if pos_m is None else float(pos_m)
                poi_id, x, y = self._add_marker(
                    lane_id, edge_id, pos_for_marker)
                new_acc.marker_poi_id, new_acc.marker_x, new_acc.marker_y = poi_id, x, y
        else:
            # неизвестный режим
            return None

        # регистрируем активную аварию
        self.active[lane_id] = new_acc
        return new_acc

    def clear_accident(self, lane_id: str) -> bool:
        """
        Очищает (завершает) аварию на полосе lane_id:
        - восстанавливает состояние полосы, если режим lane_block,
        - удаляет спавненное препятствие, если режим obstacle,
        - удаляет маркер.
        Возвращает True если что-то было удалено, иначе False.
        """
        acc = self.active.pop(lane_id, None)
        if not acc:
            return False
        if self.mode == "lane_block":
            self._restore_lane_state(acc)
        elif self.mode == "obstacle":
            if acc.obstacle_veh_id:
                self._despawn_obstacle(acc.obstacle_veh_id)
        self._remove_marker(acc.marker_poi_id)
        return True

    def clear_all(self) -> int:
        """
        Удаляет все активные аварии и возвращает число удалённых.
        Использует clear_accident для каждой записи.
        """
        cnt = 0
        for lane_id in list(self.active.keys()):
            if self.clear_accident(lane_id):
                cnt += 1
        return cnt

    def step(self, step_idx: int) -> None:
        """
        Основной шаг менеджера, вызывается каждый тик/шаг симуляции.

        Шаги:
        1) Закрываем завершившиеся аварии (step_idx >= acc.end_step) — восстанавливаем/деспавним/удаляем маркеры.
        2) С вероятностью self.prob (и если есть свободные слоты) создаём новую аварию:
           - выбираем полосу _pick_lane
           - создаём Accident и применяем поведение в зависимости от режима (lane_block/obstacle)
           - при ошибке spawn в режиме obstacle — фолбэк к lane_block
        """
        # 1) Закрываем завершившиеся аварии
        to_close = [lane for lane, acc in self.active.items()
                    if step_idx >= acc.end_step]
        for lane_id in to_close:
            acc = self.active.pop(lane_id, None)
            if not acc:
                continue
            if self.mode == "lane_block":
                self._restore_lane_state(acc)
            elif self.mode == "obstacle":
                if acc.obstacle_veh_id:
                    self._despawn_obstacle(acc.obstacle_veh_id)
            self._remove_marker(acc.marker_poi_id)

        # 2) Создаём новую (по вероятности)
        if len(self.active) >= self.max_concurrent:
            return
        if self.rng.random() >= self.prob:
            return

        lane_id = self._pick_lane()
        if not lane_id:
            return

        dur = int(self.rng.randint(self.min_dur, self.max_dur))
        prev_speed, prev_allowed = self._store_prev_state(lane_id)

        new_acc = Accident(
            lane_id=lane_id,
            start_step=step_idx,
            end_step=step_idx + dur,
            prev_max_speed=prev_speed,
            prev_allowed=prev_allowed,
        )

        try:
            edge_id = traci.lane.getEdgeID(lane_id)
        except Exception:
            edge_id = ""

        if self.mode == "lane_block":
            # Блокируем полосу и ставим маркер в безопасной точке
            self._apply_lane_block(lane_id)
            pos_for_marker = self._safe_pos_on_lane(lane_id)
            poi_id, x, y = self._add_marker(lane_id, edge_id, pos_for_marker)
            new_acc.marker_poi_id = poi_id
            new_acc.marker_x = x
            new_acc.marker_y = y

        elif self.mode == "obstacle":
            # Пытаемся заспавнить препятствие; при провале — фолбэк на lane_block
            try:
                veh_id, e_id, pos, l_idx = self._spawn_obstacle(lane_id, dur)
                new_acc.obstacle_veh_id = veh_id
                new_acc.obstacle_edge_id = e_id
                new_acc.obstacle_pos = pos
                new_acc.obstacle_lane_index = l_idx
                poi_id, x, y = self._add_marker(lane_id, e_id, pos)
                new_acc.marker_poi_id = poi_id
                new_acc.marker_x = x
                new_acc.marker_y = y
            except Exception:
                self._apply_lane_block(lane_id)
                pos_for_marker = self._safe_pos_on_lane(lane_id)
                poi_id, x, y = self._add_marker(
                    lane_id, edge_id, pos_for_marker)
                new_acc.marker_poi_id = poi_id
                new_acc.marker_x = x
                new_acc.marker_y = y

        self.active[lane_id] = new_acc

    def shutdown(self) -> None:
        """
        Отключение менеджера: восстанавливает все изменённые полосы, удаляет все препятствия и маркеры.
        Очищает self.active.
        """
        for lane_id, acc in list(self.active.items()):
            if self.mode == "lane_block":
                self._restore_lane_state(acc)
            elif self.mode == "obstacle":
                if acc.obstacle_veh_id:
                    self._despawn_obstacle(acc.obstacle_veh_id)
            self._remove_marker(acc.marker_poi_id)
        self.active.clear()

    # ---------- Внутренние ----------
    def _pick_lane(self) -> Optional[str]:
        """
        Выбирает случайную полосу из lane_candidates, исключая те, на которых уже есть аварии.
        Возвращает None, если свободных полос нет.
        """
        free = [l for l in self.lane_candidates if l not in self.active]
        if not free:
            return None
        return self.rng.choice(free)