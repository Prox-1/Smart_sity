# utils/accident_utils.py
from __future__ import annotations
from typing import Iterable

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import random

import traci
from traci import constants as tc  # noqa: F401


@dataclass
class Accident:
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
    Псевдослучайные "аварии" на полосах + маркеры в SUMO-GUI.

    Режимы:
      - lane_block: полоса временно закрывается (maxSpeed=0, запрет классов).
      - obstacle: спавним "препятствие" (ТС) и фиксируем его stop на полосе.

    Устойчиво к ограничениям vClass на полосах; при невозможности спавна препятствия — фолбэк к lane_block.
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
        # Исключаем внутренние ребра (начинаются с ":")
        self.lane_candidates = [
            l for l in lane_ids
            if not traci.lane.getEdgeID(l).startswith(":")
        ]
        self.used_vclasses = set(used_vclasses)
        self.rng = rng
        self.mode = mode
        self.prob = prob_per_step
        self.min_dur = min_duration_steps
        self.max_dur = max_duration_steps
        self.max_concurrent = max_concurrent
        self.min_margin = min_margin_from_ends_m

        self.enable_markers = enable_markers
        self.marker_color = marker_color
        self.marker_layer = marker_layer
        self.marker_w, self.marker_h = marker_size
        self.marker_type = marker_type
        self.marker_label = marker_label

        self.active: Dict[str, Accident] = {}  # lane_id -> Accident
        # маршруты для obstacle (1 ребро)
        self._edge_route_id: Dict[str, str] = {}
        self._vclass_vtype: Dict[str, str] = {}  # кэш vtype по vClass

        # Набор предпочтительных vClass для подбора, если allowed пуст
        self._vclass_preference = [
            "passenger", "bus", "delivery", "authority", "taxi",
            "motorcycle", "evehicle", "emergency", "truck", "trailer",
            "coach", "tram", "rail_urban"
        ]

    # ---------- Вспомогательные ----------

    def _lane_index_from_id(self, lane_id: str) -> int:
        try:
            return int(lane_id.rsplit("_", 1)[1])
        except Exception:
            return 0

    def _safe_pos_on_lane(self, lane_id: str) -> float:
        length = float(traci.lane.getLength(lane_id))
        if length <= 1.0:
            return max(0.0, length * 0.5)
        # Используем безопасные отступы, адаптивно к длине
        margin = max(0.5, min(self.min_margin, length * 0.2))
        min_pos = margin
        max_pos = max(min_pos + 0.1, length - margin)
        pos = float(self.rng.uniform(min_pos, max_pos))
        # Жёстко гарантируем, что не выйдем за пределы
        pos = min(max(0.1, pos), max(0.2, length - 0.2))
        return pos

    def _store_prev_state(self, lane_id: str) -> Tuple[float, Optional[List[str]]]:
        prev_speed = traci.lane.getMaxSpeed(lane_id)
        allowed = traci.lane.getAllowed(lane_id)  # [] => "все разрешены"
        prev_allowed = None if len(allowed) == 0 else allowed
        return prev_speed, prev_allowed

    def _restore_lane_state(self, acc: Accident) -> None:
        try:
            traci.lane.setMaxSpeed(acc.lane_id, acc.prev_max_speed)
        except Exception:
            pass
        try:
            if acc.prev_allowed is None:
                traci.lane.setAllowed(acc.lane_id, [])
            else:
                traci.lane.setAllowed(acc.lane_id, acc.prev_allowed)
        except Exception:
            pass

    def _ensure_edge_route(self, edge_id: str) -> str:
        rid = self._edge_route_id.get(edge_id)
        if rid is not None:
            return rid
        rid = f"__acc_route__{edge_id}"
        try:
            traci.route.add(rid, [edge_id])
        except traci.TraCIException:
            pass
        self._edge_route_id[edge_id] = rid
        return rid

    def _get_allowed_vclass_for_lane(self, lane_id: str) -> Optional[str]:
        """
        Подбирает vClass, разрешённый на КОНКРЕТНОЙ полосе.
        Если lane.getAllowed() непуст, берём из него.
        Иначе берём первый из предпочтений, которого нет в disallowed.
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
        vt = self._vclass_vtype.get(vclass)
        if vt:
            return vt
        vt = f"__acc_vtype__{vclass}"
        try:
            # Создаём на базе DEFAULT_VEHTYPE при наличии
            base_types = set(traci.vehicletype.getIDList())
            if "DEFAULT_VEHTYPE" in base_types:
                try:
                    traci.vehicletype.copy("DEFAULT_VEHTYPE", vt)
                except Exception:
                    # Возможно уже создан
                    pass
            else:
                # Минимальное объявление типа
                try:
                    traci.vehicletype.add(vt)
                except Exception:
                    pass

            # Назначаем целевой vClass
            try:
                traci.vehicletype.setVehicleClass(vt, vclass)
            except Exception:
                # Если не удалось установить vClass, пусть будет как есть
                pass

            # Небольшой визуальный твик (необязательно)
            try:
                traci.vehicletype.setColor(vt, (255, 128, 0, 255))
                traci.vehicletype.setLength(vt, 4.5)
                traci.vehicletype.setWidth(vt, 2.0)
            except Exception:
                pass
        finally:
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
        Возвращает словарь {edge_id: impact в [0,1]}, где impact ~ доля поражённых полос.
        Масштабирует по типу режима: lane_block суровее, obstacle мягче.
        Только для edge_ids из входа (остальные игнорируются).
        """
        edge_ids = list(edge_ids)
        impacts: Dict[str, float] = {e: 0.0 for e in edge_ids}
        if not self.active or not edge_ids:
            return impacts

        # какие полосы поражены у каждого ребра
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

        # выберем коэффициент тяжести по текущему режиму менеджера
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

    def _spawn_obstacle(self, lane_id: str, duration: int) -> Tuple[str, str, float, int]:
        """
        Спавнит ТС с подходящим vClass для данной полосы, ставит его на stop.
        При невозможности — поднимет исключение (внешний код сделает фолбэк).
        """
        edge_id = traci.lane.getEdgeID(lane_id)
        lane_index = self._lane_index_from_id(lane_id)
        length = float(traci.lane.getLength(lane_id))
        pos = self._safe_pos_on_lane(lane_id)
        pos = min(max(0.1, pos), max(0.2, length - 0.2))

        vclass = self._get_allowed_vclass_for_lane(lane_id)
        if not vclass:
            raise RuntimeError("No suitable vClass for this lane")

        vtype_id = self._ensure_obstacle_vtype(vclass)
        route_id = self._ensure_edge_route(edge_id)
        veh_id = f"__acc_{lane_id.replace('#', '_').replace(':', '_')}_{int(traci.simulation.getTime())}"

        # Добавляем ТС типа vtype_id с маршрутом из одного ребра
        traci.vehicle.add(veh_id, route_id, typeID=vtype_id)

        # Перемещаем на нужную полосу/позицию и фиксируем
        try:
            traci.vehicle.moveTo(veh_id, lane_id, pos)
        except Exception:
            # Если moveTo не сработал, продолжим через stop
            pass

        # Жёсткий stop
        try:
            traci.vehicle.setStop(
                veh_id,
                edgeID=edge_id,
                pos=pos,
                laneIndex=lane_index,
                duration=max(1, int(duration)),
                flags=0
            )
        except Exception:
            # fallback — вручную "заморозим" машину
            try:
                traci.vehicle.setSpeedMode(veh_id, 0)
                traci.vehicle.setSpeed(veh_id, 0.0)
            except Exception:
                pass

        # На всякий случай фиксируем режим скорости = ручной 0
        try:
            traci.vehicle.setSpeedMode(veh_id, 0)
            traci.vehicle.setSpeed(veh_id, 0.0)
        except Exception:
            pass

        return veh_id, edge_id, pos, lane_index

    def _despawn_obstacle(self, veh_id: str) -> None:
        try:
            traci.vehicle.remove(veh_id)
        except Exception:
            pass

    # ---------- GUI POI-маркеры ----------

    def _add_marker(self, lane_id: str, edge_id: str, pos: float) -> Tuple[str, float, float]:
        if not self.enable_markers:
            return "", float("nan"), float("nan")

        lane_index = self._lane_index_from_id(lane_id)

        x = y = 0.0
        try:
            x, y = traci.simulation.convert2D(edge_id, pos, lane_index)
        except Exception:
            try:
                shape = traci.lane.getShape(lane_id)
                if shape:
                    mid = len(shape) // 2
                    x, y = shape[mid]
            except Exception:
                pass

        poi_id = f"__acc_poi__{edge_id.replace('#', '_').replace(':', '_')}_{lane_index}_{int(traci.simulation.getTime())}"

        try:
            traci.poi.add(
                poi_id, x, y,
                color=self.marker_color,
                layer=self.marker_layer,
                type=self.marker_type,
                width=self.marker_w,
                height=self.marker_h
            )
        except Exception:
            traci.poi.add(poi_id, x, y, self.marker_color)
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
            pass

        return poi_id, x, y

    def _remove_marker(self, poi_id: Optional[str]) -> None:
        if not poi_id:
            return
        try:
            traci.poi.remove(poi_id)
        except Exception:
            pass

    # ---------- Публичные методы ----------

    def step(self, step_idx: int) -> None:
        # 1) Закрываем завершившиеся аварии
        to_close = [lane for lane, acc in self.active.items()
                    if step_idx >= acc.end_step]
        for lane_id in to_close:
            acc = self.active.pop(lane_id)
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

        edge_id = traci.lane.getEdgeID(lane_id)

        if self.mode == "lane_block":
            # Блокируем полосу и ставим маркер в безопасной точке
            self._apply_lane_block(lane_id)
            pos_for_marker = self._safe_pos_on_lane(lane_id)
            poi_id, x, y = self._add_marker(lane_id, edge_id, pos_for_marker)
            new_acc.marker_poi_id = poi_id
            new_acc.marker_x = x
            new_acc.marker_y = y

        elif self.mode == "obstacle":
            # Пытаемся заспавнить препятствие; при провале — фолбэк
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
        free = [l for l in self.lane_candidates if l not in self.active]
        if not free:
            return None
        return self.rng.choice(free)
