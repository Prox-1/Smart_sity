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
        # Обёртки для безопасности: отфильтруем полосы, на которых треги не падают
        self.lane_candidates = []
        for l in lane_ids:
            try:
                # если lane не существует — getEdgeID бросит; пропускаем
                e = traci.lane.getEdgeID(l)
                if not e.startswith(":"):
                    self.lane_candidates.append(l)
            except Exception:
                # пропустим проблемные полосы
                continue

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
        self._veh_seq = 0
        self.episode = -1
        self.counter = 0
        self.spawned_ids = set()
        self.pending_ops = []
        self.cooldown_until = 0

    # ---------- Безопасные проверки ----------

    def _veh_exists(self, vid: str) -> bool:
        try:
            return vid in set(traci.vehicle.getIDList())
        except Exception:
            return False

    def _lane_exists(self, lane_id: str) -> bool:
        try:
            _ = traci.lane.getLength(lane_id)
            return True
        except Exception:
            return False

    def _edge_exists(self, edge_id: str) -> bool:
        try:
            _ = traci.edge.getLaneNumber(edge_id)
            return True
        except Exception:
            return False

    def _poi_exists(self, poi_id: str) -> bool:
        try:
            return poi_id in traci.poi.getIDList()
        except Exception:
            return False

    def _safe_sim_time_int(self) -> int:
        try:
            t = traci.simulation.getTime()
            return int(t)
        except Exception:
            return 0

    # ---------- Вспомогательные ----------

    def _lane_index_from_id(self, lane_id: str) -> int:
        try:
            return int(lane_id.rsplit("_", 1)[1])
        except Exception:
            return 0

    def _safe_pos_on_lane(self, lane_id: str) -> float:
        try:
            length = float(traci.lane.getLength(lane_id))
        except Exception:
            # если не можем получить длину — вернём 0.5 (безопасно)
            return 0.5
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
                traci.lane.setAllowed(acc.lane_id, [])
            else:
                traci.lane.setAllowed(acc.lane_id, acc.prev_allowed)
        except Exception:
            pass

    def _ensure_edge_route(self, edge_id: str) -> str:
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
            base_types = set()
            try:
                base_types = set(traci.vehicletype.getIDList())
            except Exception:
                base_types = set()
            if "DEFAULT_VEHTYPE" in base_types:
                try:
                    traci.vehicletype.copy("DEFAULT_VEHTYPE", vt)
                except Exception:
                    # Возможно уже создан или не поддерживается
                    pass
            else:
                try:
                    # Если add не поддерживает параметры — просто пробуем
                    traci.vehicletype.add(vt)
                except Exception:
                    pass

            try:
                traci.vehicletype.setVehicleClass(vt, vclass)
            except Exception:
                pass

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
        edge_id = traci.lane.getEdgeID(lane_id)
        lane_index = self._lane_index_from_id(lane_id)
        length = float(traci.lane.getLength(lane_id))
        pos = self._safe_pos_on_lane(
            lane_id) if pos_override is None else float(pos_override)
        pos = min(max(0.1, pos), max(0.2, length - 0.2))
        vclass = self._get_allowed_vclass_for_lane(lane_id)
        if not vclass:
            raise RuntimeError("No suitable vClass for this lane")
        vtype_id = self._ensure_obstacle_vtype(vclass)
        route_id = self._ensure_edge_route(edge_id)
        self._veh_seq += 1
        veh_id = f"__acc_{lane_id.replace('#', '_').replace(':', '_')}_{int(traci.simulation.getTime())}_{self._veh_seq}"
        # Add with robust fallback
        traci.vehicle.add(veh_id, route_id, typeID=vtype_id)
        # Validate laneIndex bounds
        try:
            nlanes = int(traci.edge.getLaneNumber(edge_id))
            lane_index = max(0, min(lane_index, max(0, nlanes - 1)))
        except Exception:
            lane_index = max(0, lane_index)
        try:
            if self._veh_exists(veh_id):
                traci.vehicle.moveTo(veh_id, lane_id, pos)
        except Exception:
            pass
        try:
            if self._veh_exists(veh_id):
                traci.vehicle.setStop(
                    veh_id,
                    edgeID=edge_id,
                    pos=pos,
                    laneIndex=lane_index,
                    duration=max(1, int(duration)),
                    flags=0
                )
        except Exception:
            try:
                if self._veh_exists(veh_id):
                    traci.vehicle.setSpeedMode(veh_id, 0)
                    traci.vehicle.setSpeed(veh_id, 0.0)
            except Exception:
                pass
        try:
            if self._veh_exists(veh_id):
                traci.vehicle.setSpeedMode(veh_id, 0)
                traci.vehicle.setSpeed(veh_id, 0.0)
        except Exception:
            pass
        return veh_id, edge_id, pos, lane_index

    def _despawn_obstacle(self, veh_id: str) -> None:
        try:
            if self._veh_exists(veh_id):
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
            # convert2D иногда бросает — ловим
            x, y = traci.simulation.convert2D(edge_id, pos, lane_index)
        except Exception:
            try:
                shape = traci.lane.getShape(lane_id)
                if shape:
                    mid = len(shape) // 2
                    x, y = shape[mid]
            except Exception:
                # оставим x,y = 0.0
                pass

        poi_id = f"__acc_poi__{edge_id.replace('#', '_').replace(':', '_')}_{lane_index}_{self._safe_sim_time_int()}"

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
            # fallback: попытаемся добавить минимально
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
            pass

        return poi_id, x, y

    def _remove_marker(self, poi_id: Optional[str]) -> None:
        if not poi_id:
            return
        try:
            # Если poi не существует — remove бросит; проверим список
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
            self._apply_lane_block(lane_id)
            pos_for_marker = self._safe_pos_on_lane(
                lane_id) if pos_m is None else float(pos_m)
            poi_id, x, y = self._add_marker(lane_id, edge_id, pos_for_marker)
            new_acc.marker_poi_id, new_acc.marker_x, new_acc.marker_y = poi_id, x, y

        elif use_mode == "obstacle":
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
                # Фолбэк
                self._apply_lane_block(lane_id)
                pos_for_marker = self._safe_pos_on_lane(
                    lane_id) if pos_m is None else float(pos_m)
                poi_id, x, y = self._add_marker(
                    lane_id, edge_id, pos_for_marker)
                new_acc.marker_poi_id, new_acc.marker_x, new_acc.marker_y = poi_id, x, y
        else:
            return None

        self.active[lane_id] = new_acc
        return new_acc

    def clear_accident(self, lane_id: str) -> bool:
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
        cnt = 0
        for lane_id in list(self.active.keys()):
            if self.clear_accident(lane_id):
                cnt += 1
        return cnt

    def step(self, step_idx: int) -> None:
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
