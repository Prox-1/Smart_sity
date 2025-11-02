from traci import constants as tc
import random
import traci


def get_all_tls_phases(tls_id):
    """
    Возвращает список всех фаз для указанного светофора.
    Каждая фаза представлена как строка RYG-состояний.
    """
    logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)
    phases_list = []
    for phase in logic[0].phases:
        phases_list.append(phase.state)
    return phases_list


def set_phase_duration_by_action(tls_id: str, action_str: int):
    """
    Применяет действие (изменение длительности) к текущей фазе светофора
    с использованием traci.trafficlight.setPhaseDuration().
    Args:
        tls_id (str): ID светофора.
        action_str (str): Строковое представление действия, например, '+20', '-10', '0'.
                          Предполагает изменение текущей оставшейся длительности фазы.
    """
    current_remaining_duration = traci.trafficlight.getPhaseDuration(tls_id)
    current_phase_index = traci.trafficlight.getPhase(tls_id)
    all_logics_definitions = traci.trafficlight.getCompleteRedYellowGreenDefinition(
        tls_id)
    min_dur = 0
    max_dur = 180
    if all_logics_definitions:
        current_program_id = traci.trafficlight.getProgram(tls_id)
        active_logic = None
        for logic_def in all_logics_definitions:
            if logic_def.programID == current_program_id:
                active_logic = logic_def
                break
        if active_logic and 0 <= current_phase_index < len(active_logic.phases):
            current_phase_object = active_logic.phases[current_phase_index]
            min_dur = current_phase_object.minDur
            max_dur = current_phase_object.maxDur
        else:
            pass
            # print(
            #   f"Warning: Could not find active logic or phase object for TLS_ID: {tls_id}, Phase Index: {current_phase_index}. Using      default min/max durations.")
    else:
        pass
        # print(
        #   f"Warning: No traffic light logic definitions found for TLS_ID: {tls_id}. Using default min/max durations.")

    change_value = int(action_str)
    new_desired_duration = current_remaining_duration + change_value  # type: ignore
    final_duration = max(min_dur, min(new_desired_duration, max_dur))

    traci.trafficlight.setPhaseDuration(tls_id, final_duration)

    # print(f"TLS {tls_id}: Action '{action_str}'. Old remaining: {current_remaining_duration:.1f}s, New set duration: {final_duration:.1f}s (min:{min_dur:.1f}s, max:{max_dur:.1f}s)")


def set_phase_duration_for_new_phase(tls_id: str, delta_sec: int):
    """
    Вызывать сразу ПОСЛЕ перехода на новую фазу. Устанавливает итоговую длительность этой фазы
    как (базовая_длительность + delta), зажимая в [minDur, maxDur]. Не трогает фазу в середине.
    """

    current_phase_index = traci.trafficlight.getPhase(tls_id)
    all_logics = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)
    if not all_logics:
        return

    current_program_id = traci.trafficlight.getProgram(tls_id)
    active_logic = next(
        (lg for lg in all_logics if lg.programID == current_program_id), None)

    if not active_logic or not (0 <= current_phase_index < len(active_logic.phases)):
        return

    ph = active_logic.phases[current_phase_index]
    # duration может быть не задан, тогда берем minDur
    base = getattr(ph, "duration", ph.minDur)
    min_dur = ph.minDur
    max_dur = ph.maxDur if ph.maxDur > 0 else max(
        base, min_dur)  # на всякий случай
    desired_total = base + int(delta_sec)
    total = max(min_dur, min(desired_total, max_dur))
    # ВАЖНО: setPhaseDuration задает ОСТАВШЕЕСЯ время текущей фазы,
    # поэтому вызывать сразу после смены фазы — это фактически задать её общую длительность.
    traci.trafficlight.setPhaseDuration(tls_id, total)


def get_tls_controlled_edges(tls_id):
    controlled_lanes = traci.trafficlight.getControlledLanes(
        tls_id)
    controlled_edges = set()
    for lane_id in controlled_lanes:
        controlled_edges.add(traci.lane.getEdgeID(lane_id))
    return controlled_edges
