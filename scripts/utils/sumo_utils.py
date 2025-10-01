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
    max_dur = 9999
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


def get_tls_controlled_edges(tls_id):
    controlled_lanes = traci.trafficlight.getControlledLanes(
        tls_id)
    controlled_edges = set()
    for lane_id in controlled_lanes:
        controlled_edges.add(traci.lane.getEdgeID(lane_id))
    return controlled_edges
