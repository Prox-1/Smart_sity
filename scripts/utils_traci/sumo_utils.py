import traci


def get_all_tls_phases(tls_id):
    """
    Возвращает список всех фаз (в виде строк RYG) для заданного светофора (TLS).

    Описание:
        Функция использует TraCI API для получения полной
        дефиниции логики светофора и извлекает строковое
        представление состояния каждой фазы.

    Параметры:
        tls_id (str): идентификатор светофора в SUMO.

    Возвращает:
        List[str]: список фаз в формате RYG (например "rGrG"), в том порядке,
                   в котором они определены в логике программы светофора.
    """
    logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)
    phases_list = []

    # logic[0] — это первая (и часто единственная) логика/программа, у которой есть атрибут phases
    for phase in logic[0].phases:
        phases_list.append(phase.state)

    return phases_list


def set_phase_duration_by_action(tls_id: str, action_str: int):
    """
    Изменяет оставшуюся длительность текущей фазы в соответствии с действием.

    Описание:
        Функция получает оставшуюся длительность текущей фазы и корректирует её
        на величину action_str (например +5, -10, 0). При этом результат "защёлкивается"
        в диапазон [minDur, maxDur] для данной фазы, если логика светофора доступна.

    Особенности:
        - Если не найдена дефиниция логики или объект фазы — используются
          значения по умолчанию min_dur=0 и max_dur=180 (или оставшиеся значения min/max).
        - action_str ожидается как значение, которое можно привести к int (в секундах).

    Параметры:
        tls_id (str): ID светофора.
        action_str (int|str): изменение длительности (в секундах), например +5, -5 или 0.

    Возвращает:
        None

    Замечание:
        traci.trafficlight.setPhaseDuration устанавливает оставшееся время текущей фазы.
        Здесь мы предполагаем, что изменение применяется к оставшемуся времени (не к базовой длительности).
    """
    # Получаем оставшуюся длительность текущей фазы
    current_remaining_duration = traci.trafficlight.getPhaseDuration(tls_id)

    # Индекс текущей фазы
    current_phase_index = traci.trafficlight.getPhase(tls_id)

    # Получаем все определения логики для данного TLS (могут быть разные программы)
    all_logics_definitions = traci.trafficlight.getCompleteRedYellowGreenDefinition(
        tls_id)

    # Значения по умолчанию (на случай отсутствия информации о фазе)
    min_dur = 0
    max_dur = 180

    if all_logics_definitions:
        # Узнаём текущую программу (programID), чтобы найти соответствующую логику
        current_program_id = traci.trafficlight.getProgram(tls_id)
        active_logic = None

        # Находим активную логику по programID
        for logic_def in all_logics_definitions:
            if logic_def.programID == current_program_id:
                active_logic = logic_def
                break

        # Если логика найдена и индекс фазы валиден — извлекаем min/max длительности для текущей фазы
        if active_logic and 0 <= current_phase_index < len(active_logic.phases):
            current_phase_object = active_logic.phases[current_phase_index]
            min_dur = current_phase_object.minDur
            max_dur = current_phase_object.maxDur
        else:
            # Если активной логики или объекта фазы не найдено — используем значения по умолчанию
            pass

    else:
        # Если нет определений логики — используем значения по умолчанию
        pass

    # Приводим action к целому и вычисляем новую желаемую длительность как оставшееся + изменение
    change_value = int(action_str)
    new_desired_duration = current_remaining_duration + change_value  # type: ignore

    # Жёсткая граница в пределах [min_dur, max_dur]
    final_duration = max(min_dur, min(new_desired_duration, max_dur))

    # Устанавливаем новую оставшуюся длительность текущей фазы
    traci.trafficlight.setPhaseDuration(tls_id, final_duration)


def set_phase_duration_for_new_phase(tls_id: str, delta_sec: int):
    """
    Устанавливает итоговую длительность только что начавшейся фазы.

    Описание:
        Вызывать сразу ПОСЛЕ перехода на новую фазу. Эта функция берёт базовую
        длительность фазы (если явно указана) или minDur и добавляет delta_sec,
        затем обрезает результат по границам [minDur, maxDur] и устанавливает
        его как оставшееся время текущей фазы.

    Почему это важно:
        traci.trafficlight.setPhaseDuration устанавливает ОСТАВШЕЕЕСЯ время текущей фазы.
        Если вызвать эту функцию сразу после смены фазы — вы тем самым задаёте её общую
        длительность (base + delta_sec), что удобно для реализации действий агента,
        применяемых на момент смены фазы.

    Параметры:
        tls_id (str): идентификатор светофора.
        delta_sec (int): смещение в секундах, которое нужно добавить к базовой длительности фазы
                         (может быть отрицательным для уменьшения длительности).

    Возвращает:
        None
    """
    current_phase_index = traci.trafficlight.getPhase(tls_id)
    all_logics = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)

    if not all_logics:
        # Нет доступной логики светофора — ничего не делаем
        return

    current_program_id = traci.trafficlight.getProgram(tls_id)

    # Находим активную логику по programID (если есть)
    active_logic = next(
        (lg for lg in all_logics if lg.programID == current_program_id), None)

    if not active_logic or not (0 <= current_phase_index < len(active_logic.phases)):
        # Нет информации о фазе — ничего не делаем
        return

    ph = active_logic.phases[current_phase_index]

    # duration может быть не задан (0 или None) — в таком случае используем minDur как базу
    base = getattr(ph, "duration", ph.minDur)
    min_dur = ph.minDur
    # Если maxDur не задан или отрицателен, используем защитный максимум (минимум из base/min_dur)
    max_dur = ph.maxDur if ph.maxDur > 0 else max(base, min_dur)

    desired_total = base + int(delta_sec)
    total = max(min_dur, min(desired_total, max_dur))

    # Устанавливаем оставшееся время текущей фазы — при вызове сразу после смены фазы это задаст её общую длительность
    traci.trafficlight.setPhaseDuration(tls_id, total)


def get_tls_controlled_edges(tls_id):
    """
    Возвращает набор (set) идентификаторов ребер (edge IDs), контролируемых данным светофором.

    Описание:
        Функция использует traci.trafficlight.getControlledLanes для получения списка
        полос (lane IDs), а затем по каждой полосе получает её соответствующее ребро
        через traci.lane.getEdgeID.

    Параметры:
        tls_id (str): идентификатор светофора.

    Возвращает:
        Set[str]: множество идентификаторов ребер, которые находятся под управлением данного TLS.
    """
    controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
    controlled_edges = set()

    for lane_id in controlled_lanes:
        # для каждой полосы получаем её ребро и добавляем в множество (убирает дубликаты)
        controlled_edges.add(traci.lane.getEdgeID(lane_id))

    return controlled_edges