from utils.test_utils import *
from tqdm import tqdm
import csv
from utils import q_learning, sumo_utils
import libsumo as traci
import os
import random
import sys
import numpy as np
from pathlib import Path
from utils.accident_utils import AccidentManager

USING_LIBSUMO = True

def write_csv_header(path, fields):
    """
    Записывает заголовок CSV-файла и возвращает открытый файловый объект и writer.
    
    Аргументы:
    - path (str): путь к файлу, который будет создан/перезаписан.
    - fields (list[str]): список имён столбцов для заголовка CSV.
    
    Возвращает:
    - (file, csv.DictWriter): кортеж с открытым файловым объектом (в режиме записи)
      и объектом csv.DictWriter, готовым для записи строк-словарей.
    
    Примечание:
    - Файл открывается с кодировкой utf-8 и newline="" чтобы корректно писать CSV в разных ОС.
    """
    f = open(path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    return f, writer


def dedup(seq):
    """
    Удаляет дубликаты из последовательности, сохраняя исходный порядок элементов.
    
    Аргументы:
    - seq (iterable): входная последовательность (список, кортеж и т.д.).
    
    Возвращает:
    - list: новый список с теми же элементами, но без повторов в порядке первого вхождения.
    
    Реализация:
    - Используется множество `seen` для отслеживания уже встреченных значений.
    """
    seen = set()   # множество для проверки уже встретившихся элементов
    out = []       # выходной список без дубликатов
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def sum_halting_on_lanes(lane_ids):
    """
    Суммирует количество остановившихся (halting) транспортных средств по указанным полосам.
    
    Аргументы:
    - lane_ids (iterable[str]): список идентификаторов полос (lane IDs) в SUMO.
    
    Возвращает:
    - int: суммарное количество остановленных автомобилей по всем переданным полосам.
    
    Использует:
    - traci.lane.getLastStepHaltingNumber(lid) для получения числа остановленных на каждой полосе.
    """
    total = 0
    for lid in lane_ids:
        total += traci.lane.getLastStepHaltingNumber(lid)
    return total


def sum_waiting_time_on_lanes(lane_ids):
    """
    Суммирует общее время ожидания (waiting time) по указанным полосам.
    
    Аргументы:
    - lane_ids (iterable[str]): список идентификаторов полос (lane IDs).
    
    Возвращает:
    - float: суммарное время ожидания по всем полосам (в секундах), возвращаемое traci.
    
    Замечания:
    - Возвращаемое значение зависит от того, как SUMO считает и агрегирует waiting time для полос.
    """
    total = 0.0
    for lid in lane_ids:
        total += traci.lane.getWaitingTime(lid)
    return total


def weighted_mean_speed_on_lanes(lane_ids):
    """
    Вычисляет средневзвешенную по числу автомобилей скорость для набора полос.
    
    Аргументы:
    - lane_ids (iterable[str]): список идентификаторов полос (lane IDs).
    
    Возвращает:
    - float: средняя скорость по всем переданным полосам, взвешенная по количеству автомобилей.
      Если автомобилей не найдено (все нулевые), возвращается 0.0.
    
    Логика:
    - Для каждой полосы запрашивается количество автомобилей и средняя скорость на полосе.
      Скорость умножается на число автомобилей и добавляется в аккумулятор; затем делится
      на общее число автомобилей, чтобы получить средневзвешенное значение.
    """
    total_veh = 0
    acc = 0.0
    for lid in lane_ids:
        n = traci.lane.getLastStepVehicleNumber(lid)  # число автомобилей на полосе в последний шаг
        if n > 0:
            acc += traci.lane.getLastStepMeanSpeed(lid) * n
            total_veh += n
    return (acc / total_veh) if total_veh > 0 else 0.0