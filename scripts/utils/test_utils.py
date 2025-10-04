import traci
import os
import csv


def write_csv_header(path, fields):
    f = open(path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    return f, writer


def dedup(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def sum_halting_on_lanes(lane_ids):
    total = 0
    for lid in lane_ids:
        total += traci.lane.getLastStepHaltingNumber(lid)
    return total


def sum_waiting_time_on_lanes(lane_ids):
    total = 0.0
    for lid in lane_ids:
        total += traci.lane.getWaitingTime(lid)
    return total


def weighted_mean_speed_on_lanes(lane_ids):
    total_veh = 0
    acc = 0.0
    for lid in lane_ids:
        n = traci.lane.getLastStepVehicleNumber(lid)
        if n > 0:
            acc += traci.lane.getLastStepMeanSpeed(lid) * n
            total_veh += n
    return (acc / total_veh) if total_veh > 0 else 0.0
