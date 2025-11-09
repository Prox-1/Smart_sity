# utils/safe_traci.py
import traci
from traci import exceptions as traci_ex


def vehicle_exists(veh_id: str) -> bool:
    try:
        return veh_id in traci.vehicle.getIDList()
    except traci_ex.TraciException:
        # Для libsumo/разных версий может быть другой контракт — безопасный fallback
        try:
            _ = traci.vehicle.getParameter(
                veh_id, "type")  # просто чтобы проверить
            return True
        except Exception:
            return False
    except Exception:
        return False


def lane_exists(lane_id: str) -> bool:
    try:
        return lane_id in traci.lane.getIDList()
    except Exception:
        return False


def safe_traci_call(fn, *args, swallow=True, default=None, **kwargs):
    """
    Выполняет вызов traci-функции в try/except и возвращает default при ошибке.
    swallow=True — подавлять TraciException (логировать по желанию).
    """
    from traci import exceptions as traci_ex
    try:
        return fn(*args, **kwargs)
    except traci_ex.FatalTraCIError:
        # переломное состояние — пробрасываем, т.к. это серьёзно
        raise
    except Exception as e:
        if swallow:
            # можно подключить логирование сюда
            # print(f"safe_traci_call swallowed {e} for {fn.__name__} {args} {kwargs}")
            return default
        raise
