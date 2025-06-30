from functools import wraps
from time import perf_counter
from typing import Callable, Any, TypeVar, ParamSpec
import numpy as np
from utils.logger import setup_logger
from utils.config import DEBUG_MODE

# Generic Typage
P = ParamSpec("P")
R = TypeVar("R")

def get_logger():
    return setup_logger(debug_mode=DEBUG_MODE)


# Log execution time
def log_time(function: Callable[P, R]) -> Callable[P, R]:
    @wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        logger = get_logger()
        start = perf_counter()
        result = function(*args, **kwargs)
        end = perf_counter()
        logger.debug(f'\n[TIMER] "{function.__name__}" executed in {end - start:.4f} seconds')
        return result
    return wrapper


# Log function call
def format_arg(arg: Any) -> str:
    if isinstance(arg, np.ndarray):
        return f"ndarray(shape={arg.shape}, dtype={arg.dtype})"
    elif isinstance(arg, (list, tuple)) and len(arg) > 10:
        return f"{type(arg).__name__}(len={len(arg)})"
    elif isinstance(arg, dict):
        return f"dict(keys={list(arg.keys())})"
    else:
        return repr(arg)

def log_call(function: Callable[P, R]) -> Callable[P, R]:
    @wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        logger = get_logger()
        formatted_args = ", ".join(format_arg(a) for a in args)
        formatted_kwargs = ", ".join(f"{k}={format_arg(v)}" for k, v in kwargs.items())
        logger.debug(
            "[CALL] \"{name}\"\n"
            "          - args:   [{args}]\n"
            "          - kwargs: {{ {kwargs} }}".format(
                name=function.__name__,
                args=formatted_args if formatted_args else "None",
                kwargs=formatted_kwargs if formatted_kwargs else "None"
            )
        )
        return function(*args, **kwargs)
    return wrapper
