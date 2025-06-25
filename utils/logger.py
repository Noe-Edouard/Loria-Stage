import logging
from pathlib import Path
import time
from functools import wraps
import colorlog  # Pour les couleurs en console

BASE_DIR = Path(__file__).resolve().parent.parent
DEBUG_MODE = True  # Active/desactive les logs

def init_logger(name="benchmark_logger", log_dir=BASE_DIR / "logs", level=logging.INFO):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / f"{name}.log"
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.hasHandlers():
        # Console handler avec couleurs
        ch = logging.StreamHandler()
        ch.setFormatter(colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)-8s:%(reset)s %(message)s",
            log_colors={
                'DEBUG':    'cyan',
                'INFO':     'green',
                'WARNING':  'yellow',
                'ERROR':    'red',
                'CRITICAL': 'bold_red',
            }
        ))
        logger.addHandler(ch)

        # File handler (sans couleur)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        ))
        logger.addHandler(fh)

    return logger


debug_log = init_logger('debug_logger')
# Ajouter un debug externe
def debug_logger(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        if DEBUG_MODE:
            # debug_log.info(f"[CALL] {function.__name__} called with args={args}, kwargs={kwargs}")
            start = time.perf_counter()
            result = function(*args, **kwargs)
            end = time.perf_counter()
            debug_log.info(f"[TIMER] {function.__name__} executed in {end - start:.4f} seconds")
            return result
        else:
            return function(*args, **kwargs)
    return wrapper
