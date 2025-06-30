import logging
from logging.handlers import RotatingFileHandler
from utils.logger import setup_logger

def test_logger(tmp_path):
    for debug_mode in [True, False]:
        mode = "debug" if debug_mode else "normal"
        log_dir = tmp_path / f"outputs/logs_{mode}"
        log_file = log_dir / f"testlogger_{mode}.log"

        logger = setup_logger(f"testlogger_{mode}", log_dir, debug_mode=debug_mode)
        print(f"\n>>> MODE : {mode.upper()}")
        logger.debug("Message DEBUG (visible si debug_mode=True)")
        logger.info("Message INFO")
        logger.warning("Message WARNING")
        logger.error("Message ERROR")
        logger.critical("Message CRITICAL")

        # Vérifications
        assert logger.name == f"testlogger_{mode}"
        assert log_dir.exists()
        assert log_file.exists()

        handler_types = [type(h) for h in logger.handlers]
        assert logging.StreamHandler in handler_types
        assert RotatingFileHandler in handler_types

    
    
