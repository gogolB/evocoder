# evocoder/utils/logger.py

import logging
import sys
from pathlib import Path
from typing import Dict, Optional

# Attempt to import settings for log level and potential log file path
try:
    from ..config import settings
except ImportError:
    # Fallback if run standalone or if settings module is not found via relative import
    # This should ideally not happen when used as part of the package.
    class MockSettings:
        LOG_LEVEL = "INFO"
        LOG_FILE_PATH = None # Default to no file logging if settings can't be imported
    settings = MockSettings()

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Cache for logger instances to avoid re-configuration
_loggers: Dict[str, logging.Logger] = {}

def setup_logger(
    name: str = "evocoder",
    level_str: Optional[str] = None,
    log_file: Optional[Path] = None,
    log_to_console: bool = True
) -> logging.Logger:
    """
    Configures and returns a logger instance.

    Args:
        name (str): The name for the logger. Defaults to "evocoder".
        level_str (Optional[str]): The logging level as a string (e.g., "DEBUG", "INFO").
                                   If None, uses LOG_LEVEL from settings.
        log_file (Optional[Path]): Path to a file to log messages to.
                                   If None, uses LOG_FILE_PATH from settings (if defined),
                                   otherwise only logs to console (if enabled).
        log_to_console (bool): Whether to output logs to the console. Defaults to True.

    Returns:
        logging.Logger: The configured logger instance.
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)

    # Determine log level
    effective_level_str = level_str or getattr(settings, "LOG_LEVEL", "INFO").upper()
    level = logging.getLevelName(effective_level_str)
    if not isinstance(level, int): # Check if getLevelName returned a valid level
        print(f"Warning: Invalid LOG_LEVEL '{effective_level_str}'. Defaulting to INFO.")
        level = logging.INFO
    logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # Clear existing handlers to avoid duplicate logs if re-configured
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console Handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File Handler
    effective_log_file = log_file or getattr(settings, "LOG_FILE_PATH", None)
    if effective_log_file:
        try:
            log_file_path = Path(effective_log_file)
            log_file_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
            file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info(f"Logging to file: {log_file_path}")
        except Exception as e:
            logger.error(f"Failed to set up file logger at {effective_log_file}: {e}", exc_info=False)
            # Fallback to console logging if file logging setup fails but console is enabled
            if not log_to_console: # If console was disabled and file failed, add a console handler
                emergency_console_handler = logging.StreamHandler(sys.stdout)
                emergency_console_handler.setFormatter(formatter)
                logger.addHandler(emergency_console_handler)
                logger.warning("File logging failed, emergency console logging enabled.")


    # Prevent messages from propagating to the root logger if handlers are added
    if logger.hasHandlers():
        logger.propagate = False
    
    _loggers[name] = logger
    return logger

# Default application logger instance
# Other modules can import this directly: from evocoder.utils.logger import app_logger
app_logger = setup_logger()

if __name__ == "__main__":
    # Example usage and test of the logger setup
    print("--- Testing logger.py ---")

    # Test default app_logger
    app_logger.debug("This is a debug message from app_logger (should not appear if level is INFO).")
    app_logger.info("This is an info message from app_logger.")
    app_logger.warning("This is a warning message from app_logger.")
    app_logger.error("This is an error message from app_logger.")
    
    try:
        1 / 0
    except ZeroDivisionError:
        app_logger.exception("An exception occurred (logged with traceback by app_logger).")

    print("\n--- Testing custom logger ---")
    # Create a custom logger
    custom_logger = setup_logger(
        name="my_module_logger",
        level_str="DEBUG", # Override level for this logger
        # log_file=Path("custom_module_test.log") # Example file logging
    )
    custom_logger.debug("This is a debug message from custom_logger (should appear).")
    custom_logger.info("This is an info message from custom_logger.")

    # print(f"\nCheck if custom_module_test.log was created and populated if log_file was uncommented.")
    print(f"Default logger level: {logging.getLevelName(app_logger.level)}")
    print(f"Custom logger level: {logging.getLevelName(custom_logger.level)}")

    # Test reconfiguration (should use cached logger and not add duplicate handlers)
    print("\n--- Testing logger reconfiguration (should use cached and not duplicate handlers) ---")
    reconfigured_app_logger = setup_logger(name="evocoder", level_str="DEBUG") # Default name is "evocoder"
    assert len(reconfigured_app_logger.handlers) <= 2 # Should not add more handlers if already configured
    reconfigured_app_logger.debug("This DEBUG message from reconfigured_app_logger should now appear.")
    print(f"Reconfigured app_logger handlers: {len(reconfigured_app_logger.handlers)}")


    print("\n--- Testing logger with explicit file ---")
    # Ensure data directory exists for test log file
    data_dir_for_log_test = Path(__file__).resolve().parent.parent.parent / "data"
    data_dir_for_log_test.mkdir(parents=True, exist_ok=True)
    test_log_file_path = data_dir_for_log_test / "test_output.log"
    
    if test_log_file_path.exists():
        test_log_file_path.unlink() # Clean up previous test log

    file_test_logger = setup_logger(
        name="file_test",
        level_str="INFO",
        log_file=test_log_file_path,
        log_to_console=False # Test file-only logging
    )
    file_test_logger.info("This message should go to the test_output.log file only.")
    file_test_logger.warning("This warning also goes to test_output.log.")
    
    if test_log_file_path.exists():
        print(f"Test log created at: {test_log_file_path}")
        with open(test_log_file_path, "r") as f:
            log_content = f.read()
            if "This message should go to the test_output.log file only." in log_content:
                print("File logging confirmed.")
            else:
                print("File logging FAILED or message not found.")
        # test_log_file_path.unlink() # Clean up
    else:
        print(f"Test log file NOT created at: {test_log_file_path}")

    print("\nLogger tests finished.")
