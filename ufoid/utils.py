import logging
from pathlib import Path

from rich.logging import RichHandler

ROOT_PATH = Path(__file__).parent.parent
LOG_LOCATION = "./var/log/app.log"
LOG_LEVEL = logging.INFO
LOG_FORMAT = "[%(asctime)s]" " %(message)s" #[%(pathname)s:%(lineno)d]"


class RelativePathFormatter(logging.Formatter):
    """A custom log formatter that makes file paths relative to the project's root directory.

    Args:
        logging.Formatter: The base formatter for log messages.

    Methods:
        format(record): Format the log record, making file paths relative to the project's root directory.
    """

    def format(self, record):
        try:
            record.pathname = str(Path(record.pathname).relative_to(ROOT_PATH))
        except ValueError:
            pass
        return super().format(record)


def get_logger(__name__):
    """Create and configure a custom logger with both file and console handlers.

    Args:
        __name__ (str): The name of the logger, typically obtained by using __name__ from the calling module.

    Returns:
        logging.Logger: A configured logger instance.
    """

    def _get_custom_handler(handler, format, filter=None):
        handler.setFormatter(RelativePathFormatter(format))
        if filter:
            handler.addFilter(filter)
        return handler

    log_dir = Path(LOG_LOCATION).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    _logger = logging.getLogger(__name__)
    _logger.setLevel(LOG_LEVEL)

    file_handler = logging.FileHandler(LOG_LOCATION)
    custom_file_handler = _get_custom_handler(file_handler, LOG_FORMAT)
    _logger.addHandler(custom_file_handler)

    console_handler = _get_custom_handler(
        RichHandler(show_time=False, show_path=False), LOG_FORMAT
    )
    _logger.addHandler(console_handler)

    return _logger
