import logging
import os
from typing import Optional
from logging.handlers import RotatingFileHandler

class LoggingConfig:
    DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    DEFAULT_LEVEL = logging.DEBUG
    MAX_BYTES = 10 * 1024 * 1024  # 10MB
    BACKUP_COUNT = 5

    def __init__(
        self,
        log_level: int = DEFAULT_LEVEL,
        log_format: str = DEFAULT_FORMAT,
        log_file: Optional[str] = None
    ):
        self.log_level = log_level
        self.log_format = log_format
        self.log_file = log_file
        self.loggers_to_silence = [
            'aiosqlite',
            # 'sentence_transformers',
            'sqlalchemy.engine',
            'urllib3.connectionpool',
            'asyncio'
        ]

    def _create_formatter(self) -> logging.Formatter:
        """
        Create a formatter for the log messages.
        Returns:
            logging.Formatter: The formatter for the log messages.
        """
        return logging.Formatter(self.log_format)

    def _setup_file_handler(
            self,
            formatter: logging.Formatter
    ) -> Optional[RotatingFileHandler]:
        """
        Setup file handler if log_file is provided.
        Args:
            formatter: The formatter to use for the file handler.
        Returns:
            Optional[RotatingFileHandler]: File handler if log_file is provided, otherwise None.
        """
        if not self.log_file:
            return None
        
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        file_handler = RotatingFileHandler(
            self.log_file,
            maxBytes=self.MAX_BYTES,
            backupCount=self.BACKUP_COUNT
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(self.log_level)
        return file_handler

    def _setup_console_handler(
            self,
            formatter: logging.Formatter
    ) -> logging.StreamHandler:
        """
        Setup console handler.
        Args:
            formatter: The formatter to use for the console handler.
        Returns:
            logging.StreamHandler: The console handler.
        """
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(self.log_level)
        return console_handler

    def _silence_loggers(self) -> None:
        """
        Silence loggers that are too verbose.
        Returns:
            None
        """
        for logger_name in self.loggers_to_silence:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

    def configure(self) -> None:
        """
        Setup the logging configuration.
        Returns
            None
        """
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)

        root_logger.handlers.clear()

        formatter = self._create_formatter()

        console_handler = self._setup_console_handler(formatter)
        root_logger.addHandler(console_handler)

        file_handler = self._setup_file_handler(formatter)
        if file_handler:
            root_logger.addHandler(file_handler)

        self._silence_loggers()

def configure_logging(
    log_level: int = LoggingConfig.DEFAULT_LEVEL,
    log_format: str = LoggingConfig.DEFAULT_FORMAT,
    log_file: Optional[str] = None
) -> None:
    """
    Configure the logger system.
    Args:
        log_level: Logger's level
        log_format: Logger's format
        log_file: Logger's file path
    """
    config = LoggingConfig(log_level, log_format, log_file)
    config.configure()
