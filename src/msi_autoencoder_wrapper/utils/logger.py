import json
import logging
import sys
from typing import Any


class JsonFormatter(logging.Formatter):
    """
    Custom formatter to output logs in a structured JSON format.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the specified record as a JSON string.

        :param record: The log record to format.
        :type record: logging.LogRecord
        :return: The formatted JSON string.
        :rtype: str
        """
        # Base log record structure
        ## Extract standard telemetry fields
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "line": record.lineno,
            "message": record.getMessage(),
        }

        # Contextual metadata processing
        ## Check for extra dictionary attributes injected into the record
        if hasattr(record, "data") and isinstance(record.data, dict):
            log_data["data"] = record.data

        ## Check for exception information and format the traceback
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def get_custom_logger(
    name: str, level: int = logging.INFO, to_json: bool = False
) -> logging.Logger:
    """
    Factory function to initialize and configure a synchronized logger instance.

    :param name: The name of the logger, typically pass ``__name__``.
    :type name: str
    :param level: The logging threshold level, defaults to ``logging.INFO``.
    :type level: int, optional
    :param to_json: Flag to toggle structured JSON output, defaults to ``False``.
    :type to_json: bool, optional
    :return: Configured instance of logging.Logger.
    :rtype: logging.Logger
    """
    # Logger instance resolution
    ## Fetch the logger corresponding to the module hierarchical path
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 

    # Handler synchronization block
    ## Prevent double printing logs 
    logger.propagate = False
    ## Prevent handler duplication when modules are re-imported or reloaded
    if not logger.handlers:
        ### Stream routing configuration
        #### Route all log records directly to standard output stream
        handler = logging.StreamHandler(sys.stdout)

        ### Formatter selection logic
        if to_json:
            #### Structured logging deployment
            #### Assign the customized JSON serialization formatter
            formatter = JsonFormatter()
        else:
            #### Standard text logging deployment
            #### Define the immutable blueprint for console readability
            #### Modify this single string to propagate formatting updates globally
            text_format = (
                "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
            )
            formatter = logging.Formatter(text_format)

        ### Finalize handler pipeline
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger