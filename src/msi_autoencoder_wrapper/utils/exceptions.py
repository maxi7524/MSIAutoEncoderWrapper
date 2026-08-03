"""
Domain-specific exceptions and standard error raising factories for the MSI library.
"""

from typing import List, NoReturn
from .logger import get_custom_logger

# Logger initialization
logger = get_custom_logger(__name__)


# =====================================================================
# Section: Exception Classes Definitions
# =====================================================================


class MSILibException(Exception):
    """
    Base exception class for all msi_lib runtime and configuration errors.
    """
    pass


class ProjectConfigError(MSILibException):
    """
    Raised when the project configuration is invalid, corrupted, or missing required attributes.
    """
    pass


class ExternalServiceError(MSILibException):
    """Raised when an external service rejects or cannot complete an operation."""

    pass


class DownloadLimitError(ExternalServiceError):
    """Raised when an external service reports an exhausted download quota."""

    pass


class ModelNotInitializedError(MSILibException):
    """
    Raised when executing operations on a model that has not been initialized or built.
    """
    pass


class IncompatibleInterfaceError(MSILibException):
    """
    Raised when type or structural mismatches occur between component managers.
    """
    pass


class WorkspaceConfigError(MSILibException):
    """
    Raised when workspace directory structures or paths are invalid.
    """
    pass


class ValidationError(MSILibException):
    """
    Raised when one or more required runtime validation criteria fail.
    """
    def __init__(self, missing_components: List[str]) -> None:
        """
        Initialize the validation error with a list of missing elements.

        :param missing_components: List of descriptions of missing components.
        :type missing_components: List[str]
        """
        self.missing_components = missing_components
        message = f"Validation failed. Missing or invalid components: {', '.join(missing_components)}"
        super().__init__(message)


# =====================================================================
# Section: Standardized Error Raising Factories
# =====================================================================

def raise_project_config_error(context_name: str, message: str) -> NoReturn:
    """
    Helper function to raise standardized project configuration errors.
    """
    formatted_msg = f"[{context_name.upper()} CONFIG ERROR] - {message}"
    logger.error("Project configuration invalid for %s: %s", context_name, message)
    raise ProjectConfigError(formatted_msg)


def raise_external_service_error(context_name: str, message: str) -> NoReturn:
    """Raise a contextualized external-service failure.

    :param context_name: External service or operation name.
    :type context_name: str
    :param message: Actionable failure description.
    :type message: str
    :raises ExternalServiceError: Always.
    """
    formatted_msg = f"[{context_name.upper()} SERVICE ERROR] - {message}"
    logger.error("External service failure for %s: %s", context_name, message)
    raise ExternalServiceError(formatted_msg)


def raise_download_limit_error(context_name: str, message: str) -> NoReturn:
    """Raise a contextualized external download-quota failure.

    :param context_name: External service name.
    :type context_name: str
    :param message: Actionable quota description.
    :type message: str
    :raises DownloadLimitError: Always.
    """
    formatted_msg = f"[{context_name.upper()} DOWNLOAD LIMIT] - {message}"
    logger.error("Download limit reached for %s: %s", context_name, message)
    raise DownloadLimitError(formatted_msg)


def raise_incompatible_interface_error(context_name: str, message: str) -> NoReturn:
    """
    Helper function to raise standardized interface/shape mismatch errors.
    """
    formatted_msg = f"[{context_name.upper()} INTERFACE ERROR] - {message}"
    logger.error("Incompatible interface detected in %s: %s", context_name, message)
    raise IncompatibleInterfaceError(formatted_msg)

def raise_validation_error(context_name: str, message: str) -> NoReturn:
    """
    Helper function to raise standardized validation exceptions with uniform printing and logging.

    :param context_name: The name of the context or module where validation failed.
    :type context_name: str
    :param message: The descriptive error message.
    :type message: str
    :raises ValidationError: Always raises ValidationError with formatted message.
    """
    # Error Message Standardization
    ## Format message template with module context identification tag
    formatted_msg = f"[{context_name.upper()} ERROR] - {message}"
    
    ## Log the validation failure using performance-efficient lazy evaluation
    logger.error("Validation failed in context %s: %s", context_name, message)
    
    raise ValidationError([formatted_msg])


def raise_workspace_error(context_name: str, message: str) -> NoReturn:
    """
    Helper function to raise standardized workspace configuration exceptions with uniform printing and logging.

    :param context_name: The name of the workspace context or directory target.
    :type context_name: str
    :param message: The descriptive error message.
    :type message: str
    :raises WorkspaceConfigError: Always raises WorkspaceConfigError.
    """
    # Workspace Error Formatting
    ## Create a unified diagnostic string
    formatted_msg = f"[{context_name.upper()} WORKSPACE ERROR] - {message}"
    
    ## Emit lazy evaluation logger message
    logger.error("Workspace configuration rejected for %s: %s", context_name, message)
    
    raise WorkspaceConfigError(formatted_msg)


def raise_model_initialization_error(model_name: str, message: str) -> NoReturn:
    """
    Helper function to raise standardized model initialization errors.

    :param model_name: Name of the model attempting initialization.
    :type model_name: str
    :param message: Descriptive error message explaining why compilation failed.
    :type message: str
    :raises ModelNotInitializedError: Always raises ModelNotInitializedError.
    """
    # Model State Failure Formatting
    ## Build unified crash diagnostic message
    formatted_msg = f"[MODEL {model_name.upper()} INIT ERROR] - {message}"
    
    ## Log critical structural issue with traceback capabilities
    logger.error("Model initialisation aborted for %s: %s", model_name, message)
    
    raise ModelNotInitializedError(formatted_msg)
