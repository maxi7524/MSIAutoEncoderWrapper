"""
Domain-specific exceptions for the MSI library core module.
"""

class MSILibException(Exception):
    """
    Base exception class for all msi_lib runtime errors.
    """
    pass

class ProjectConfigError(MSILibException):
    """
    Raised when the project JSON configuration file is corrupted or invalid.
    """
    pass

class ModelNotInitializedError(MSILibException):
    """
    Raised when an operation requiring a compiled model is executed prematurely.
    """
    pass

class IncompatibleInterfaceError(MSILibException):
    """
    Raised when shape or type mismatches occur between component managers.
    """
    pass

class WorkspaceConfigError(MSILibException):
    """
    Raised when workspace path configuration or custom layouts are invalid.
    """
    pass

class ValidationError(MSILibException):
    """
    Raised when one or more required runtime objects or files are missing.
    Accumulates all missing targets in the error message.
    """
    def __init__(self, missing_components: list):
        """
        Initialize the validation error with a list of missing elements.

        :param missing_components: List of descriptions of missing components.
        :type missing_components: list
        """
        self.missing_components = missing_components
        message = f"Validation failed. Missing required components/files: {', '.join(missing_components)}"
        super().__init__(message)