"""
Domain-specific exceptions for the MSI library core module.
"""

class MSILibException(Exception):
    """Base exception class for all msi_lib runtime errors."""
    pass

class ProjectConfigError(MSILibException):
    """Raised when the project JSON configuration file is corrupted, missing required keys, or invalid."""
    pass

class ModelNotInitializedError(MSILibException):
    """Raised when an operation requiring a compiled model (e.g., fit, save, transform) is executed before initialization."""
    pass

class IncompatibleInterfaceError(MSILibException):
    """Raised when shape or type mismatches occur between component managers (e.g., binners, architectures, datasets)."""
    pass