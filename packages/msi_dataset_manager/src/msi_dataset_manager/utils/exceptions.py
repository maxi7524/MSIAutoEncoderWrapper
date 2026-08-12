"""Dataset-manager exceptions and contextual raising helpers."""

from typing import NoReturn


class DatasetManagerError(Exception):
    """Base dataset-manager failure."""


class ValidationError(DatasetManagerError):
    """Invalid provider, filesystem, or canonical annotation data."""


class ExternalServiceError(DatasetManagerError):
    """Provider request failure."""


class DownloadLimitError(ExternalServiceError):
    """Provider refused binary download because of service policy or quota."""


class WorkspaceError(DatasetManagerError):
    """Invalid workspace state."""


class ProjectConfigError(DatasetManagerError):
    """Invalid runtime provider configuration."""


MSILibException = DatasetManagerError


def raise_validation_error(context_name: str, message: str) -> NoReturn:
    raise ValidationError(f"[{context_name.upper()} VALIDATION ERROR] - {message}")


def raise_workspace_error(context_name: str, message: str) -> NoReturn:
    raise WorkspaceError(f"[{context_name.upper()} WORKSPACE ERROR] - {message}")


def raise_external_service_error(context_name: str, message: str) -> NoReturn:
    raise ExternalServiceError(f"[{context_name.upper()} SERVICE ERROR] - {message}")


def raise_download_limit_error(context_name: str, message: str) -> NoReturn:
    raise DownloadLimitError(f"[{context_name.upper()} DOWNLOAD ERROR] - {message}")


def raise_project_config_error(context_name: str, message: str) -> NoReturn:
    raise ProjectConfigError(f"[{context_name.upper()} CONFIG ERROR] - {message}")
