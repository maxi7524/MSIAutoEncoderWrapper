"""Session-only authentication helpers for the METASPACE strategy."""

from __future__ import annotations

import os
import sys
from typing import Any, Callable, Dict, Mapping, Optional

from ...utils.exceptions import MSILibException, raise_project_config_error


METASPACE_API_KEY_ENV = "METASPACE_API_KEY"


def metaspace_client_options(
    options: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return METASPACE client options with a session API key when available.

    :param options: Explicit non-secret or runtime client options.
    :type options: Mapping[str, Any] | None
    :return: Client options with ``api_key`` sourced from the current process
        environment unless it was supplied explicitly.
    :rtype: Dict[str, Any]

    The environment value is read on demand and is never written to a library
    configuration file.
    """
    resolved = dict(options or {})
    session_api_key = os.environ.get(METASPACE_API_KEY_ENV, "").strip()
    if "api_key" not in resolved and session_api_key:
        resolved["api_key"] = session_api_key
    return resolved


def validate_metaspace_session(
    client_factory: Optional[Callable[..., Any]] = None,
) -> bool:
    """Validate the session API key against METASPACE.

    :param client_factory: Optional client constructor used by tests.
    :type client_factory: Callable[..., Any] | None
    :return: ``True`` when METASPACE confirms an authenticated session.
    :rtype: bool
    :raises ProjectConfigError: If the environment key is missing, the client
        cannot be imported, or authentication is rejected.
    """
    api_key = os.environ.get(METASPACE_API_KEY_ENV, "").strip()
    if not api_key:
        raise_project_config_error(
            "MetaspaceAuthentication",
            f"Environment variable {METASPACE_API_KEY_ENV} is not set.",
        )
    if client_factory is None:
        try:
            from metaspace import SMInstance
        except ImportError:
            try:
                from metaspace.sm_annotation_utils import SMInstance
            except ImportError:
                raise_project_config_error(
                    "MetaspaceAuthentication",
                    "The METASPACE Python client is not installed.",
                )
        client_factory = SMInstance
    try:
        client = client_factory(api_key=api_key)
        authenticated = bool(client.logged_in())
    except Exception as error:
        raise_project_config_error(
            "MetaspaceAuthentication",
            f"METASPACE authentication request failed: {error}",
        )
    if not authenticated:
        raise_project_config_error(
            "MetaspaceAuthentication",
            "METASPACE rejected the API key or the account is unavailable.",
        )
    return True


def main() -> int:
    """Validate the current shell session and return a process exit code."""
    try:
        validate_metaspace_session()
    except MSILibException as error:
        print(f"METASPACE login failed: {error}", file=sys.stderr)
        return 1
    print("METASPACE login successful. The API key is active in this shell session.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
