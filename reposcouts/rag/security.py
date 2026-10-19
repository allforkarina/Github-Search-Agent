"""Sanitize repository evidence before it leaves this process."""
import os
import sys

from reposcouts.observability.tracing import sanitize_trace_payload



def configured_secrets() -> tuple[str, ...]:
    return tuple(
        value for key, value in os.environ.items()
        if value and (
            key.upper().endswith(("TOKEN", "API_KEY", "APIKEY", "PASSWORD", "SECRET", "SECRET_ACCESS_KEY"))
            or "_SECRET_" in key.upper()
        )
    )


def sanitize_content(text: str) -> str:
    # Avoid content truncation here: evidence budgets omit whole chunks instead.
    return sanitize_trace_payload(
        text, secrets=configured_secrets(), max_string_chars=sys.maxsize,
    )


def sanitize_data(value):
    if isinstance(value, str):
        return sanitize_content(value)
    if isinstance(value, dict):
        safe = sanitize_trace_payload(value, secrets=configured_secrets(), max_string_chars=24000)
        return {key: sanitize_data(item) for key, item in safe.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_data(item) for item in value]
    return value
