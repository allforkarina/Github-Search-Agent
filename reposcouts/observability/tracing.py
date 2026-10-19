from typing import Any

import re
import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)

# Regular expressions.
_BEARER_RE = re.compile(
    r"\bBearer[ \t]+[A-Za-z0-9._~+/=-]+",   # Bearer " " "\t" 可以多次出现(+means).
    re.IGNORECASE,                          # 忽略大小写
)

_GITHUB_TOKEN_RE = re.compile(
    r"\b(?:gh[pousr]_|github_pat_)[A-Za-z0-9_.-]+"  # (?:A|B) 不是A就是B, + means 1 or more times.
)

_API_TOKEN_RE = re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{12,}\b")
_SECRET_NAME = r"(?:api[_-]?key|access[_-]?token|github[_-]?token|token|secret|authorization|password|client[_-]?secret)"
_QUOTED_SECRET_RE = re.compile(r'''(?i)(\b''' + _SECRET_NAME + r'''\b["']?[ \t]*[:=][ \t]*)(["'])([^"'\r\n]*)(\2)''')
_ENV_SECRET_RE = re.compile(r"(?im)^([ \t]*[A-Z0-9_]*(?:TOKEN|API_KEY|PASSWORD|SECRET)[ \t]*=[ \t]*)([^\r\n]+)")
_PRIVATE_KEY_RE = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)")


def _mask_secret_lines(value: str) -> str:
    """Keep source line locations valid even when a secret spans multiple lines."""
    return "".join(
        "[REDACTED]" + line[len(line.rstrip("\r\n")):]
        for line in value.splitlines(keepends=True)
    )

# 统一小写、无连字符 只在dict中起作用
_SANITIZE_KEYS = (
    "authorization", "proxyauthorization", "auth", # authentication keywords
    "password", "passwd", "pwd", # password keywords
    "token", "accesstoken", "refreshtoken", # token keywords
    "githubtoken", "clientsecret", "clientid", # github keywords
    "cookie", "setcookie", # cookie keywords
)

def sanitize_trace_payload(
    value: Any, 
    *, 
    secrets: tuple[str, ...] = (), 
    max_string_chars: int = 8000, 
) -> Any: 
    """
    Then this function sanitizes the trace payload to prevent leaking info and limit the length.
    Args:
        value (Any): The trace payload to be sanitized.
        secrets (tuple[str, ...], optional): once appear in payload should be masked.
        max_string_chars (int, optional): Maximum number of characters allowed in each item. Defaults to 8000.
    Returns:
        Any: The sanitized trace payload.
    """
    if max_string_chars <= 0: 
        raise ValueError("max_string_chars must be a positive integer.")

    # 优先匹配长字段 避免部分替换
    sorted_secrets = sorted(
        {secret for secret in secrets if secret}, 
        key=len, 
        reverse=True, 
    )

    def clean_text(text: str, *, truncate: bool = True) -> str: 
        """
        Clean the text by masking secrets in the text, 
        and optionally truncating the text to a maximum length.
        Args:
            text (str): The text to be cleaned.
            truncate (bool, optional): Whether to truncate the text. Defaults to True.
        Returns:
            str: The cleaned text.
        """
        # sanitize
        for secret in sorted_secrets:
            text = text.replace(secret, _mask_secret_lines(secret))

        # find unknown github token, and replace with [REDACTED]
        text = _BEARER_RE.sub("Bearer [REDACTED]", text)
        text = _GITHUB_TOKEN_RE.sub("[REDACTED]", text)
        text = _API_TOKEN_RE.sub("[REDACTED]", text)
        text = _QUOTED_SECRET_RE.sub(lambda match: match[1] + match[2] + "[REDACTED]" + match[2], text)
        text = _ENV_SECRET_RE.sub(lambda match: match[1] + "[REDACTED]", text)
        text = _PRIVATE_KEY_RE.sub(lambda match: _mask_secret_lines(match[0]), text)

        if truncate and len(text) > max_string_chars:
            return text[:max_string_chars] + "[TRUNCATED]"

        return text
    
    # Situation 1: value is a dict.
    if isinstance(value, dict): 
        sanitized_dict = {} # sanitized dict
        for k, v in value.items(): 
            if not isinstance(k, str): 
                raise ValueError(f"Key {k} is not a string. Trace dictionary keys must be strings.")

            normalized_key = k.lower() # 小写
            normalized_key = re.sub(r"[^a-z0-9]", "", normalized_key) # 去掉非字母数字字符
            
            # state: key is in sensitive or not. value should be masked or not.
            sensitive = (
                normalized_key in _SANITIZE_KEYS 
                or normalized_key.endswith("apikey")
            )
            sanitized_dict[clean_text(k, truncate=False)] = (
                "[REDACTED]" if sensitive 
                else sanitize_trace_payload(v, secrets=secrets, max_string_chars=max_string_chars)
            )

        return sanitized_dict

    # Situation 2: value is a list.
    if isinstance(value, list):
        sanitized_list = [] # sanitized list
        for item in value: 
            sanitized_list.append(
                sanitize_trace_payload(item, secrets=secrets, max_string_chars=max_string_chars)
            )

        return sanitized_list

    # Situation 3: value is a string.
    if isinstance(value, str): 
        cleaned_value = clean_text(value)
        return cleaned_value

    # Situation 4: value is a primitive type (int, float, bool, None)
    if isinstance(value, (int, float, bool, type(None))):
        return value

    # If the value is not a supported type, return it as is.
    raise TypeError(f"Unsupported type {type(value)} for sanitization. Supported types are dict, list, str, int, float, bool, and None.")


def build_trace_client(
    *, 
    secrets: tuple[str, ...] = (), 
    max_string_chars: int = 8000, 
): 
    """
    Build a trace client with sanitization rules.
    Args:
        secrets (tuple[str, ...], optional): once appear in payload should be masked.
        max_string_chars (int, optional): Maximum number of characters allowed in each item. Defaults to 8000.
    """
    from langsmith import Client

    def anonymize(payload: dict) -> dict: 
        return sanitize_trace_payload(
            payload, 
            secrets=secrets,
            max_string_chars=max_string_chars, 
        )

    # anonymizer is a function decide how to sanitize the trace payload.
    return Client(anonymizer=anonymize)


def build_run_tracer(
    *,
    client,
    project_name: str,
    summary_provider: Callable[[str], dict],
):
    from langchain_core.tracers.langchain import LangChainTracer

    class RunSummaryTracer(LangChainTracer):
        def _persist_run(self, run) -> None:
            try:
                execution_status = (
                    "failed" if run.error is not None else "completed"
                )
                summary = dict(summary_provider(execution_status))

                extra = dict(run.extra or {})
                metadata = dict(extra.get("metadata") or {})
                metadata.update(summary)        # update summary to metadata.
                extra["metadata"] = metadata    # overwrite metadata.

                outputs = dict(run.outputs or {})
                outputs["observability"] = summary  # add summary to outputs.

                run.extra = extra
                run.outputs = outputs
            except Exception:
                logger.warning(
                    "Failed to attach run summary to root trace."
                )

            super()._persist_run(run)

    return RunSummaryTracer(
        client=client,
        project_name=project_name,
    )

if __name__ == "__main__": 
    # Example usage
    payload = {
        "Authorization": "Bearer abcdefghijklmnopqrstuvwxyz",
        "password": "mysecretpassword",
        "nested": {
            "token": "ghp_1234567890abcdef",
            "list": ["item1", "item2", "Bearer xyz"],
        },
        "normal_key": "normal_value",
    }

    secrets = ("mysecretpassword", "ghp_1234567890abcdef")
    sanitized_payload = sanitize_trace_payload(payload, secrets=secrets, max_string_chars=50)
    print(sanitized_payload)
