from typing import Any

from pydantic import BaseModel


def to_jsonable(value: Any) -> Any:
    """Convert workflow values into JSON-compatible data."""

    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")

    if isinstance(value, dict):
        return {
            str(key): to_jsonable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]

    return value

