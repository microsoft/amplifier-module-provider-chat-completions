"""Small Chat Completions image projection and diagnostic-copy helpers.

No URL fetching, history mutation, model capability inference or media storage.
"""

import base64
import binascii
import re
from typing import Any
from urllib.parse import urlsplit

from amplifier_core.message_models import (
    ImageBlock,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
)

UNAVAILABLE = (
    "[Image unavailable: this Chat Completions position/source is unsupported; "
    "no pixels supplied. If needed, attach a supported image in a user message "
    "and select a backend/model with documented vision support.]"
)
NOT_RETAINED = "[Image transport omitted from diagnostics]"
_DATA_IMAGE = re.compile(r"data:image/[^\s;,]+;base64,[A-Za-z0-9+/=_-]+")


def _is_image(value: Any) -> bool:
    return isinstance(value, ImageBlock) or (
        isinstance(value, dict)
        and (
            (value.get("type") == "image" and isinstance(value.get("source"), dict))
            or (value.get("type") == "image_url" and "image_url" in value)
            or (value.get("type") == "input_image" and "image_url" in value)
            or (
                value.get("type") == "base64"
                and str(value.get("media_type", "")).startswith("image/")
            )
        )
    )


def contains_images(value: Any) -> bool:
    """Inspect content blocks and known result envelopes, not business fields."""
    if _is_image(value):
        return True
    if isinstance(value, Message):
        return contains_images(value.content)
    if isinstance(value, ToolResultBlock):
        return contains_images(value.output)
    if isinstance(value, dict):
        # ToolResult.output may wrap canonical blocks in content/output.
        # A nested product record with type=image is not a content position.
        return any(
            contains_images(value[key]) for key in ("content", "output") if key in value
        )
    if isinstance(value, (list, tuple)):
        return any(contains_images(item) for item in value)
    return False


def known_image_values(value: Any) -> tuple[str, ...]:
    """Exact transport strings from actual content slots, never guessed text."""
    if isinstance(value, ImageBlock):
        return tuple(
            item
            for key in ("data", "url")
            if isinstance(item := value.source.get(key), str) and item
        )
    if _is_image(value):
        if value.get("type") == "image":
            return known_image_values(ImageBlock(source=value["source"]))
        if value.get("type") == "image_url":
            url = (
                value["image_url"].get("url")
                if isinstance(value["image_url"], dict)
                else None
            )
        else:
            url = value.get("image_url") or value.get("data")
        return (url,) if isinstance(url, str) and url else ()
    if isinstance(value, Message):
        return known_image_values(value.content)
    if isinstance(value, ToolResultBlock):
        return known_image_values(value.output)
    if isinstance(value, dict):
        return tuple(
            part
            for key in ("content", "output")
            if key in value
            for part in known_image_values(value[key])
        )
    if isinstance(value, (list, tuple)):
        return tuple(part for item in value for part in known_image_values(item))
    return ()


def user_image_part(block: ImageBlock) -> dict[str, Any]:
    """Encode documented image_url variants; invalid sources stay explicit."""
    source = block.source
    if source.get("type") == "base64":
        mime, data = source.get("media_type"), source.get("data")
        if mime not in ("image/png", "image/jpeg", "image/webp", "image/gif"):
            return {"type": "text", "text": UNAVAILABLE}
        if not isinstance(data, str) or not data:
            return {"type": "text", "text": UNAVAILABLE}
        try:
            if not base64.b64decode(data, validate=True):
                return {"type": "text", "text": UNAVAILABLE}
        except (ValueError, binascii.Error):
            return {"type": "text", "text": UNAVAILABLE}
        url = f"data:{mime};base64,{data}"
    elif source.get("type") == "url":
        url = source.get("url")
        if not isinstance(url, str):
            return {"type": "text", "text": UNAVAILABLE}
        try:
            parsed = urlsplit(url)
            valid = (
                parsed.scheme in ("https", "http")
                and bool(parsed.hostname)
                and parsed.username is None
                and parsed.password is None
            )
        except ValueError:
            valid = False
        if not valid:
            return {"type": "text", "text": UNAVAILABLE}
    else:
        return {"type": "text", "text": UNAVAILABLE}
    return {"type": "image_url", "image_url": {"url": url}}


def _is_thinking(value: Any) -> bool:
    """Recognize Core private blocks only at the known content positions below."""
    return isinstance(value, ThinkingBlock) or (
        isinstance(value, dict)
        and value.get("type") == "thinking"
        and isinstance(value.get("thinking"), str)
    )


def _tool_copy(value: Any) -> Any:
    """Replace structured pixels only; retain other fields, status and order."""
    if _is_thinking(value):
        return ""  # Match ordinary message conversion: never stringify signatures.
    if _is_image(value):
        return UNAVAILABLE
    if isinstance(value, TextBlock):
        return value.text
    if isinstance(value, ToolResultBlock):
        return _tool_copy(value.model_dump())
    if isinstance(value, dict):
        if (
            value.get("type") == "text"
            and isinstance(value.get("text"), str)
            and set(value) <= {"type", "text"}
        ):
            return value["text"]
        return {
            key: _tool_copy(item) if key in ("content", "output") else item
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_tool_copy(item) for item in value if not _is_thinking(item)]
    return value


def tool_output_text(value: Any) -> str:
    # Preserve legacy non-image output serialization exactly.
    if not contains_images(value):
        return str(value)
    projected = _tool_copy(value)
    if isinstance(projected, list):
        return "\n".join(str(item) for item in projected)
    return str(projected)


def diagnostic_copy(value: Any, image_values: tuple[str, ...] = ()) -> Any:
    """Omit structured transports and exact known echoes in raw diagnostics.

    Opaque ordinary text is not decoded or classified as base64/URLs.
    """
    if _is_image(value):
        return {"type": "image_omitted", "reason": NOT_RETAINED}
    if isinstance(value, dict):
        return {key: diagnostic_copy(item, image_values) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [diagnostic_copy(item, image_values) for item in value]
    if isinstance(value, str):
        result = _DATA_IMAGE.sub(NOT_RETAINED, value)
        for transport in image_values:
            result = result.replace(transport, NOT_RETAINED)
        return result
    return value
