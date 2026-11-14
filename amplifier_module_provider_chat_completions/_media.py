"""Route-local fallback and non-retaining diagnostic copies. No storage API."""

import base64
import copy
import re
from typing import Any

from amplifier_core.message_models import ImageBlock, TextBlock

MEDIA_NAMESPACE = "experimental.media.v1"
UNAVAILABLE = (
    "[Image unavailable: selected endpoint/model image support is unqualified; "
    "no pixels supplied]"
)
NOT_RETAINED = "[Image transport not retained in diagnostics]"
_DATA_IMAGE = re.compile(r"data:image/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=_-]+")
# Literal IDs/snapshots from official model pages fetched 2026-11-14. Absence
# means unqualified, NOT that the model is incapable. Never infer from a prefix.
OFFICIAL_USER_IMAGE_MODELS = {
    "gpt-6.1-sol": "https://developers.openai.com/api/docs/models/gpt-6.1-sol",
    "gpt-5.4": "https://developers.openai.com/api/docs/models/gpt-5.4",
    "gpt-5.4-2026-03-05": "https://developers.openai.com/api/docs/models/gpt-5.4",
    "gpt-5.5": "https://developers.openai.com/api/docs/models/gpt-5.5",
    "gpt-5.5-2026-04-23": "https://developers.openai.com/api/docs/models/gpt-5.5",
}
OFFICIAL_BASE_URLS = ("https://api.openai.com/v1", "https://api.openai.com/v1/")


def user_image_part(block: ImageBlock) -> dict[str, Any] | None:
    """Encode verified private PNG/JPEG transport, never resolve refs or paths."""
    source = block.source
    mime, data = source.get("media_type"), source.get("data")
    if source.get("type") != "base64" or mime not in ("image/png", "image/jpeg"):
        return None
    if not isinstance(data, str) or not data:
        return None
    try:
        decoded = base64.b64decode(data, validate=True)
    except (ValueError, TypeError):
        return None
    # The session owner verifies decoding/dimensions. This final encoder only
    # accepts the original supported transport; it neither transcodes nor logs.
    if not decoded:
        return None
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def contains_media(value: Any) -> bool:
    """Only actual structured media; never parse strings or invent capabilities."""
    if isinstance(value, ImageBlock):
        return True
    if isinstance(value, dict):
        if MEDIA_NAMESPACE in value or value.get("type") in (
            "image",
            "image_url",
            "input_image",
        ):
            return True
        if value.get("type") == "base64" and str(
            value.get("media_type", "")
        ).startswith("image/"):
            return True
        return any(contains_media(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_media(item) for item in value)
    return False


def diagnostic_copy(value: Any) -> Any:
    """Omit known Core/vendor image transports without ingesting or hashing.

    This is a copy, not a media ref: no digest, path or implied resolvability.
    Strings are not JSON-decoded. Ordinary opaque reasoning/tool text survives.
    """
    if isinstance(value, ImageBlock):
        return {"type": "image_not_retained", "reason": NOT_RETAINED}
    if isinstance(value, dict):
        kind = value.get("type")
        if kind in ("image", "image_url", "input_image", "image_not_retained"):
            return {"type": "image_not_retained", "reason": NOT_RETAINED}
        if kind == "base64" and str(value.get("media_type", "")).startswith("image/"):
            return {"type": "image_not_retained", "reason": NOT_RETAINED}
        return {key: diagnostic_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [diagnostic_copy(item) for item in value]
    if isinstance(value, str):
        return _DATA_IMAGE.sub(NOT_RETAINED, value)
    return copy.deepcopy(value)


def tool_output_text(value: Any) -> str:
    """Handle actual Core rich output only; never promote JSON-looking text."""
    if isinstance(value, ImageBlock):
        return UNAVAILABLE
    if isinstance(value, TextBlock):
        return value.text
    if isinstance(value, (list, tuple)):
        return "\n".join(tool_output_text(item) for item in value)
    if isinstance(value, dict):
        if value.get("type") in ("image", "image_url", "input_image") or (
            value.get("type") == "base64"
            and str(value.get("media_type", "")).startswith("image/")
        ):
            return UNAVAILABLE
        if value.get("type") == "text" and isinstance(value.get("text"), str):
            return value["text"]
        # Actual structured mappings are not JSON-looking text. Remove known
        # nested image blocks before rendering remaining status/output fields.
        return str({key: _tool_output_copy(item) for key, item in value.items()})
    return str(value)


def _tool_output_copy(value: Any) -> Any:
    if isinstance(value, (ImageBlock, TextBlock)):
        return tool_output_text(value)
    if isinstance(value, dict):
        if value.get("type") in ("image", "image_url", "input_image", "base64"):
            return tool_output_text(value)
        return {key: _tool_output_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_tool_output_copy(item) for item in value]
    return value
