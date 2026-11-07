"""Provider-local immutable selectors; no transport or Foundation dependency."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from fnmatch import fnmatchcase


@dataclass(frozen=True)
class ModelFilters:
    include: tuple[str, ...] = ("*",)
    exclude: tuple[str, ...] = ()
    hide_dated: bool = False
    legacy_filtered_model: str | None = None

    @classmethod
    def parse(cls, value: Mapping[str, object]) -> "ModelFilters":
        if not isinstance(value, Mapping):
            raise TypeError("model filters must be a mapping")
        if set(value) - {"include", "exclude", "hide_dated", "legacy_filtered_model"}:
            raise ValueError("unknown model filter key")

        def patterns(key: str, default: tuple[str, ...]) -> tuple[str, ...]:
            raw = value.get(key, default)
            if not isinstance(raw, (list, tuple)) or any(
                not isinstance(pattern, str) or not pattern.strip() for pattern in raw
            ):
                raise ValueError(
                    f"model filters {key} must be a list of nonblank strings"
                )
            return tuple(raw)

        hide_dated = value.get("hide_dated", False)
        if not isinstance(hide_dated, bool):
            raise TypeError("model filters hide_dated must be a boolean")
        legacy = value.get("legacy_filtered_model")
        if legacy is not None and (not isinstance(legacy, str) or not legacy.strip()):
            raise ValueError("legacy_filtered_model must be a nonblank string or null")
        return cls(
            patterns("include", ("*",)), patterns("exclude", ()), hide_dated, legacy
        )

    def select(self, ids: list[str]) -> list[str]:
        # The explicit old filtered option preserves its documented first-row fallback.
        # New defaults and every explicit filters mapping avoid it unless requested.
        if self.legacy_filtered_model is not None:
            ids = [i for i in ids if i == self.legacy_filtered_model] or ids[:1]
        selected = {
            model_id
            for model_id in ids
            if any(fnmatchcase(model_id.casefold(), p.casefold()) for p in self.include)
            and not any(
                fnmatchcase(model_id.casefold(), p.casefold()) for p in self.exclude
            )
        }
        if self.hide_dated:
            selected = {
                model_id
                for model_id in selected
                if not (
                    (match := re.search(r"-\d{4}-\d{2}-\d{2}$", model_id))
                    and model_id[: match.start()] in selected
                )
            }
        return sorted(selected)

    def as_dict(self) -> dict:
        result = {
            "include": list(self.include),
            "exclude": list(self.exclude),
            "hide_dated": self.hide_dated,
        }
        if self.legacy_filtered_model is not None:
            result["legacy_filtered_model"] = self.legacy_filtered_model
        return result


def selector_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "include": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "default": ["*"],
                "description": "Case-insensitive OS-independent glob patterns; [] includes none.",
            },
            "exclude": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "default": [],
                "description": "Case-insensitive glob patterns; exclusion wins.",
            },
            "hide_dated": {
                "type": "boolean",
                "default": False,
                "description": "Hide YYYY-MM-DD suffixes only when the selected undated alias exists.",
            },
            "legacy_filtered_model": {
                "type": ["string", "null"],
                "default": None,
                "description": "Deprecated filtered=true policy: exact configured ID, otherwise first endpoint row. Never applied by {}.",
            },
        },
    }
