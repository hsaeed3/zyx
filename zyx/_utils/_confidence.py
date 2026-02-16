"""zyx._utils._confidence

NOTE: This is a re-implementation of the ideas provided by Ruthvik Bandari's
pull request implementing confidence scoring in `instructor`. You can view
the original pull request here:

[PR](https://github.com/567-labs/instructor/pull/1968)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import enum
from typing import Any, Dict, List, Sequence, TypeVar

from .._aliases import (
    PydanticAIAgentResult,
)

__all__ = (
    "ConfidenceLevel",
    "FieldConfidence",
    "Confidence",
    "score_confidence",
)


Output = TypeVar("Output")


_HIGH_THRESHOLD = 0.90
_MEDIUM_THRESHOLD = 0.75
_LOW_THRESHOLD = 0.50


class ConfidenceLevel(enum.Enum):
    """
    Confidence level of a semantic operation's result.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


@dataclass
class FieldConfidence:
    """Per-field confidence breakdown for structured outputs."""

    field_name: str
    """Name of the output field."""

    confidence: float
    """Probability-based confidence score in ``[0, 1]``."""

    level: ConfidenceLevel
    """Qualitative confidence level."""

    token_count: int = 0
    """Number of tokens that were mapped to this field."""

    avg_logprob: float = 0.0
    """Mean log-probability across the tokens for this field."""


@dataclass
class Confidence:
    """Aggregated confidence information extracted from token log-probabilities.

    This is attached to :class:`ParsedResult` when the ``confidence`` flag
    is passed to ``parse`` / ``aparse``.
    """

    overall: float
    """Overall confidence score — geometric mean of per-token probabilities."""

    level: ConfidenceLevel
    """Qualitative confidence band for the overall score."""

    fields: Dict[str, FieldConfidence]
    """Per-field confidence breakdown (empty for simple/primitive outputs)."""

    token_count: int
    """Total number of content tokens that had log-probability data."""

    model: str = ""
    """Model name that produced the response."""

    @property
    def is_reliable(self) -> bool:
        """``True`` when overall confidence is :attr:`ConfidenceLevel.HIGH`."""
        return self.level is ConfidenceLevel.HIGH

    @property
    def low_confidence_fields(self) -> List[str]:
        """Field names whose confidence is LOW or VERY_LOW."""
        return [
            name
            for name, fc in self.fields.items()
            if fc.level in (ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW)
        ]

    def _should_hide_fields_in_repr(self) -> bool:
        """Hide `fields` in repr for primitive/single-value outputs.

        In primitive output cases we currently score a synthetic `"value"` field
        so we can still compute a sensible `overall` score without brittle
        string matching. That extra wrapper is an implementation detail and
        shouldn't clutter the default repr.
        """
        if not self.fields:
            return True
        if len(self.fields) != 1:
            return False
        fc = self.fields.get("value")
        return fc is not None and fc.field_name == "value"

    def __repr__(self) -> str:
        base = (
            "Confidence("
            f"overall={self.overall}, "
            f"level={self.level!r}, "
            f"token_count={self.token_count}"
        )
        if self.model:
            base += f", model={self.model!r}"

        if self._should_hide_fields_in_repr():
            return base + ")"

        return base + f", fields={self.fields!r})"


def _score_confidence_level(score: float) -> ConfidenceLevel:
    if score >= _HIGH_THRESHOLD:
        return ConfidenceLevel.HIGH
    if score >= _MEDIUM_THRESHOLD:
        return ConfidenceLevel.MEDIUM
    if score >= _LOW_THRESHOLD:
        return ConfidenceLevel.LOW
    return ConfidenceLevel.VERY_LOW


def _logprob_to_prob(lp: float) -> float:
    try:
        return float(math.exp(lp))
    except (OverflowError, ValueError):
        return 0.0


def _geometric_mean_prob(probs: Sequence[float]) -> float:
    """Geometric mean via log-sum, clamping near-zero values."""
    if not probs:
        return 0.0
    log_sum = sum(math.log(max(p, 1e-10)) for p in probs)
    return math.exp(log_sum / len(probs))


def _extract_logprob_tokens(
    runs: Sequence[PydanticAIAgentResult[Any]],
) -> List[Dict[str, Any]]:
    """Walk backwards through run messages and return the first list of
    token-level log-prob entries found.

    Checks both ``ModelResponse.provider_details`` (Chat API) and
    ``TextPart.provider_details`` (Responses API).
    """
    from pydantic_ai.messages import ModelResponse, TextPart

    for run in reversed(runs):
        try:
            messages = list(run.all_messages())
        except Exception:
            continue

        for msg in reversed(messages):
            if not isinstance(msg, ModelResponse):
                continue

            # --- Chat API path: logprobs on the response itself ---
            details = msg.provider_details or {}
            if isinstance(details, dict):
                logprobs = details.get("logprobs") or details.get("log_probs")
                if isinstance(logprobs, list) and logprobs:
                    return [
                        {
                            "token": item.get("token", ""),
                            "logprob": item.get("logprob"),
                            "probability": _logprob_to_prob(item["logprob"])
                            if item.get("logprob") is not None
                            else 0.0,
                        }
                        for item in logprobs
                        if isinstance(item, dict)
                    ]

            # --- Responses API path: logprobs on TextPart ---
            for part in msg.parts:
                if isinstance(part, TextPart) and part.provider_details:
                    logprobs = part.provider_details.get("logprobs")
                    if isinstance(logprobs, list) and logprobs:
                        return [
                            {
                                "token": item.get("token", ""),
                                "logprob": item.get("logprob"),
                                "probability": _logprob_to_prob(
                                    item["logprob"]
                                )
                                if item.get("logprob") is not None
                                else 0.0,
                            }
                            for item in logprobs
                            if isinstance(item, dict)
                        ]
    return []


def _map_tokens_to_fields(
    tokens: List[Dict[str, Any]],
    extracted_data: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    """Best-effort mapping of token spans to output field values.

    Looks for the string representation of each field value within the
    concatenated token text to find the tokens that correspond to it.
    """
    field_tokens: Dict[str, List[Dict[str, Any]]] = {
        k: [] for k in extracted_data
    }
    if not tokens:
        return field_tokens

    full_text = "".join(t.get("token", "") for t in tokens)

    for field_name, value in extracted_data.items():
        if value is None:
            continue
        value_str = str(value).strip()
        if not value_str:
            continue

        start_idx = full_text.lower().find(value_str.lower())
        if start_idx == -1 and isinstance(value, (int, float)):
            start_idx = full_text.lower().find(str(value))

        if start_idx != -1:
            char_count = 0
            value_end = start_idx + len(value_str)
            for token_data in tokens:
                token = token_data.get("token", "")
                token_start = char_count
                token_end = char_count + len(token)
                if token_start < value_end and token_end > start_idx:
                    field_tokens[field_name].append(token_data)
                char_count = token_end

        # Fallback: if we couldn't locate the value, use the global average
        if not field_tokens[field_name]:
            avg_prob = (
                sum(t.get("probability", 0.0) for t in tokens) / len(tokens)
                if tokens
                else 0.5
            )
            field_tokens[field_name] = [
                {"probability": avg_prob, "token": "", "logprob": 0.0}
            ]

    return field_tokens


def _score_field(
    field_name: str,
    value: Any,
    tokens: List[Dict[str, Any]],
) -> FieldConfidence:
    if not tokens:
        return FieldConfidence(
            field_name=field_name,
            confidence=0.5,
            level=ConfidenceLevel.LOW,
        )

    probs = [t.get("probability", 0.0) for t in tokens]
    logprobs = [t.get("logprob", 0.0) for t in tokens]
    confidence = _geometric_mean_prob(probs)
    avg_lp = sum(logprobs) / len(logprobs) if logprobs else 0.0

    return FieldConfidence(
        field_name=field_name,
        confidence=round(confidence, 4),
        level=_score_confidence_level(confidence),
        token_count=len(tokens),
        avg_logprob=round(avg_lp, 4),
    )


def _is_primitive_output(output: Any) -> bool:
    """True if output is a primitive (str, int, float, bool, None)."""
    return output is None or isinstance(output, (str, int, float, bool))


def score_confidence(
    runs: Sequence[PydanticAIAgentResult[Any]],
    output: Any,
    *,
    model_name: str | None = None,
) -> Confidence | None:
    """Compute a :class:`Confidence` score from token log-probabilities.

    Handles both structured outputs (per-field mapping) and primitive outputs
    (str, int, float, bool): for primitives the entire token sequence is used
    to compute a single overall confidence so scoring does not depend on
    string-matching the value in the token stream.

    Returns ``None`` when no usable log-prob data is available (e.g. the
    model/provider does not support log-probabilities).
    """
    tokens = _extract_logprob_tokens(runs)
    if not tokens:
        return None

    # Build a dict of fields for structured outputs; primitives get a single "value" key
    if hasattr(output, "model_dump"):
        output_map: Dict[str, Any] = output.model_dump()
    elif isinstance(output, dict):
        output_map = output
    else:
        output_map = {"value": output}

    is_primitive = _is_primitive_output(output)

    if is_primitive:
        # For primitives, use the full token sequence for the single "value" field
        # so we don't rely on string-matching (e.g. JSON "true" vs True, or spacing).
        field_token_map = {"value": tokens}
    else:
        field_token_map = _map_tokens_to_fields(tokens, output_map)

    field_results: Dict[str, FieldConfidence] = {}
    for fname, fvalue in output_map.items():
        field_results[fname] = _score_field(
            fname, fvalue, field_token_map.get(fname, [])
        )

    if field_results:
        overall = sum(fc.confidence for fc in field_results.values()) / len(
            field_results
        )
    else:
        probs = [t.get("probability", 0.0) for t in tokens]
        overall = _geometric_mean_prob(probs) if probs else 0.0

    return Confidence(
        overall=round(overall, 4),
        level=_score_confidence_level(overall),
        fields=field_results,
        token_count=len(tokens),
        model=model_name or "",
    )
