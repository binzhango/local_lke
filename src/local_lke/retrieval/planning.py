"""Bounded, inspectable query transformation and plan validation."""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, ValidationError

from local_lke.errors import RetrievalError
from local_lke.models import (
    MetadataFilterPlan,
    QueryRoute,
    QueryTransformPlan,
    RetrievalStrategy,
    RewriteStrategy,
)
from local_lke.providers import ChatProvider


class MetadataPlanParser:
    """Generate a JSON-only metadata plan and validate its closed vocabulary."""

    def __init__(self, chat: ChatProvider) -> None:
        self.chat = chat

    def plan(self, question: str) -> MetadataFilterPlan:
        prompt = f"""Extract only document metadata constraints as JSON.
Allowed fields: filename, media_type, parser_strategy, chunk_strategy, page_number, created_at.
Allowed operators: eq, ne, in, contains, gt, gte, lt, lte.
Return exactly {{"conditions": [...], "allow_unfiltered_fallback": false}}.
Do not return SQL. Use an empty conditions list when no explicit constraint exists.

Question: {question}
JSON plan:"""
        return parse_model_plan(self.chat.generate(prompt), MetadataFilterPlan)


def build_query_plan(
    question: str,
    *,
    strategy: RetrievalStrategy,
    rewrite: RewriteStrategy,
    max_subqueries: int,
) -> QueryTransformPlan:
    """Create one deterministic plan with fixed fan-out and no agent loop."""
    normalized = " ".join(question.split())
    lowered = normalized.casefold()
    if strategy is RetrievalStrategy.STRUCTURED:
        route = QueryRoute.STRUCTURED
    elif _looks_multi_part(normalized):
        route = QueryRoute.MULTI_PART
    elif any(word in lowered for word in ("compare", "summarize", "overview", "across")):
        route = QueryRoute.BROAD_SYNTHESIS
    else:
        route = QueryRoute.SIMPLE_LOOKUP

    subqueries = (
        _decompose(normalized, max_subqueries)
        if route is QueryRoute.MULTI_PART
        else [normalized]
    )
    rewritten: str | None = None
    if rewrite is RewriteStrategy.STEP_BACK:
        rewritten = f"Background principles and context needed to answer: {normalized}"
    elif rewrite is RewriteStrategy.HYDE:
        rewritten = f"A relevant source passage would directly explain {normalized}"
    if rewritten and rewritten not in subqueries:
        subqueries = [*subqueries, rewritten][:max_subqueries]

    return QueryTransformPlan(
        original_query=question,
        normalized_query=normalized,
        route=route,
        subqueries=subqueries,
        rewrite=rewrite,
        rewritten_query=rewritten,
        rationale=(
            "Bounded deterministic routing; transformations are retrieval probes, not evidence."
        ),
    )


def parse_model_plan[PlanT: BaseModel](raw: str, model: type[PlanT]) -> PlanT:
    """Validate model-produced JSON as a typed plan; never accept executable code."""
    candidate = raw.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.IGNORECASE)
    try:
        payload = json.loads(candidate)
        return model.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise RetrievalError(
            "The model did not produce a valid allowlisted query plan",
            code="invalid_query_plan",
        ) from exc


def resolved_filters(filters: MetadataFilterPlan | None) -> MetadataFilterPlan:
    return filters or MetadataFilterPlan()


def _looks_multi_part(question: str) -> bool:
    lowered = question.casefold()
    return (
        question.count("?") > 1
        or ";" in question
        or " and also " in lowered
        or " as well as " in lowered
        or ("compare " in lowered and (" and " in lowered or " vs " in lowered))
    )


def _decompose(question: str, limit: int) -> list[str]:
    parts = re.split(
        r"\s*(?:\?|;|\band also\b|\bas well as\b)\s*",
        question,
        flags=re.IGNORECASE,
    )
    cleaned = [" ".join(part.split()).strip(" ,.") for part in parts]
    unique: list[str] = []
    for part in cleaned:
        if len(part) >= 2 and part.casefold() not in {item.casefold() for item in unique}:
            unique.append(part)
    return unique[:limit] or [question]
