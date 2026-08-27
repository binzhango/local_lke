"""Schema-validated generation, citation integrity, repair, and degradation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, cast, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from local_lke.errors import GenerationError
from local_lke.generation.prompting import PROMPT_VERSION, PromptEvidence, build_generation_prompt
from local_lke.models import (
    AdvancedRetrievalTrace,
    AnswerResponse,
    AnswerStatus,
    Citation,
    CitedClaim,
    ComparisonAnswer,
    ConfidenceExplanation,
    ConfidenceLevel,
    FactListAnswer,
    GenerationTrace,
    OutputMode,
    QueryRoute,
    StructuredAnswer,
    StructuredSchemaName,
    TraceSummary,
    structured_citation_ids,
)
from local_lke.providers import ChatProvider


class ConversationalModelOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    claims: list[CitedClaim] = Field(min_length=1, max_length=30)


@runtime_checkable
class NativeStructuredChat(Protocol):
    def generate_structured(self, prompt: str, schema: type[BaseModel]) -> BaseModel: ...


@dataclass(frozen=True)
class GenerationEvidence:
    citation: Citation
    text: str
    retrieved: bool = True
    active: bool = True


@dataclass(frozen=True)
class GenerationRequest:
    question: str
    evidence: Sequence[GenerationEvidence]
    output_mode: OutputMode = OutputMode.CONVERSATIONAL
    schema_name: StructuredSchemaName | None = None
    uncovered_subquestions: Sequence[str] = ()
    route: QueryRoute | None = None
    answerability: str = "sufficient"
    sufficient: bool = True
    timings_ms: dict[str, float] | None = None
    retrieval_trace: AdvancedRetrievalTrace | None = None


class GenerationService:
    """Own all model output parsing; callers only receive validated responses."""

    def __init__(
        self,
        chat: ChatProvider,
        *,
        max_repair_attempts: int = 1,
        prefer_native_structured_output: bool = False,
    ) -> None:
        self.chat = chat
        self.max_repair_attempts = max(0, max_repair_attempts)
        self.prefer_native_structured_output = prefer_native_structured_output

    def generate(self, request: GenerationRequest) -> AnswerResponse:
        return self._generate(request)

    def generate_streaming(self, request: GenerationRequest) -> AnswerResponse:
        """Buffer model JSON until the stream ends; partial output is never committed."""

        if request.output_mode is OutputMode.EVIDENCE_ONLY or not request.sufficient:
            return self._generate(request)
        prompt, _schema = self._prompt_and_schema(request)
        try:
            raw = "".join(self.chat.stream(prompt))
        except Exception as exc:
            if _is_provider_failure(exc):
                raise GenerationError(
                    "The local model stream ended before a validated answer was available",
                    code="generation_stream_interrupted",
                ) from exc
            raise
        return self._generate(request, initial_raw=raw, force_parser=True)

    def _generate(
        self,
        request: GenerationRequest,
        *,
        initial_raw: str | None = None,
        force_parser: bool = False,
    ) -> AnswerResponse:
        registry = _validate_evidence(request.evidence)
        if not request.sufficient:
            return self._abstention(request)
        if not registry:
            raise GenerationError(
                "Sufficient generation requires retrieved active evidence",
                code="citation_integrity_error",
            )
        if request.output_mode is OutputMode.STRUCTURED and request.schema_name is None:
            raise GenerationError(
                "schema_name is required for structured output",
                code="generation_contract_error",
            )
        if request.output_mode is not OutputMode.STRUCTURED and request.schema_name is not None:
            raise GenerationError(
                "schema_name is only valid for structured output",
                code="generation_contract_error",
            )
        if request.output_mode is OutputMode.EVIDENCE_ONLY:
            citations = list(registry.values())
            return self._response(
                request=request,
                status=AnswerStatus.ANSWERED,
                answer=_extractive_answer(citations),
                structured=None,
                citations=citations,
                trace=GenerationTrace(
                    prompt_version=PROMPT_VERSION,
                    output_mode=request.output_mode,
                    schema_name=None,
                    model_output_committed=False,
                ),
                warnings=["Evidence-only mode bypassed model synthesis."],
            )

        prompt, schema = self._prompt_and_schema(request)
        errors: list[str] = []
        attempts = 0
        native_used = False
        parsed: BaseModel | None = None
        for attempt_index in range(self.max_repair_attempts + 1):
            attempts += 1
            try:
                if attempt_index == 0 and initial_raw is not None:
                    parsed = _parse_json(initial_raw, schema)
                elif (
                    attempt_index == 0
                    and not force_parser
                    and self.prefer_native_structured_output
                    and isinstance(self.chat, NativeStructuredChat)
                ):
                    native_used = True
                    parsed = self.chat.generate_structured(prompt, schema)
                    if not isinstance(parsed, schema):
                        parsed = schema.model_validate(parsed)
                else:
                    repair_prompt = self._build_prompt(
                        request, schema, validation_feedback=errors
                    )
                    parsed = _parse_json(self.chat.generate(repair_prompt), schema)
                _validate_model_citations(parsed, set(registry))
                break
            except GenerationError:
                raise
            except Exception as exc:
                if _is_provider_failure(exc):
                    return self._degraded(
                        request,
                        registry,
                        attempts=attempts,
                        errors=errors,
                        reason="local model synthesis was unavailable",
                        native_used=native_used,
                    )
                errors.append(_safe_validation_error(exc))
                parsed = None

        if parsed is None:
            return self._degraded(
                request,
                registry,
                attempts=attempts,
                errors=errors,
                reason="model output failed the selected schema after bounded repair",
                native_used=native_used,
            )

        citations = [
            registry[item] for item in sorted(_model_citation_ids(parsed), key=_id_order)
        ]
        structured = cast(
            StructuredAnswer | None,
            parsed if request.output_mode is OutputMode.STRUCTURED else None,
        )
        if isinstance(parsed, ConversationalModelOutput):
            answer = parsed.answer
        elif isinstance(parsed, (FactListAnswer, ComparisonAnswer)):
            answer = parsed.summary
        else:  # pragma: no cover - guarded by the selected schema
            raise GenerationError(
                "Validated output did not match the selected schema",
                code="generation_contract_error",
            )
        warnings = ["Model output required one bounded schema repair."] if errors else []
        return self._response(
            request=request,
            status=AnswerStatus.ANSWERED,
            answer=answer,
            structured=structured,
            citations=citations,
            trace=GenerationTrace(
                prompt_version=PROMPT_VERSION,
                output_mode=request.output_mode,
                schema_name=request.schema_name,
                attempts=attempts,
                repair_attempts=max(0, attempts - 1),
                native_structured_output=native_used,
                validation_errors=errors,
                model_output_committed=True,
            ),
            warnings=warnings,
        )

    def _prompt_and_schema(
        self, request: GenerationRequest
    ) -> tuple[str, type[BaseModel]]:
        schema = _schema_for(request.output_mode, request.schema_name)
        return self._build_prompt(request, schema), schema

    def _build_prompt(
        self,
        request: GenerationRequest,
        schema: type[BaseModel],
        *,
        validation_feedback: Sequence[str] = (),
    ) -> str:
        evidence = [
                PromptEvidence(
                    citation_id=item.citation.citation_id,
                    source_id=item.citation.source_id,
                    version_id=(
                        str(item.citation.document_version_id)
                        if item.citation.document_version_id
                        else None
                    ),
                    locator=item.citation.locator,
                    text=item.text,
                )
                for item in request.evidence
            ]
        return build_generation_prompt(
            question=request.question,
            evidence=evidence,
            output_mode=request.output_mode,
            schema_name=request.schema_name,
            output_schema=schema.model_json_schema(),
            uncovered_subquestions=request.uncovered_subquestions,
            route=request.route,
            answerability=request.answerability,
            validation_feedback=validation_feedback,
        )

    def _degraded(
        self,
        request: GenerationRequest,
        registry: dict[str, Citation],
        *,
        attempts: int,
        errors: list[str],
        reason: str,
        native_used: bool,
    ) -> AnswerResponse:
        citations = list(registry.values())[:3]
        return self._response(
            request=request,
            status=AnswerStatus.DEGRADED,
            answer=_extractive_answer(citations),
            structured=None,
            citations=citations,
            trace=GenerationTrace(
                prompt_version=PROMPT_VERSION,
                output_mode=request.output_mode,
                schema_name=request.schema_name,
                attempts=attempts,
                repair_attempts=max(0, attempts - 1),
                native_structured_output=native_used,
                validation_errors=errors,
                degraded_reason=reason,
                model_output_committed=False,
            ),
            warnings=[f"Degraded extractive response: {reason}."],
        )

    def _abstention(self, request: GenerationRequest) -> AnswerResponse:
        gap = request.answerability or "retrieved evidence is insufficient"
        return AnswerResponse(
            status=AnswerStatus.ABSTAINED,
            answer=(
                f"I cannot answer from the selected evidence because {gap}. "
                "Add or select a source that covers the missing information, then try again."
            ),
            citations=[],
            confidence=ConfidenceExplanation(
                level=ConfidenceLevel.LOW,
                rationale="The retrieval gate found an explicit evidence gap.",
            ),
            uncovered_subquestions=list(request.uncovered_subquestions),
            route=request.route,
            warnings=["Generation was not attempted because evidence was insufficient."],
            trace=TraceSummary(
                timings_ms=request.timings_ms or {},
                retrieval=request.retrieval_trace,
                generation=GenerationTrace(
                    prompt_version=PROMPT_VERSION,
                    output_mode=request.output_mode,
                    schema_name=request.schema_name,
                    model_output_committed=False,
                ),
            ),
        )

    def _response(
        self,
        *,
        request: GenerationRequest,
        status: AnswerStatus,
        answer: str,
        structured: StructuredAnswer | None,
        citations: list[Citation],
        trace: GenerationTrace,
        warnings: list[str],
    ) -> AnswerResponse:
        confidence = _confidence(status, citations, request.uncovered_subquestions)
        return AnswerResponse(
            status=status,
            answer=answer,
            structured_result=structured,
            citations=citations,
            confidence=confidence,
            uncovered_subquestions=list(request.uncovered_subquestions),
            route=request.route,
            warnings=warnings,
            trace=TraceSummary(
                timings_ms=request.timings_ms or {},
                retrieval=request.retrieval_trace,
                generation=trace,
            ),
        )


def _schema_for(
    mode: OutputMode, schema_name: StructuredSchemaName | None
) -> type[BaseModel]:
    if mode is OutputMode.CONVERSATIONAL:
        return ConversationalModelOutput
    if schema_name is StructuredSchemaName.FACT_LIST:
        return FactListAnswer
    if schema_name is StructuredSchemaName.COMPARISON:
        return ComparisonAnswer
    raise GenerationError("Unsupported structured schema", code="unsupported_schema")


def _parse_json(raw: str, schema: type[BaseModel]) -> BaseModel:
    if not raw.strip():
        raise ValueError("model returned an empty response")
    return schema.model_validate_json(raw)


def _validate_evidence(evidence: Sequence[GenerationEvidence]) -> dict[str, Citation]:
    registry: dict[str, Citation] = {}
    for item in evidence:
        citation_id = item.citation.citation_id
        if not item.retrieved or not item.active:
            raise GenerationError(
                "Generation evidence must come from an active retrieved source version",
                code="citation_integrity_error",
            )
        if not citation_id or citation_id in registry:
            raise GenerationError(
                "Generation evidence requires unique stable citation IDs",
                code="citation_integrity_error",
            )
        registry[citation_id] = item.citation
    return registry


def _validate_model_citations(parsed: BaseModel, allowed: set[str]) -> None:
    referenced = _model_citation_ids(parsed)
    if not referenced:
        raise ValueError("model output contains no citation IDs")
    unknown = referenced - allowed
    if unknown:
        raise ValueError("model output references citation IDs outside the retrieval manifest")


def _model_citation_ids(parsed: BaseModel) -> set[str]:
    if isinstance(parsed, ConversationalModelOutput):
        return {citation for claim in parsed.claims for citation in claim.citation_ids}
    if isinstance(parsed, (FactListAnswer, ComparisonAnswer)):
        return structured_citation_ids(parsed)
    return set()


def _extractive_answer(citations: Sequence[Citation]) -> str:
    return "\n\n".join(f"[{item.citation_id}] {item.excerpt}" for item in citations)


def _confidence(
    status: AnswerStatus, citations: Sequence[Citation], uncovered: Sequence[str]
) -> ConfidenceExplanation:
    if status is AnswerStatus.DEGRADED or uncovered:
        level = ConfidenceLevel.LOW
        rationale = "The response is extractive or some requested parts lack evidence."
    elif len(citations) >= 2:
        level = ConfidenceLevel.HIGH
        rationale = "The validated answer references multiple retrieved evidence items."
    else:
        level = ConfidenceLevel.MEDIUM
        rationale = "The validated answer is supported by one retrieved evidence item."
    return ConfidenceExplanation(level=level, rationale=rationale)


def _safe_validation_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        details = []
        for item in exc.errors(include_url=False, include_input=False)[:5]:
            location = ".".join(str(part) for part in item.get("loc", ())) or "response"
            details.append(f"{location}: {item.get('type', 'validation_error')}")
        return "; ".join(details)
    message = str(exc).casefold()
    if "empty" in message:
        return "response: empty_model_output"
    if "citation" in message:
        return "citation_ids: citation_integrity_error"
    return f"response: {type(exc).__name__}"


def _is_provider_failure(exc: Exception) -> bool:
    from local_lke.errors import ProviderUnavailableError

    return isinstance(exc, (ProviderUnavailableError, TimeoutError, ConnectionError))


def _id_order(value: str) -> int:
    return int(value.removeprefix("C"))
