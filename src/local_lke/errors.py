"""Actionable errors crossing provider and API boundaries."""


class LKEError(Exception):
    """Base application error safe to map to a public response."""

    code = "lke_error"
    component: str | None = None


class ProviderUnavailableError(LKEError):
    code = "provider_unavailable"

    def __init__(self, message: str, *, component: str) -> None:
        super().__init__(message)
        self.component = component


class PipelineNotReadyError(LKEError):
    code = "pipeline_not_ready"
    component = "pipeline"


class IngestionError(LKEError):
    """An upload or ingestion request could not be safely completed."""

    code = "ingestion_error"
    component = "ingestion"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


class IndexingError(LKEError):
    """An embedding, vector-index, or multimodal operation failed safely."""

    code = "indexing_error"
    component = "indexing"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


class NotFoundError(LKEError):
    code = "not_found"

    def __init__(self, message: str, *, component: str) -> None:
        super().__init__(message)
        self.component = component


class RetrievalError(LKEError):
    """A retrieval or structured-query request was rejected safely."""

    code = "retrieval_error"
    component = "retrieval"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


class GenerationError(LKEError):
    """A generation contract or citation-integrity boundary was rejected."""

    code = "generation_error"
    component = "generation"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


class EvaluationError(LKEError):
    """An evaluation dataset, run, or regression gate was invalid."""

    code = "evaluation_error"
    component = "evaluation"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


class AuthenticationError(LKEError):
    """A request did not present a configured Chapter 7 bearer credential."""

    code = "authentication_required"
    component = "security"


class AuthorizationError(LKEError):
    """An authenticated principal cannot perform an operation."""

    code = "permission_denied"
    component = "security"
