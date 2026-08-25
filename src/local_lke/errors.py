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

