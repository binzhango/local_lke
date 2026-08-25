"""Application factories used by every delivery surface."""

from local_lke.providers import LangChainChatProvider, LocalHuggingFaceEmbeddings
from local_lke.rag import RAGPipeline
from local_lke.settings import Settings


def create_pipeline(settings: Settings) -> RAGPipeline:
    return RAGPipeline(
        chat=LangChainChatProvider(settings),
        embeddings=LocalHuggingFaceEmbeddings(settings.embedding_model),
        default_top_k=settings.default_top_k,
    )

