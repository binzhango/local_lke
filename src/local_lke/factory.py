"""Application factories used by every delivery surface."""

from local_lke.ingestion import IngestionService
from local_lke.providers import LangChainChatProvider, LocalHuggingFaceEmbeddings
from local_lke.rag import RAGPipeline
from local_lke.settings import Settings
from local_lke.storage import (
    SqlAlchemyIngestionRepository,
    create_database_engine,
    create_session_factory,
)


def create_pipeline(settings: Settings) -> RAGPipeline:
    return RAGPipeline(
        chat=LangChainChatProvider(settings),
        embeddings=LocalHuggingFaceEmbeddings(settings.embedding_model),
        default_top_k=settings.default_top_k,
    )


def create_ingestion_service(settings: Settings) -> IngestionService:
    engine = create_database_engine(settings.database_url)
    repository = SqlAlchemyIngestionRepository(create_session_factory(engine), engine)
    return IngestionService(repository, settings)
