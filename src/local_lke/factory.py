"""Application factories used by every delivery surface."""

from local_lke.indexing import (
    IndexingService,
    MultimodalIndexingService,
    SqlAlchemyIndexRepository,
)
from local_lke.ingestion import IngestionService
from local_lke.providers import (
    LangChainChatProvider,
    LocalCLIPEmbeddings,
    LocalHuggingFaceEmbeddings,
)
from local_lke.rag import RAGPipeline
from local_lke.retrieval import (
    AdvancedRetrievalService,
    LocalCrossEncoderReranker,
    MetadataPlanParser,
    StructuredDataService,
    StructuredPlanParser,
)
from local_lke.settings import Settings
from local_lke.storage import (
    SqlAlchemyIngestionRepository,
    create_database_engine,
    create_session_factory,
)


def create_pipeline(settings: Settings) -> RAGPipeline:
    return RAGPipeline(
        chat=LangChainChatProvider(settings),
        embeddings=LocalHuggingFaceEmbeddings(
            settings.embedding_model,
            revision=settings.embedding_model_revision,
            normalized=settings.embedding_normalize,
            document_prefix=settings.embedding_document_prefix,
            query_prefix=settings.embedding_query_prefix,
        ),
        default_top_k=settings.default_top_k,
        generation_max_repair_attempts=settings.generation_max_repair_attempts,
        generation_native_structured_output=settings.generation_native_structured_output,
    )


def create_ingestion_service(settings: Settings) -> IngestionService:
    engine = create_database_engine(settings.database_url)
    repository = SqlAlchemyIngestionRepository(create_session_factory(engine), engine)
    return IngestionService(repository, settings)


def create_retrieval_services(
    settings: Settings,
    pipeline: RAGPipeline,
    ingestion: IngestionService,
    indexing: IndexingService | None = None,
) -> tuple[AdvancedRetrievalService, StructuredDataService]:
    repository = ingestion.repository
    if not isinstance(repository, SqlAlchemyIngestionRepository):
        raise TypeError("Chapter 4 services require the SQLAlchemy repository")
    reranker = (
        LocalCrossEncoderReranker(settings.reranker_model)
        if settings.reranker_enabled
        else None
    )
    retrieval = AdvancedRetrievalService(
        repository=repository,
        embeddings=pipeline.embeddings,
        chat=pipeline.chat,
        settings=settings,
        reranker=reranker,
        metadata_planner=MetadataPlanParser(pipeline.chat),
        indexing=indexing,
        generation=pipeline.generation,
    )
    structured = StructuredDataService(
        repository,
        settings,
        StructuredPlanParser(pipeline.chat),
    )
    return retrieval, structured


def create_indexing_services(
    settings: Settings,
    pipeline: RAGPipeline,
    ingestion: IngestionService,
) -> tuple[IndexingService, MultimodalIndexingService]:
    repository = ingestion.repository
    if not isinstance(repository, SqlAlchemyIngestionRepository):
        raise TypeError("Chapter 3 services require the SQLAlchemy repository")
    indexes = SqlAlchemyIndexRepository(repository.sessions, repository.engine)
    indexing = IndexingService(indexes, pipeline.embeddings, settings)
    multimodal = MultimodalIndexingService(
        indexes,
        LocalCLIPEmbeddings(
            settings.multimodal_model,
            revision=settings.multimodal_model_revision,
            dimension=settings.multimodal_dimension,
        ),
        settings,
    )
    return indexing, multimodal
