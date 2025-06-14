from pathlib import Path
from contextlib import asynccontextmanager

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.services.chroma_db_service import ChromaDBService
from app.services.factory.embedding_service_factory import EmbeddingServiceFactory
from app.services.factory.llm_service_factory import LLMServiceFactory
from app.services.rag_service import RAGService

from app.api.endpoints import router, set_rag_service

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def load_initial_documents(rag_service: RAGService):
    """Загрузка начальных документов при старте"""
    data_dir = Path("./documents")

    if not data_dir.exists() or not data_dir.is_dir():
        logger.warning(
            f"Папка с документами {data_dir.resolve()} не найдена. Документы не будут загружены в ChromaDB автоматически."
        )
        return

    loaded_count = 0
    files_to_load = list(data_dir.glob("*.txt"))
    if not files_to_load:
        logger.warning(
            f"Не найдены .txt файлы документов в папке {data_dir.resolve()}."
        )
        return

    logger.info(
        f"Найдено {len(files_to_load)} документов в {data_dir.resolve()} для загрузки в ChromaDB."
    )

    for file_path in files_to_load:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            success = await rag_service.add_document(content, file_path.name)

            if success:
                loaded_count += 1
                logger.info(f"Успешно загружен и обработан файл {file_path.name}")
            else:
                logger.error(f"Не удалось обработать файл {file_path.name}")
        except Exception as e:
            logger.error(
                f"В процессе загрузки файла {file_path.name} произошла ошибка: {e}",
                exc_info=True,
            )

    try:
        info = rag_service.chroma_db.get_collection_info()
        logger.info(
            f"Загружены файлы в количестве: {loaded_count}. "
            f"Количество документов (чанков) в коллекции '{info.get('name', 'N/A')}': {info.get('documents_count', 0)}"
        )
    except Exception as e:
        logger.error(
            f"Ошибка при получении информации о коллекции после загрузки: {e}",
            exc_info=True,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""

    logger.info("Старт RAG-системы...")

    embedding_service = EmbeddingServiceFactory.create_service(
        api_provider=settings.embedding_api_provider,
        model=settings.embedding_api_model,
        api_key=settings.embedding_api_key,
        verify_ssl_certs=settings.verify_ssl_certs,
        ca_bundle_file=settings.mincifry_cert_path,
    )

    llm_service = LLMServiceFactory.create_service(
        api_provider=settings.llm_api_provider,
        model=settings.llm_api_model,
        api_key=settings.llm_api_key,
        verify_ssl_certs=settings.verify_ssl_certs,
        ca_bundle_file=settings.mincifry_cert_path,
    )

    vector_db = ChromaDBService(
        chroma_db_host=settings.chroma_db_host,
        chroma_db_port=settings.chroma_db_port,
        chroma_db_collection_name=settings.chroma_db_collection_name,
        embedding_service=embedding_service,
    )

    rag_service = RAGService(vector_db, llm_service)

    set_rag_service(rag_service)

    health_info = await rag_service.health_check()
    logger.info(f"Статус сервисов: {health_info}")

    if (
        health_info.get("chroma_db_status") == "healthy"
        and health_info.get("llm_status") == "healthy"
    ):
        logger.info("Основные сервисы работают, Загрузка начальных файлов.")
        await load_initial_documents(rag_service)
    else:
        logger.warning(
            "Один или несколько сервисов не работают. Пропуск загрузки начальных файлов."
        )

    logger.info("RAG-система успешно запущена.")
    yield
    logger.info("Завершение работы RAG-системы...")


app = FastAPI(
    title="UrFU AI Ассистент - RAG System",
    description="Масштабируемая RAG-система для университетского ИИ-ассистента",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1", tags=["RAG"])


@app.get("/")
async def root():
    """Корневой endpoint"""
    return {
        "message": "UrFU AI Assistant - RAG System",
        "version": "1.1.0",
        "embedding_provider": settings.embedding_api_provider,
        "llm_provider": settings.llm_api_provider,
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )
