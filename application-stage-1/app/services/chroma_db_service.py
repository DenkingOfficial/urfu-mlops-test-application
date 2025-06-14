from typing import Any
import chromadb
import logging

from langchain_chroma import Chroma
from chromadb.config import Settings as ChromaSettings

from app.services.base.embedding_service_base import EmbeddingServiceBase

logger = logging.getLogger(__name__)


class ChromaDBService:

    def __init__(
        self,
        chroma_db_host: str,
        chroma_db_port: str,
        chroma_db_collection_name: str,
        embedding_service: EmbeddingServiceBase,
    ):
        self.embedding_service = embedding_service
        embedding_service_info = self.embedding_service.get_service_info()

        logger.info(
            f"Embedding сервис: {embedding_service_info.get('api_provider')}; Модель: {embedding_service_info.get('model')}",
        )

        COLLECTION_METADATA = {
            "description": "Коллекция документов УрФУ",
            "embedding_service": embedding_service_info.get("service", "Unknown"),
            "embedding_model": embedding_service_info.get("model", "Unknown"),
            "hnsw:space": "cosine",
        }

        logger.info(f"Подключение к ChromaDB на {chroma_db_host}:{chroma_db_port}")
        self.chroma_db_interface = Chroma(
            client=chromadb.HttpClient(
                host=chroma_db_host,
                port=int(chroma_db_port),
            ),
            client_settings=ChromaSettings(anonymized_telemetry=False),
            collection_name=chroma_db_collection_name,
            collection_metadata=COLLECTION_METADATA,
            create_collection_if_not_exists=True,
            embedding_function=self.embedding_service.client,
        )
        logger.info(
            f"Инициализирована ChromaDB с количеством документов: {self.chroma_db_interface._collection.count()}.",
        )

    async def add_documents(
        self,
        documents: list[str],
        ids: list[str],
    ) -> bool:
        """Добавление документов в коллекцию ChromaDB"""
        if not documents:
            logger.warning("Нет документов для добавления")
            return True
        try:
            logger.info(
                f"Добавление документов ({len(documents)}) с ID: {ids[:3]}{'...' if len(ids) > 3 else ''}"
            )
            self.chroma_db_interface.add_texts(
                texts=documents,
                ids=ids,
            )
            logger.info(f"Документы ({len(documents)}) успешно добавлены в коллекцию")
            return True
        except Exception as e:
            logger.error(
                f"При добавлении документов в коллецию произошла ошибка: {e}",
                exc_info=True,
            )
            return False

    async def search(
        self,
        query: str,
        limit: int = 4,
    ) -> list[dict[str, Any]]:
        """Поиск похожих документов в коллекции ChromaDB"""
        if not query.strip():
            logger.warning("Текст запроса не указан")
            return []

        try:
            logger.info(f"Поиск по запросу: '{query}'")
            results = self.chroma_db_interface.similarity_search_with_relevance_scores(
                query=query,
                k=limit,
            )
            if results:
                formatted_results = [
                    {
                        "id": doc.id,
                        "content": doc.page_content,
                        "similarity_score": score,
                    }
                    for doc, score in results
                ]
                formatted_results.sort(
                    key=lambda x: x["similarity_score"], reverse=True
                )
                logger.info(
                    f"Возврат {len(formatted_results)} результатов. Наивысший similarity_score: {formatted_results[0]['similarity_score'] if formatted_results else 'N/A'}"
                )
                return formatted_results
            return []
        except Exception as e:
            logger.error(
                f"При поиске документов произошла ошибка: {e}",
                exc_info=True,
            )
            return []

    def get_collection_info(self) -> dict[str, Any]:
        """Получение информации о коллекции ChromaDB"""
        try:
            return {
                "name": self.chroma_db_interface._collection_name,
                "documents_count": self.chroma_db_interface._collection.count(),
                "metadata": self.chroma_db_interface._collection_metadata,
            }
        except Exception as e:
            logger.error(
                f"При получении информации о коллекции произошла ошибка: {e}",
                exc_info=True,
            )
            return {"error": str(e)}

    def clear_collection(self) -> bool:
        """Очистка коллекции ChromaDB"""
        try:
            logger.info(
                f"Попытка очистить коллецию: {self.chroma_db_interface._collection_name}"
            )
            self.chroma_db_interface.reset_collection()
            logger.info(
                f"Коллекция {self.chroma_db_interface._collection_name} очищена"
            )
            return True
        except Exception as e:
            logger.error(
                f"При очистке коллекции произошла ошибка: {e}",
                exc_info=True,
            )
            return False

    async def health_check(self) -> bool:
        """Проверка работоспособности сервиса"""
        try:
            logger.info("Проверка работоспособности сервисов")
            status = (
                bool(self.chroma_db_interface._client.heartbeat())
                and await self.embedding_service.health_check()
            )
            return status
        except Exception as e:
            logger.error(
                f"При проверке работоспособности сервисов произошла ошибка: {e}",
                exc_info=True,
            )
            return False

    async def get_embedding_service_info(self) -> dict[str, Any]:
        """Получение информации о сервисе эмбеддингов"""
        try:
            service_info = self.embedding_service.get_service_info()
            health_check = await self.embedding_service.health_check()
            return {
                **service_info,
                "health_status": "OK" if health_check else "DOWN",
            }
        except Exception as e:
            logger.error(f"Произошла ошибка: {e}")
            return {
                "error": str(e),
            }
