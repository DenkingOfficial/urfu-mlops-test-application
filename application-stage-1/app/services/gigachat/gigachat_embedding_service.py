from typing import Optional
import logging

from langchain_gigachat.embeddings import GigaChatEmbeddings

from app.services.base.embedding_service_base import EmbeddingServiceBase

logger = logging.getLogger(__name__)


class GigaChatEmbeddingService(EmbeddingServiceBase):
    """Сервис для получения эмбеддингов от GigaChat"""

    MODEL_DIMENSIONS = {
        "Embeddings": 1024,
        "EmbeddingsGigaR": 1024,
    }

    def __init__(
        self,
        api_key: Optional[str],
        model: str = "Embeddings",
        ca_bundle_file: Optional[str] = None,
        verify_ssl_certs: bool = False,
        scope: str = "GIGACHAT_API_PERS",
        timeout: float = 10.0,
    ):
        self.model = model
        self.verify_ssl_certs = verify_ssl_certs
        self.scope = scope
        self.timeout = timeout

        if self.model not in self.MODEL_DIMENSIONS:
            logger.error(
                f"Модель не поддерживается: {self.model}, доступные модели: {list(self.MODEL_DIMENSIONS.keys())}"
            )
            raise

        try:
            self.client = GigaChatEmbeddings(
                credentials=api_key,
                model=self.model,
                ca_bundle_file=ca_bundle_file,
                verify_ssl_certs=(
                    self.verify_ssl_certs if ca_bundle_file is None else True
                ),
                scope=self.scope,
                timeout=self.timeout,
                prefix_query=""
            )
        except Exception as e:
            logger.error(
                f"При инициализации Embedding сервиса GigaChat произошла ошибка: {e}",
                exc_info=True,
            )
            raise

    async def health_check(self) -> bool:
        try:
            embedding = await self.client.aembed_query("Проверка")
            return bool(embedding) and len(embedding) > 0
        except Exception as e:
            logger.error(f"При проверке работоспособности Embedding сервиса GigaChat произошла ошибка: {e}")
            return False

    def get_embedding_dimension(self) -> int:
        embedding_dimension = self.MODEL_DIMENSIONS.get(self.model, 0)
        if embedding_dimension == 0:
            logger.error(
                f"Модель не поддерживается: {self.model}"
            )
            raise
        return embedding_dimension

    def get_service_info(self) -> dict[str, str]:
        return {
            "service": "GigaChat Embedding Service",
            "model": self.model,
            "scope": self.scope,
            "dimension": str(self.get_embedding_dimension()),
            "verify_ssl_certs": str(self.verify_ssl_certs),
            "timeout": str(self.timeout),
        }
