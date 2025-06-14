from abc import ABC, abstractmethod
from typing import Any
from langchain_core.embeddings import Embeddings


class EmbeddingServiceBase(ABC):
    """Абстрактный класс для Embedding сервисов"""

    client: Embeddings

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Проверка работы Embedding сервиса

        Returns:
            bool: True если сервис доступен, иначе False
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Получение размерности вектора эмбеддинга

        Returns:
            int: Размерность вектора
        """
        pass

    @abstractmethod
    def get_service_info(self) -> dict[str, Any]:
        """
        Получение информации о сервисе

        Returns:
            dict[str, Any]: Словарь с информацией о сервисе
        """
        pass
