from abc import ABC, abstractmethod


class LLMServiceBase(ABC):
    """Абстрактный класс для LLM сервисов"""

    @abstractmethod
    async def generate_response(
        self,
        prompt: str,
        context: str,
    ) -> str:
        """
        Генерация ответа

        Args:
            prompt (str): Запрос пользователя
            context (str): Контекст из документов

        Returns:
            str: Ответ
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Проверка работы LLM сервиса

        Returns:
            bool: True если сервис доступен, иначе False
        """
