from typing import Optional
import logging

from langchain_gigachat import GigaChat
from langchain_core.messages import HumanMessage, SystemMessage

from app.services.base.llm_service_base import LLMServiceBase

logger = logging.getLogger(__name__)


class GigaChatLLMService(LLMServiceBase):
    """Сервис для работы с GigaChat API через Langchain"""

    def __init__(
        self,
        api_key: Optional[str],
        model: str = "GigaChat",
        ca_bundle_file: Optional[str] = None,
        verify_ssl_certs: bool = False,
        scope: str = "GIGACHAT_API_PERS",
        timeout: float = 10.0,
    ):
        self.model = model
        self.verify_ssl_certs = verify_ssl_certs
        self.scope = scope
        self.timeout = timeout

        try:
            self.client = GigaChat(
                credentials=api_key,
                model=self.model,
                ca_bundle_file=ca_bundle_file,
                verify_ssl_certs=(
                    self.verify_ssl_certs if ca_bundle_file is None else True
                ),
                scope=self.scope,
                timeout=self.timeout,
            )
        except Exception as e:
            logger.error(
                f"При инициализации LLM сервиса GigaChat произошла ошибка: {e}",
                exc_info=True,
            )
            raise

    async def generate_response(
        self, prompt: str, context: str
    ) -> str | list[str | dict]:
        """Генерация ответа через GigaChat с использованием Langchain"""
        full_prompt = self._create_prompt(prompt, context)

        messages = [
            SystemMessage(
                content="Ты - помощник студентов Уральского Федерального университета. Отвечай кратко и по делу на русском языке."
            ),
            HumanMessage(content=full_prompt),
        ]

        try:
            response = await self.client.ainvoke(
                messages, max_tokens=500, temperature=0.7
            )
            if hasattr(response, "content"):
                return response.content
            else:
                logger.error(f"Неверный формат ответа от API: {response}")
                return "При обработке ответа произошла ошибка"
        except Exception as e:
            logger.error(f"Произошла ошибка в LLM сервисе GigaChat: {e}")
            return "Сервис временно недоступен"

    async def health_check(self) -> bool:
        """Проверка доступности GigaChat API через Langchain"""
        try:
            await self.client.ainvoke("Привет")
            return True
        except Exception as e:
            logger.error(
                f"При проверке работоспособности LLM сервиса GigaChat произошла ошибка: {e}"
            )
            return False

    def _create_prompt(self, prompt: str, context: str) -> str:
        """Создание промпта с контекстом"""
        return f"""
Контекст из документов университета:
{context}

Вопрос студента: {prompt}

Ответь на вопрос, используя только информацию из предоставленного контекста. 
Если информации недостаточно, скажи о том, что не обладаешь такой информацией и что студенту необходимо связаться с представителем университета напрямую.
"""
