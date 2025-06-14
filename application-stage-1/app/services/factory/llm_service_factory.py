from typing import Any, Optional
import logging

from app.services.base.llm_service_base import LLMServiceBase
from app.services.gigachat.gigachat_llm_service import GigaChatLLMService

logger = logging.getLogger(__name__)


class LLMServiceFactory:
    """Фабрика для создания LLM сервисов"""

    @staticmethod
    def create_service(
        api_provider: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> LLMServiceBase:
        api_provider = api_provider.lower()

        verify_ssl_certs = kwargs.pop("verify_ssl_certs", False)
        ca_bundle_file = kwargs.pop("ca_bundle_file", None)

        available_services = LLMServiceFactory.get_available_services()
        service_config = available_services.get(api_provider)

        if not service_config:
            available_types = list(available_services.keys())
            logger.error(
                f"LLM сервис не поддерживается: {api_provider}, доступные сервисы: {available_types}"
            )
            raise

        if service_config.get("requires_api_key", False) and not api_key:
            logger.error(
                f"Для работы LLM сервиса {service_config['name']} необходим API ключ"
            )
            raise

        if not LLMServiceFactory._check_model_availability(
            api_provider, model, available_services
        ):
            available_models = service_config.get("models", [])
            logger.error(
                f"Модель '{model}' не поддерживается для LLM сервиса '{api_provider}', доступные модели: {available_models}"
            )
            raise

        default_model = service_config["models"][0]
        actual_model = model or default_model
        logger.info(
            f"Создание {service_config['name']} LLM сервиса с моделью: {actual_model}"
        )
        if api_provider == "gigachat":
            if not api_key:
                logger.error("Для работы LLM сервиса GigaChat необходим API ключ")
                raise
            return GigaChatLLMService(
                api_key=api_key,
                model=actual_model,
                ca_bundle_file=ca_bundle_file,
                verify_ssl_certs=verify_ssl_certs,
                **kwargs,
            )
        else:
            available_types = list(available_services.keys())
            logger.error(
                f"LLM сервис не поддерживается: {api_provider}, доступные сервисы: {available_types}"
            )
            raise

    @staticmethod
    def get_available_services() -> dict[str, dict[str, Any]]:
        return {
            "gigachat": {
                "name": "GigaChat",
                "models": [
                    "GigaChat",
                    "GigaChat-Pro",
                ],
                "requires_api_key": True,
            },
        }

    @staticmethod
    def _check_model_availability(
        api_provider: str,
        model: Optional[str],
        available_services_config: dict,
    ) -> bool:
        if model is None:
            return True

        service_info = available_services_config.get(api_provider)
        if not service_info:
            return False
        if model not in service_info.get("models", []):
            return False
        return True


create_llm_service = LLMServiceFactory.create_service
