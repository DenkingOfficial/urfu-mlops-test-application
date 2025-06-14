from typing import Any, Optional
import logging

from app.services.base.embedding_service_base import EmbeddingServiceBase
from app.services.gigachat.gigachat_embedding_service import GigaChatEmbeddingService

logger = logging.getLogger(__name__)


class EmbeddingServiceFactory:
    """Фабрика для создания сервисов эмбеддингов"""

    @staticmethod
    def create_service(
        api_provider: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> EmbeddingServiceBase:
        api_provider = api_provider.lower()

        verify_ssl_certs = kwargs.pop("verify_ssl_certs", False)
        ca_bundle_file = kwargs.pop("ca_bundle_file", None)

        available_services = EmbeddingServiceFactory.get_available_services()
        service_config = available_services.get(api_provider)

        if not service_config:
            available_types = list(available_services.keys())
            logger.error(
                f"Embedding сервис не поддерживается: {api_provider}, доступные сервисы: {available_types}"
            )
            raise

        if service_config.get("requires_api_key", False) and not api_key:
            logger.error(
                f"Для работы Embedding сервиса {service_config['name']} необходим API ключ"
            )
            raise

        if not EmbeddingServiceFactory._check_model_availability(
            api_provider, model, available_services
        ):
            available_models = service_config.get("models", [])
            logger.error(
                f"Модель '{model}' не поддерживается для Embedding сервиса '{api_provider}', доступные модели: {available_models}"
            )
            raise

        default_model = service_config["models"][0]
        actual_model = model or default_model
        logger.info(
            f"Создание {service_config['name']} Embedding сервиса с моделью: {actual_model}"
        )
        if api_provider == "gigachat":
            if not api_key:
                logger.error("Для работы Embedding сервиса GigaChat необходим API ключ")
                raise
            return GigaChatEmbeddingService(
                api_key=api_key,
                model=actual_model,
                ca_bundle_file=ca_bundle_file,
                verify_ssl_certs=verify_ssl_certs,
                **kwargs,
            )
        else:
            available_types = list(available_services.keys())
            logger.error(
                f"Embedding сервис не поддерживается: {api_provider}, доступные сервисы: {available_types}"
            )
            raise

    @staticmethod
    def get_available_services() -> dict[str, dict[str, Any]]:
        return {
            "gigachat": {
                "name": "GigaChat",
                "models": ["Embeddings", "EmbeddingsGigaR"],
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


create_embedding_service = EmbeddingServiceFactory.create_service
