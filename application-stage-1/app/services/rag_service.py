import time
import logging
from typing import Any

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)

from app.models.schemas import QueryResponse
from app.services.base.llm_service_base import LLMServiceBase
from app.services.chroma_db_service import ChromaDBService


logger = logging.getLogger(__name__)


class RAGService:
    """Сервис для работы с RAG"""

    def __init__(
        self,
        chroma_db: ChromaDBService,
        llm_service: LLMServiceBase,
    ):
        self.chroma_db = chroma_db
        self.llm_service = llm_service
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=250,
            chunk_overlap=50,
            length_function=len,
        )

    async def process_query(self, prompt: str) -> QueryResponse:
        """Обработка запроса пользователя"""
        start_time = time.time()

        try:
            logger.info(f"Процессинг запроса: '{prompt}'")
            search_results = await self.chroma_db.search(prompt)
            logger.info(f"Найдено {len(search_results)} результатов из ChromaDB")

            context_text = self._prepare_context(search_results)
            logger.debug(f"Подготовлен контекст для LLM: {context_text[:500]}...")

            answer = await self.llm_service.generate_response(prompt, context_text)
            logger.info(f"Длина сгенерированного ответа: {len(answer)} символов")

            confidence = self._calculate_confidence(search_results, answer)
            logger.info(f"Рассчитанная уверенность в ответе: {confidence}")

            processing_time = time.time() - start_time

            return QueryResponse(
                answer=answer,
                confidence=confidence,
                processing_time=processing_time,
            )

        except Exception as e:
            logger.error(f"Ошибка при обработке запроса: {e}", exc_info=True)
            processing_time = time.time() - start_time
            return QueryResponse(
                answer="При обработке запроса произошла ошибка.",
                confidence=0.0,
                processing_time=processing_time,
            )

    def _prepare_context(self, search_results: list[dict[str, Any]]) -> str:
        """Подготовка контекста из результатов поиска"""
        if not search_results:
            return "Информация не найдена в базе знаний."

        context_parts = []
        for _, result in enumerate(search_results[:3], 1):
            content = result.get("content", "")
            similarity = result.get("similarity_score", 0.0)

            context_parts.append(f"Релевантность: {similarity:.3f}:\n{content}\n")
        context = "\n---\n".join(context_parts)
        logger.info(
            f"Подготовлен контекст из {len(search_results[:3])} источников. Длина контекста: {len(context)}"
        )
        return context

    async def add_document(
        self,
        content: str,
        filename: str,
    ) -> bool:
        """Добавление документа в систему"""
        try:
            logger.info(f"Добавление документа: {filename}")
            chunks = self.text_splitter.split_text(content)
            logger.info(f"Документ '{filename}' разделен на {len(chunks)} чанков.")

            if not chunks:
                logger.warning(
                    f"В процессе разделения файла '{filename}' получилось 0 чанков."
                )
                return True

            ids = [f"{filename}_{i}" for i in range(len(chunks))]

            success = await self.chroma_db.add_documents(
                documents=chunks,
                ids=ids,
            )
            if success:
                logger.info(
                    f"Документ {filename} успешно добавлен (Кол-во чанков: {len(chunks)})."
                )
            else:
                logger.error(f"Не удалось добавить документ {filename} в ChromaDB.")
            return success

        except Exception as e:
            logger.error(
                f"При добавлении документа {filename} произошла ошибка: {e}",
                exc_info=True,
            )
            return False

    async def health_check(self):
        """Проверка работоспособности сервиса"""
        chroma_db_healthy = await self.chroma_db.health_check()
        llm_healthy = await self.llm_service.health_check()

        chroma_db_status = "healthy" if chroma_db_healthy else "unhealthy"
        llm_status = "healthy" if llm_healthy else "unhealthy"

        documents_count = 0
        if chroma_db_healthy:
            try:
                db_info = self.chroma_db.get_collection_info()
                documents_count = db_info.get("documents_count", 0)
            except Exception as e:
                logger.error(
                    f"При получении количества документов произошла ошибка: {e}."
                )
                chroma_db_status = f"error checking count: {e}"

        return {
            "chroma_db_status": chroma_db_status,
            "llm_status": llm_status,
            "documents_count": documents_count,
        }

    def _calculate_confidence(
        self, search_results: list[dict[str, Any]], answer: str
    ) -> float:
        """Расчет уверенности в ответе"""
        if not search_results:
            return 0.0

        top_results = sorted(
            search_results, key=lambda x: x.get("similarity_score", 0.0), reverse=True
        )[:3]

        if not top_results:
            return 0.0

        avg_similarity = sum(r.get("similarity_score", 0.0) for r in top_results) / len(
            top_results
        )
        max_similarity = top_results[0].get("similarity_score", 0.0)

        relevant_count = sum(
            1 for r in search_results if r.get("similarity_score", 0.0) > 0.01
        )
        relevance_factor = min(relevant_count / 3.0, 1.0)

        answer_length_factor = min(len(answer) / 200.0, 1.0)
        uncertainty_phrases = [
            "не знаю",
            "не могу",
            "недостаточно информации",
            "извините",
            "не удается",
            "не найдено",
            "отсутствует",
            "не предоставлен",
            "нет в контексте",
            "нет данных",
        ]
        has_uncertainty = any(
            phrase in answer.lower() for phrase in uncertainty_phrases
        )
        uncertainty_penalty = 0.3 if has_uncertainty else 0.0

        confidence = (
            avg_similarity * 0.4
            + max_similarity * 0.3
            + relevance_factor * 0.2
            + answer_length_factor * 0.1
        ) - uncertainty_penalty

        final_confidence = max(0.0, min(confidence, 1.0))

        logger.info(
            f"Расчет уверенности в ответе: avg_sim={avg_similarity:.4f}, max_sim={max_similarity:.4f}, "
            f"relevant_count={relevant_count}, uncertainty={has_uncertainty}, final={final_confidence:.4f}"
        )

        return round(final_confidence, 3)
