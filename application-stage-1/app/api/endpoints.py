from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Dict, Any
import logging

from app.models.schemas import (
    DocumentUploadResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)

router = APIRouter()

rag_service: RAGService | None = None


@router.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Обработка запроса пользователя

    Пример запроса:
    ```json
    {
        "prompt": "Когда начинается зимняя сессия?"
    }
    """
    if not rag_service:
        raise HTTPException(status_code=500, detail="RAG service not initialized")

    try:
        response = await rag_service.process_query(prompt=request.prompt)
        return response
    except Exception as e:
        logger.error(f"При обработке запроса произошла ошибка: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка состояния системы"""
    if not rag_service:
        logger.error("RAG-система не инициализирована")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")
    try:
        health_info = await rag_service.health_check()

        return HealthResponse(
            status=(
                "healthy"
                if health_info["llm_status"] == "healthy"
                and health_info["chroma_db_status"] == "healthy"
                else "degraded"
            ),
            chroma_db_status=health_info["chroma_db_status"],
            llm_status=health_info["llm_status"],
            documents_count=health_info["documents_count"],
        )
    except Exception as e:
        logger.error(f"При проверке работоспособности системы произошла ошибка: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


@router.post("/upload-document", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Загрузка нового документа в систему
    Поддерживаемые форматы: .txt
    """

    if not rag_service:
        logger.error("RAG-система не инициализирована")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")

    if file.filename and not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Для добавления поддерживаются только .txt файлы")

    try:
        content = await file.read()
        text_content = content.decode("utf-8")

        if file.filename:
            success = await rag_service.add_document(text_content, file.filename)
            if success:
                chunks = rag_service.text_splitter.split_text(text_content)
                return DocumentUploadResponse(
                    message="Документ успешно добавлен",
                    filename=file.filename,
                    chunks_count=len(chunks),
                    success=True,
                )
            else:
                raise HTTPException(
                    status_code=500, detail="Внутренняя ошибка сервера"
                )
        else:
            raise HTTPException(status_code=400, detail="Файл отсутствует")

    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400, detail="Кодировка файла не поддерживается, используйте UTF-8."
        )
    except Exception as e:
        logger.error(f"При загрузке документа произошла ошибка: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


@router.get("/documents/count")
async def get_documents_count() -> Dict[str, Any]:
    """Получение количества документов в системе"""
    if not rag_service:
        logger.error("RAG-система не инициализирована")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")
    try:
        info = rag_service.chroma_db.get_collection_info()
        return info
    except Exception as e:
        logger.error(f"При подсчете количества документов произошла ошибка: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


@router.get("/embedding/test")
async def test_embedding():
    """Тестирование сервиса эмбеддингов"""
    if not rag_service:
        logger.error("RAG-система не инициализирована")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")
    try:
        test_text = "Это тестовый текст для проверки эмбеддингов"
        embedding = await rag_service.chroma_db.embedding_service.client.aembed_query(
            test_text
        )

        return {
            "text": test_text,
            "embedding_dimension": len(embedding) if embedding else 0,
            "embedding_preview": embedding[:5] if embedding else [],
            "success": len(embedding) > 0 if embedding else False,
        }
    except Exception as e:
        logger.error(
            f"При проверке работоспособности сервиса эмбеддингов произошла ошибка: {e}"
        )
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")


def set_rag_service(service: RAGService):
    """Установка RAG сервиса (вызывается из main.py)"""
    global rag_service
    rag_service = service
