from pydantic import BaseModel, Field
from datetime import datetime


class QueryRequest(BaseModel):
    prompt: str = Field(
        "Когда начинается зимняя сессия?",
        min_length=1,
        max_length=500,
        description="Запрос пользователя",
    )


class QueryResponse(BaseModel):
    answer: str = Field(
        ...,
        description="Ответ на запрос",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Уверенность модели в ответе",
    )
    processing_time: float = Field(
        ...,
        description="Время обработки запроса",
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Время запроса",
    )


class HealthResponse(BaseModel):
    status: str
    chroma_db_status: str
    llm_status: str
    documents_count: int
    timestamp: datetime = Field(default_factory=datetime.now)


class DocumentUploadResponse(BaseModel):
    message: str
    filename: str
    chunks_count: int
    success: bool
