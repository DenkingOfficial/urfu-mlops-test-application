from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    api_host: str = "localhost"
    api_port: int = 8001
    debug: bool = True

    embedding_api_provider: str = ""
    embedding_api_model: str = ""
    embedding_api_key: Optional[str] = None

    llm_api_provider: str = ""
    llm_api_model: str = ""
    llm_api_key: Optional[str] = None

    gigachat_scope: Optional[str] = None
    mincifry_cert_path: Optional[str] = None
    verify_ssl_certs: Optional[bool] = None

    chroma_db_host: str = ""
    chroma_db_port: str = ""
    chroma_db_collection_name: str = ""

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
