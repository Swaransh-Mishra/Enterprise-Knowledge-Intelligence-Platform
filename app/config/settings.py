"""
Application Configuration.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    # -------------------------
    # Application
    # -------------------------

    APP_NAME: str = "Enterprise Knowledge Intelligence Platform"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # -------------------------
    # LLM
    # -------------------------

    LLM_PROVIDER: str = "watsonx"
    LLM_MODEL: str = "meta-llama/llama-3-3-70b-instruct"

    # -------------------------
    # IBM watsonx.ai
    # -------------------------

    WATSONX_URL: str = ""
    WATSONX_PROJECT_ID: str = ""
    WATSONX_APIKEY: str = ""

    class Config:
        env_file = ".env"


settings = Settings()