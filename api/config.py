from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str
    SERP_API_KEY: str | None = None
    SLACK_WEBHOOK_URL: str | None = None
    API_AUTH_TOKEN: str | None = None  # optional simple auth for endpoints

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()