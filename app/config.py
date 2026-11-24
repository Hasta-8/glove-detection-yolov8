"""Settings for the API, from environment variables (prefix GLOVE_) and `.env`."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from detector import DEFAULT_CONFIDENCE, DEFAULT_MODEL

MODEL_SHA256 = "cbceb1a10dc68f2377b4f351a4309771147c48598f138620bea045cd30b93be8"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="GLOVE_", extra="ignore")

    model_path: Path = DEFAULT_MODEL
    demo_dir: Path = Path("demo")
    confidence: float = Field(default=DEFAULT_CONFIDENCE, ge=0.0, le=1.0)
    torch_threads: int = Field(default=4, ge=1)
    max_upload_mb: float = Field(default=10.0, gt=0)
    max_pixels: int = Field(default=40_000_000, gt=0)
    min_side_px: int = Field(default=32, ge=1)
    busy_timeout_s: float = Field(default=30.0, gt=0)


@lru_cache
def get_settings() -> Settings:
    return Settings()
