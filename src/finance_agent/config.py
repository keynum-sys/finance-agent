"""全局配置。用 pydantic-settings 从环境变量 / .env 加载。"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # LLM
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_strong_model: str = "gpt-4o"

    # 路径
    data_dir: Path = Path("data_cache")
    chroma_dir: Path = Path("chroma_store")

    # 抽取参数
    extraction_max_retries: int = 3      # Pydantic 校验失败重试次数
    numeric_tolerance: float = 0.005     # 数值相对误差容忍 0.5%

    # 异常检测阈值
    yoy_anomaly_threshold: float = 0.5   # 同比变动超过 ±50% 触发预警


settings = Settings()
settings.data_dir.mkdir(exist_ok=True)
settings.chroma_dir.mkdir(exist_ok=True)
