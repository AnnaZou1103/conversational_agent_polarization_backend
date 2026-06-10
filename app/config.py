from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM configuration
    llm_provider: str = "openai"  # "openai", "anthropic", or "azure"
    llm_model: str = "gpt-4o"
    llm_api_key: str = ""
    llm_base_url: str | None = None  # optional override for OpenAI-compatible endpoints
    llm_api_version: str = "2024-12-01-preview"  # Azure OpenAI API version

    # Fast model used for internal pipeline steps (OBSERVE / THINK / stage
    # evaluation) where a cheaper, lower-latency model is sufficient. These run
    # on the critical path before the user sees any tokens, so shaving their
    # latency directly reduces perceived response time. Empty falls back to
    # llm_model (no behavior change). Same provider/key as the main model.
    fast_llm_model: str = ""  # e.g. "gpt-4o-mini" or "claude-haiku-4-5-20251001"

    # Server configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8080

    # Agent configuration
    default_strategy: str = (
        "common_identity"  # "common_identity" or "personal_narrative"
    )
    enable_think: bool = False  # enable internal reasoning step before each response

    # Admin
    admin_password: str

    # Logging
    log_level: str = "info"
    conversations_dir: str = (
        "conversations"  # folder where per-session JSONL logs are saved
    )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
