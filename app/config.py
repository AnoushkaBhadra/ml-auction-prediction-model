from pydantic_settings import BaseSettings
from pydantic import ConfigDict

class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")

    DATA_SOURCE: str = "demo"
    COPPER_PRICE_DEF: float = 800000
    ZINC_PRICE_DEF: float = 300000
    HISTORICAL_DATA_LMT: int = 3
    DB_URL: str = ""
    MARKET_API_URL: str = ""

    #Add these to avoid ValidationError
    metalprice_api_key: str
    usd_to_inr: str

settings = Settings()
