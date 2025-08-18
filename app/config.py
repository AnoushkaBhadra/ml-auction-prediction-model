#This file defines the Settings using pydantic-settings to load and validate .env variables. 
from pydantic_settings import BaseSettings
class Settings(BaseSettings):
    DATA_SOURCE: str = "demo" #default demo data in case of API failure
    COPPER_PRICE_DEF: float = 800000 #Fallback copper price
    ZINC_PRICE_DEF: float = 300000 #Fallback zinc price
    HISTORICAL_DATA_LMT : int = 3  #Years for storing historical data
    DB_URL: str  = ""
    MARKET_API_URL: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

