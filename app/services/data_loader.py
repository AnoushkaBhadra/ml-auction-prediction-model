import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from app.config import settings
from app.utils.logger import logger 

def load_historical_data(input_date: str) -> pd.DataFrame:
    """
    Load and filter historical auction data based on input date.
    Args:
        input_date (str): Date in 'YYYY-MM-DD' format (e.g., '2025-09-12').
    Returns:
        pd.DataFrame: Filtered auction data within 3 years.
    """
    try:
        parsed_date = datetime.strptime(input_date, "%Y-%m-%d")
        logger.info(f"Parsed input date: {parsed_date}")

        if settings.DATA_SOURCE == "demo":
            data_path = Path(__file__).parent.parent.parent / "data" / "AuctionData.xlsx"
            df = pd.read_excel(data_path)
            logger.info(f"Loaded {len(df)} rows from {data_path}")
        else:
            logger.warning("DB not implemented; using demo data as fallback")
            data_path = Path(__file__).parent.parent.parent / "data" / "AuctionData.xlsx"
            df = pd.read_excel(data_path)

        df.columns = df.columns.str.lower().str.strip()
        logger.info(f"Normalized column names: {list(df.columns)}")

        cutoff_date = parsed_date - timedelta(days = 365 * settings.HISTORICAL_DATA_LMT)
        if "auction_date" not in df.columns:
            logger.error("Missing 'auction_date' column in data")
            raise ValueError("Missing 'auction_date' column in data")
        df["auction_date"] = pd.to_datetime(df["auction_date"])
        df = df[df["auction_date"] >= cutoff_date]
        logger.info(f'Filtered to {len(df)} rows within {settings.HISTORICAL_DATA_LMT} years')

        return df
    except Exception as e:
        logger.error(f"Error loading historical data: {str(e)}")
        raise 

def load_market_data() -> pd.DataFrame:
    """
    Load market data (copper/zinc prices) for brass index calculation.
    Returns:
        pd.DataFrame: Market data (demo or placeholder).
    """
    try:
        if settings.DATA_SOURCE == "demo":
            copper_path = Path(__file__).parent.parent.parent / "data" / "marketdata_copper.xlsx"
            zinc_path = Path(__file__).parent.parent.parent / "data" / "marketdata_zinc.xlsx"
            df_copper = pd.read_excel(copper_path)
            df_zinc = pd.read_excel(zinc_path)

            logger.info(f"Loaded copper data from {len(df_copper)} rows, zinc data: {len(df_zinc)}")

            df_copper.columns = df_copper.columns.str.lower().str.strip()
            df_zinc.columns = df_zinc.columns.str.lower().str.strip()

            return {"copper" : df_copper, "zinc":df_zinc}
        else:
            logger.warning("Market API not implemented; using hardcoded prices")
        return {
            "copper": pd.DataFrame({"spot_price_copper": [settings.COPPER_PRICE_DEFAULT]}),
            "zinc": pd.DataFrame({"spot_price_zinc": [settings.ZINC_PRICE_DEFAULT]})
        }
    except Exception as e:
        logger.error(f"Error loading market data: {str(e)}")
        raise