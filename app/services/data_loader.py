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
        pd.DataFrame: Filtered auction data within historical limit.
    """
    try:
        parsed_date = pd.to_datetime(input_date)
        cutoff_date = parsed_date - timedelta(days=365 * settings.HISTORICAL_DATA_LMT)

        data_path = Path(__file__).parent.parent.parent / "data" / "AuctionData.xlsx"
        df = pd.read_excel(data_path)
        logger.info(f"Loaded {len(df)} rows from {data_path}")

        df.columns = df.columns.str.lower().str.strip()
        df.rename(columns={"auction date": "auction_date"}, inplace=True)

        if "auction_date" not in df.columns:
            logger.error("Missing 'auction_date' column in data")
            raise ValueError("Missing 'auction_date' column in data")

        df["auction_date"] = pd.to_datetime(df["auction_date"])
        df = df[df["auction_date"] >= cutoff_date]

        logger.info(f"Filtered to {len(df)} rows within {settings.HISTORICAL_DATA_LMT} years")
        return df
    except Exception as e:
        logger.error(f"Error loading historical data: {str(e)}")
        raise

def load_market_data() -> dict:
    """
    Load market data (copper/zinc prices) for brass index calculation.

    Returns:
        dict: Market data with original column names preserved.
    """
    try:
        copper_path = Path(__file__).parent.parent.parent / "data" / "marketdata_copper.xlsx"
        zinc_path = Path(__file__).parent.parent.parent / "data" / "marketdata_zinc.xlsx"

        df_copper = pd.read_excel(copper_path)
        df_zinc = pd.read_excel(zinc_path)

        logger.info(f"Loaded copper data: {len(df_copper)} rows, zinc data: {len(df_zinc)} rows")

        df_copper.columns = df_copper.columns.str.lower().str.strip()
        df_zinc.columns = df_zinc.columns.str.lower().str.strip()

        df_copper.rename(columns={"spot price(rs.)": "spot price(rs.)_copper"}, inplace=True)
        df_zinc.rename(columns={"spot price(rs.)": "spot price(rs.)_zinc"}, inplace=True)

        return {"copper": df_copper, "zinc": df_zinc}
    except Exception as e:
        logger.error(f"Error loading market data: {str(e)}")
        raise