import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import requests
from typing import Dict, Tuple
from app.config import settings
from app.utils.logger import logger

# In-memory cache for API results
METAL_PRICE_CACHE = {}

def load_historical_data(input_date: str) -> pd.DataFrame:
    """
    Load historical auction data from Excel.

    Args:
        input_date (str): Date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: Filtered auction data within historical limit.
    """
    try:
        parsed_date = pd.to_datetime(input_date)
        start_date = parsed_date - timedelta(days=settings.HISTORICAL_DATA_LMT * 365)
        file_path = Path(__file__).parent.parent.parent / "data" / "AuctionData.xlsx"
        df = pd.read_excel(file_path)
        df["auction_date"] = pd.to_datetime(df["auction_date"])
        df = df[df["auction_date"] >= start_date]
        logger.info(f"Loaded {len(df)} rows of historical data from {start_date.date()} to {input_date}")
        return df
    except Exception as e:
        logger.error(f"Error loading historical data: {str(e)}")
        raise

def load_metal_prices(input_date: str, lookback_days: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch copper and zinc prices from MetalpriceAPI for the input date and lookback period.

    Args:
        input_date (str): Date in 'YYYY-MM-DD' format.
        lookback_days (int): Number of days to fetch historical prices (default: 30).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Copper and zinc price DataFrames.
    """
    try:
        parsed_date = pd.to_datetime(input_date)
        start_date = (parsed_date - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        end_date = input_date
        cache_key = f"{start_date}_{end_date}"

        # Check cache
        if cache_key in METAL_PRICE_CACHE:
            logger.info(f"Using cached metal prices for {cache_key}")
            return METAL_PRICE_CACHE[cache_key]

        # API request
        url = "https://api.metalpriceapi.com/v1/timeframe"
        params = {
            "api_key": settings.METALPRICE_API_KEY,
            "start_date": start_date,
            "end_date": end_date,
            "base": "INR",
            "currencies": "LME-XCU,LME-XZN"
        }

        for attempt in range(3):
            try:
                response = requests.get(url, params=params, timeout=5)
                response.raise_for_status()
                data = response.json()
                if not data.get("success"):
                    logger.error(f"API error: {data.get('error', 'Unknown error')}")
                    raise ValueError("API request failed")
                break
            except requests.RequestException as e:
                logger.warning(f"API attempt {attempt + 1} failed: {str(e)}")
                if attempt == 2:
                    logger.error("Max retries reached, using fallback prices")
                    return _get_fallback_metal_prices(parsed_date, lookback_days)

        # Parse API response
        copper_data = []
        zinc_data = []
        for date, rates in data.get("rates", {}).items():
            copper_data.append({
                "date": date,
                "spot_price_copper": rates.get("LME-XCU", settings.FALLBACK_COPPER_PRICE)
            })
            zinc_data.append({
                "date": date,
                "spot_price_zinc": rates.get("LME-XZN", settings.FALLBACK_ZINC_PRICE)
            })

        # Create DataFrames
        copper_df = pd.DataFrame(copper_data)
        zinc_df = pd.DataFrame(zinc_data)
        copper_df["date"] = pd.to_datetime(copper_df["date"])
        zinc_df["date"] = pd.to_datetime(zinc_df["date"])

        # Validate
        if copper_df.empty or zinc_df.empty:
            logger.warning(f"Empty API response for {start_date} to {end_date}, using fallback")
            return _get_fallback_metal_prices(parsed_date, lookback_days)

        # Cache results
        METAL_PRICE_CACHE[cache_key] = (copper_df, zinc_df)
        logger.info(f"Fetched metal prices for {start_date} to {end_date}: {len(copper_df)} rows")
        return copper_df, zinc_df

    except Exception as e:
        logger.error(f"Error fetching metal prices: {str(e)}")
        return _get_fallback_metal_prices(parsed_date, lookback_days)

def _get_fallback_metal_prices(parsed_date: datetime, lookback_days: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return fallback copper and zinc prices for the given date range.

    Args:
        parsed_date (datetime): Parsed input date.
        lookback_days (int): Number of days for lookback.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Fallback copper and zinc DataFrames.
    """
    dates = pd.date_range(end=parsed_date, periods=lookback_days + 1)
    copper_df = pd.DataFrame({
        "date": dates,
        "spot_price_copper": settings.FALLBACK_COPPER_PRICE
    })
    zinc_df = pd.DataFrame({
        "date": dates,
        "spot_price_zinc": settings.FALLBACK_ZINC_PRICE
    })
    logger.warning(f"Using fallback prices: copper={settings.FALLBACK_COPPER_PRICE}, zinc={settings.FALLBACK_ZINC_PRICE}")
    return copper_df, zinc_df

    
def load_market_data(input_date: str = None) -> dict:
    """
    Load market data (copper/zinc prices) for brass index calculation.

    Args:
        input_date (str, optional): Date in 'YYYY-MM-DD' format.

    Returns:
        dict: Market data with original column names preserved.
    """
    try:
        if settings.DATA_SOURCE == "demo":
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
        elif settings.DATA_SOURCE == "api":
            copper_df, zinc_df = load_metal_prices(input_date or datetime.today().strftime("%Y-%m-%d"))
            copper_df.rename(columns={"spot_price_copper": "spot price(rs.)_copper"}, inplace=True)
            zinc_df.rename(columns={"spot_price_zinc": "spot price(rs.)_zinc"}, inplace=True)
            return {"copper": copper_df, "zinc": zinc_df}
        else:
            logger.warning("Unknown data source; using hardcoded prices")
            return {
                "copper": pd.DataFrame({"spot price(rs.)_copper": [settings.COPPER_PRICE_DEFAULT]}),
                "zinc": pd.DataFrame({"spot price(rs.)_zinc": [settings.ZINC_PRICE_DEFAULT]})
            }
    except Exception as e:
        logger.error(f"Error loading market data: {str(e)}")
        raise