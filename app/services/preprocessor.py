import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import joblib
from app.config import settings
from app.utils.logger import logger
from app.services.data_loader import load_historical_data, load_market_data

def map_product_group(product_description: str) -> str:
    product = product_description.lower().strip()

    cylinders = {
        '14.2 kg', '19 kg', '5 kg', '47.5 kg', '5 kg ftlr', '5 kg nd',
        '47.5 kg lotv', '19 kg sc', '19 kg ncut', '5 kg ftl', '14.2 kg omc'
    }
    valves = {'sc valve', 'liquid offtake valve'}

    if product in cylinders:
        return 'cylinders'
    elif product in valves:
        return 'valves'
    else:
        return 'other'

def compute_brass_index(copper_price: float, zinc_price: float) -> float:
    try:
        model_path = Path(__file__).parent.parent.parent / "models" / "brass_index_model.joblib"
        model = joblib.load(model_path)
        input_data = pd.DataFrame({
            "spot price(rs.)_copper": [copper_price],
            "spot price(rs.)_zinc": [zinc_price]
        })
        brass_index = model.predict(input_data)[0]
        logger.info(f"Computed brass index: {brass_index} for copper={copper_price}, zinc={zinc_price}")
        return brass_index
    except Exception as e:
        logger.error(f"Error computing brass index: {str(e)}")
        raise

def preprocess_data(product_group: str, quantity: float, input_date: str, auction_df: pd.DataFrame, market_data: dict) -> pd.DataFrame:
    try:
        parsed_date = datetime.strptime(input_date, "%Y-%m-%d")
        logger.info(f"Preprocessing for {product_group}, quantity={quantity}, date={input_date}")

        features = {
            "year": parsed_date.year,
            "month": parsed_date.month,
            "day_of_week": parsed_date.weekday(),
            "day_of_month": parsed_date.day,
            "week_of_year": parsed_date.isocalendar().week,
            "quantity": quantity
        }

        auction_df["product_group"] = auction_df["productdescription"].apply(map_product_group)
        auction_df = auction_df[auction_df["product_group"] == product_group]
        logger.info(f"Filtered auction data to {len(auction_df)} rows for {product_group}")

        if auction_df.empty:
            logger.error(f"No data for {product_group} after filtering")
            raise ValueError(f"No historical data for {product_group}")

        auction_df = auction_df.sort_values("auction_date")

        last_auction = auction_df.iloc[-1] if not auction_df.empty else None
        features["days_since_last_auction"] = (
            (parsed_date - last_auction["auction_date"]).days
            if last_auction is not None else 0
        )
        features["last_auction_price"] = last_auction["proposed_rp"] if last_auction is not None else 0
        features["last_auction_quantity"] = last_auction["quantity"] if last_auction is not None else 0

        features["ewm_proposed_rp"] = auction_df["proposed_rp"].ewm(span=7).mean().iloc[-1]
        features["ewm_last_bid_price"] = auction_df["last_bid_price"].ewm(span=7).mean().iloc[-1]

        for window in [7, 30]:
            window_timedelta = timedelta(days=window)
            window_df = auction_df[auction_df["auction_date"] >= parsed_date - window_timedelta]

            if window == 7:
                features[f"auction_frequency_{window}d"] = len(window_df)

            features[f"price_momentum_{window}d"] = (
                (window_df["proposed_rp"].iloc[-1] - window_df["proposed_rp"].iloc[0]) / window_df["proposed_rp"].iloc[0] * 100
                if not window_df.empty and len(window_df) > 1 else 0
            )
            features[f"quantity_trend_{window}d"] = (
                (window_df["quantity"].iloc[-1] - window_df["quantity"].iloc[0]) / window_df["quantity"].iloc[0] * 100
                if not window_df.empty and len(window_df) > 1 else 0
            )
            features[f"price_volatility_{window}d"] = window_df["proposed_rp"].std() if not window_df.empty else 0
            features[f"rolling_mean_{window}d_proposed_rp"] = window_df["proposed_rp"].rolling(window).mean().iloc[-1] if not window_df.empty else 0
            features[f"rolling_mean_{window}d_last_bid_price"] = window_df["last_bid_price"].rolling(window).mean().iloc[-1] if not window_df.empty else 0
            features[f"rolling_std_{window}d_proposed_rp"] = window_df["proposed_rp"].rolling(window).std().iloc[-1] if not window_df.empty else 0
            features[f"rolling_std_{window}d_last_bid_price"] = window_df["last_bid_price"].rolling(window).std().iloc[-1] if not window_df.empty else 0
            features[f"price_change_{window}d"] = (
                (window_df["proposed_rp"].iloc[-1] - window_df["proposed_rp"].iloc[0]) / window_df["proposed_rp"].iloc[0] * 100
                if not window_df.empty and len(window_df) > 1 else 0
            )
            features[f"quantity_change_{window}d"] = (
                (window_df["quantity"].iloc[-1] - window_df["quantity"].iloc[0]) / window_df["quantity"].iloc[0] * 100
                if not window_df.empty and len(window_df) > 1 else 0
            )

        one_day_df = auction_df[auction_df["auction_date"] >= parsed_date - timedelta(days=1)]
        features["price_change_1d"] = (
            (one_day_df["proposed_rp"].iloc[-1] - one_day_df["proposed_rp"].iloc[0]) / one_day_df["proposed_rp"].iloc[0] * 100
            if not one_day_df.empty and len(one_day_df) > 1 else 0
        )
        features["quantity_change_1d"] = (
            (one_day_df["quantity"].iloc[-1] - one_day_df["quantity"].iloc[0]) / one_day_df["quantity"].iloc[0] * 100
            if not one_day_df.empty and len(one_day_df) > 1 else 0
        )

        if product_group.lower().rstrip("s") == "valve":

            copper_price = market_data["copper"]["spot price(rs.)_copper"].iloc[-1] if not market_data["copper"].empty else settings.COPPER_PRICE_DEFAULT
            zinc_price = market_data["zinc"]["spot price(rs.)_zinc"].iloc[-1] if not market_data["zinc"].empty else settings.ZINC_PRICE_DEFAULT
            brass_index = compute_brass_index(copper_price, zinc_price)
            features["brass_index_poly"] = brass_index

            brass_indices = []
            for i in range(-30, 1):
                date = parsed_date + timedelta(days=i)
                temp_auction_df = auction_df[auction_df["auction_date"].dt.date == date.date()]
                if not temp_auction_df.empty:
                    temp_copper = market_data["copper"][market_data["copper"]["date"].dt.date == date.date()]["spot price(rs.)_copper"]
                    temp_zinc = market_data["zinc"][market_data["zinc"]["date"].dt.date == date.date()]["spot price(rs.)_zinc"]
                    temp_copper = temp_copper.iloc[-1] if not temp_copper.empty else settings.COPPER_PRICE_DEFAULT
                    temp_zinc = temp_zinc.iloc[-1] if not temp_zinc.empty else settings.ZINC_PRICE_DEFAULT
                    brass_indices.append(compute_brass_index(temp_copper, temp_zinc))
            brass_indices = pd.Series(brass_indices)
            features["brass_index_momentum_7d"] = (
                (brass_indices.iloc[-1] - brass_indices.iloc[-8]) / brass_indices.iloc[-8] * 100
                if len(brass_indices) > 7 else 0
            )
            features["brass_index_momentum_30d"] = (
                (brass_indices.iloc[-1] - brass_indices.iloc[0]) / brass_indices.iloc[0] * 100
                if len(brass_indices) > 1 else 0
            )
            features["brass_index_volatility_7d"] = brass_indices[-7:].std() if len(brass_indices) >= 7 else 0

        feature_df = pd.DataFrame([features])
        logger.info(f"Generated features: {list(feature_df.columns)}")
        return feature_df

    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise