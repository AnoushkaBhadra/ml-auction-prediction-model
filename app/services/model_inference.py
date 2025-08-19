import pandas as pd
from pathlib import Path
import joblib
from typing import Dict
from app.config import settings
from app.utils.logger import logger
import os

PRODUCT_GROUP_PREFIX = {
    "cylinder": "cyl",
    "cylinders": "cyl",  # handles plural
    "valve": "valve",
    "valves": "valve",
    "brass": "brass"
}

def normalize_product_group(product_group: str) -> str:
    return PRODUCT_GROUP_PREFIX.get(product_group.lower(), product_group.lower())

def load_models(product_group: str) -> Dict[str, joblib]:
    """
    Load all required quantile models for the given product group.

    Args:
        product_group (str): One of 'cylinder', 'valve', or 'brass'.

    Returns:
        Dict[str, model]: Dictionary of models keyed by quantile.
    """
    models_folder = Path(__file__).parent.parent.parent / "models"
    quantiles = ["q5", "q50", "q90"]
    prefix = normalize_product_group(product_group)

    if not prefix:
        raise ValueError(f"Unsupported product group: {product_group}")

    models = {}

    for q in quantiles:
        if prefix == "brass":
            model_filename = f"{prefix}_index_model.joblib"
            if "q50" not in models:
                model_path = models_folder / model_filename
                if not model_path.exists():
                    raise FileNotFoundError(f"Model file not found: {model_path}")
                models["q50"] = joblib.load(model_path)
            break
        else:
            model_filename = f"{prefix}_proposed_rp_{q}.joblib"
            model_path = models_folder / model_filename
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            models[q] = joblib.load(model_path)

    return models

def run_inference(product_group: str, quantity: float, input_date: str, feature_df: pd.DataFrame) -> Dict:
    """
    Run model inference and return predictions.
    Args:
        product_group (str): 'cylinder', 'valve', or 'valves'.
        quantity (float): Quantity in MT.
        input_date (str): Date in 'YYYY-MM-DD' format.
        feature_df (pd.DataFrame): Preprocessed features from preprocessor.py.
    Returns:
        Dict: Prediction result in required format.
    """
    try:
        normalized_group = normalize_product_group(product_group)

        # Load models
        models = load_models(product_group)
        logger.info(f"Running inference for {product_group}, quantity={quantity}, date={input_date}")

        # Validate features
        expected_features = [
            "year", "month", "quantity", "ewm_proposed_rp", "ewm_last_bid_price",
            "day_of_week", "day_of_month", "week_of_year", "days_since_last_auction",
            "auction_frequency_7d", "price_momentum_7d", "price_momentum_30d",
            "quantity_trend_7d", "quantity_trend_30d", "last_auction_price",
            "last_auction_quantity", "price_volatility_7d", "price_volatility_30d",
            "rolling_mean_7d_proposed_rp", "rolling_mean_30d_proposed_rp",
            "rolling_mean_7d_last_bid_price", "rolling_mean_30d_last_bid_price",
            "rolling_std_7d_proposed_rp", "rolling_std_30d_proposed_rp",
            "rolling_std_7d_last_bid_price", "rolling_std_30d_last_bid_price",
            "price_change_1d", "price_change_7d", "price_change_30d",
            "quantity_change_1d", "quantity_change_7d", "quantity_change_30d"
        ]

        if normalized_group == "valve":
            expected_features += [
                "brass_index_poly",
                "brass_index_momentum_7d",
                "brass_index_momentum_30d",
                "brass_index_volatility_7d"
            ]

        missing_features = [f for f in expected_features if f not in feature_df.columns]
        feature_df = feature_df[expected_features]

        if missing_features:
            logger.error(f"Missing features: {missing_features}")
            raise ValueError(f"Missing required features: {missing_features}")

        # Run predictions
        predictions = {}
        for q in ["q5", "q50", "q90"]:
            predictions[q] = float(models[q].predict(feature_df)[0])
        predictions["confidence_interval"] = predictions["q90"] - predictions["q50"]

        # Format output
        result = {
            "product_group": product_group,
            "quantity": quantity,
            "date": input_date,
            "predictions": predictions
        }
        logger.info(f"Generated predictions: {predictions}")
        return result

    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise