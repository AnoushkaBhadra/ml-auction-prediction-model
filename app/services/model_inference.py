import pandas as pd
from pathlib import Path
import joblib
from typing import Dict
from app.config import settings
from app.utils.logger import logger


PRODUCT_GROUP_PREFIX = {
    "cylinder": "cyl",
    "cylinders": "cyl",
    "valve": "valve",
    "valves": "valve",
    "brass": "brass"
}

# Cache models to avoid reloading
MODELS = {}

def normalize_product_group(product_group: str) -> str:
    return PRODUCT_GROUP_PREFIX.get(product_group.lower(), product_group.lower())

def load_models(product_group: str, quantiles: list = ["q5", "q50", "q90"]) -> Dict[str, Dict[str, object]]:
    """
    Load models for both proposed_rp and last_bid_price, or brass index model.
    """
    try:
        prefix = normalize_product_group(product_group)
        model_key = prefix

        if prefix == "brass":
            model_path = Path(__file__).parent.parent.parent / "models" / "brass_index_model.joblib"
            if not model_path.exists():
                raise FileNotFoundError(f"Brass model not found: {model_path}")
            return {"proposed_rp": {"q50": joblib.load(model_path)}}

        if model_key not in MODELS:
            MODELS[model_key] = {"proposed_rp": {}, "lbp": {}}
            for target in ["proposed_rp", "lbp"]:
                for q in quantiles:
                    model_name = f"{prefix}_{target}_{q}.joblib"
                    model_path = Path(__file__).parent.parent.parent / "models" / model_name
                    if not model_path.exists():
                        logger.error(f"Model file not found: {model_path}")
                        raise FileNotFoundError(f"Model {model_name} not found")
                    MODELS[model_key][target][q] = joblib.load(model_path)
                    logger.info(f"Loaded model: {model_name}")
        return MODELS[model_key]

    except Exception as e:
        logger.error(f"Error loading models for {product_group}: {str(e)}")
        raise

def run_inference(product_group: str, quantity: float, input_date: str, feature_df: pd.DataFrame) -> Dict:
    """
    Run inference for both proposed_rp and last_bid_price (or brass index).
    """
    try:
        normalized_group = normalize_product_group(product_group)
        models = load_models(product_group)
        logger.info(f"Running inference for {product_group}, quantity={quantity}, date={input_date}")

        # Define expected features
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
                "brass_index_poly", "brass_index_momentum_7d",
                "brass_index_momentum_30d", "brass_index_volatility_7d"
            ]

        missing_features = [f for f in expected_features if f not in feature_df.columns]
        if missing_features:
            logger.error(f"Missing features: {missing_features}")
            raise ValueError(f"Missing required features: {missing_features}")

        feature_df = feature_df[expected_features]

        # Run predictions
        predictions = {}

        if normalized_group == "brass":
            q50 = float(models["proposed_rp"]["q50"].predict(feature_df)[0])
            predictions["proposed_rp"] = {"q50": q50, "confidence_interval": 0.0}
        else:
            for target in ["proposed_rp", "lbp"]:
                q_values = [float(models[target][q].predict(feature_df)[0]) for q in ["q5", "q50", "q90"]]
                q_values_sorted = sorted(q_values)  # Enforce q5 ≤ q50 ≤ q90
                predictions[target] = {
                    "q5": q_values_sorted[0],
                    "q50": q_values_sorted[1],
                    "q90": q_values_sorted[2],
                    "confidence_interval": q_values_sorted[2] - q_values_sorted[1]
                }

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