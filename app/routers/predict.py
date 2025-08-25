from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from pydantic.functional_validators import field_validator
from typing import Dict
from app.services.data_loader import load_historical_data, load_market_data
from app.services.preprocessor import preprocess_data
from app.services.model_inference import run_inference
from app.utils.logger import logger

router = APIRouter(prefix="/predict", tags=["predictions"])

class PredictionInput(BaseModel):
    product_group: str = Field(..., description="Product group: 'cylinder' or 'valve'")
    quantity: float = Field(..., gt=0, description="Quantity in MT")
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    location: str = Field(..., description="Auction location or state name (not used in model)")

    @field_validator("product_group")
    def validate_product_group(cls, v):
        v_lower = v.lower().strip()
        if any(word in v_lower for word in ["cylinder", "cylinders"]):
            return "cylinder"
        elif any(word in v_lower for word in ["valve", "valves"]):
            return "valve"
        raise ValueError("product_group must be 'cylinder' or 'valve' (singular/plural variations accepted)")

    @field_validator("date")
    def validate_date(cls, v):
        from datetime import datetime
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("date must be in YYYY-MM-DD format")

class PredictionOutput(BaseModel):
    product_group: str
    quantity: float
    date: str
    location: str  # Echoed back for presentation
    predictions: Dict[str, Dict[str, float]]

@router.post("/", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """
    Predict auction price quantiles (Q5, Q50, Q90) and confidence interval for proposed_rp and last_bid_price.
    Location is accepted but not used in model inference.
    """
    try:
        logger.info(f"Received prediction request: {input_data.dict()}")

        # Load data
        auction_df = load_historical_data(input_data.date)
        market_data = load_market_data()

        # Preprocess data (location is ignored)
        feature_df = preprocess_data(
            product_group=input_data.product_group,
            quantity=input_data.quantity,
            input_date=input_data.date,
            auction_df=auction_df,
            market_data=market_data
        )

        # Run inference (location is ignored)
        result = run_inference(
            product_group=input_data.product_group,
            quantity=input_data.quantity,
            input_date=input_data.date,
            feature_df=feature_df
        )

        # Add location back into response for optics
        result["location"] = input_data.location

        logger.info(f"Prediction successful: {result}")
        return result

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))