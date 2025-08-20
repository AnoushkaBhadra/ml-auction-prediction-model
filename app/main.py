from fastapi import FastAPI
from app.routers import predict
from app.utils.logger import logger

app = FastAPI(title="ML Auction Price Prediction API")

# Include predict router
app.include_router(predict.router)

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "Welcome to the ML Auction Price Prediction API"}