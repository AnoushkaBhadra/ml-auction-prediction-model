from fastapi import FastAPI
from app.utils.logger import logger

app = FastAPI(title ="ML Auction Price Prediction API")

@app.get("/") #Basic root endpoint

async def root():
    logger.info("Root endpoint accessed")
    return{"message":"Welcome to ML Auction Price Prediction API"}
