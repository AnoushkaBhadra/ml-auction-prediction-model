import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from app.services.data_loader import load_historical_data, load_market_data

# Test historical data
df_auction = load_historical_data("2025-09-12")
print(f"Auction data rows: {len(df_auction)}")
print(df_auction.head())

# Test market data
market_data = load_market_data()
print("Copper data:", market_data["copper"].head())
print("Zinc data:", market_data["zinc"].head())