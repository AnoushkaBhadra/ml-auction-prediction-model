import pandas as pd
import numpy as np
import joblib

# --- 1. Load auction data ---
data_path = '../data/AuctionData.xlsx'
df = pd.read_excel(data_path)
df.columns = df.columns.str.lower()
df['auction_date'] = pd.to_datetime(df['auction_date'])

# --- 2. Filter for valves and date range ---
if 'product_group' not in df.columns:
    valves = [x.lower().strip() for x in ['SC Valve', 'LIQUID OFFTAKE VALVE']]
    def map_product_type(desc):
        desc = str(desc).lower().strip()
        return 'valves' if desc in valves else 'other'
    df['product_group'] = df['productdescription'].apply(map_product_type)

valve_df = df[
    (df['auction_date'] >= '2025-01-01') &
    (df['auction_date'] <= '2025-07-07') &
    (df['product_group'] == 'valves')
].copy()

# --- 3. Feature engineering ---
valve_df['year'] = valve_df['auction_date'].dt.year
valve_df['month'] = valve_df['auction_date'].dt.month
valve_df['rp_per_qty'] = valve_df['proposed_rp'] / valve_df['quantity']
valve_df['amt_per_qty'] = valve_df['total_amt'] / valve_df['quantity']
valve_df['quantity_clipped'] = valve_df['quantity'].clip(upper=100)  # adjust as per your training
valve_df['last_bid_price_clipped'] = valve_df['last_bid_price'].clip(upper=100000)  # adjust as per your training

# --- 4. Market data merge and brass index calculation ---
# Define or import cu_price and zn_price dictionaries here
# Example:
# from market_prices import cu_price, zn_price

# For this script, you must define cu_price and zn_price dicts above this code block
cu_df = pd.DataFrame(list(cu_price.items()), columns=['date', 'cu_price'])
cu_df['date'] = pd.to_datetime(cu_df['date'])
zn_df = pd.DataFrame(list(zn_price.items()), columns=['date', 'zn_price'])
zn_df['date'] = pd.to_datetime(zn_df['date'])

valve_df = valve_df.sort_values('auction_date')
cu_df = cu_df.sort_values('date')
zn_df = zn_df.sort_values('date')

valve_df = pd.merge_asof(
    valve_df, cu_df, left_on='auction_date', right_on='date', direction='backward'
).drop(columns=['date'])
valve_df = pd.merge_asof(
    valve_df, zn_df, left_on='auction_date', right_on='date', direction='backward'
).drop(columns=['date'])

brass_index_pipeline = joblib.load('../models/brass_index_model.joblib')
valve_df_renamed = valve_df.rename(columns={
    'cu_price': 'spot price(rs.)_copper',
    'zn_price': 'spot price(rs.)_zinc'
})
valve_df_renamed = valve_df_renamed.dropna(subset=['spot price(rs.)_copper', 'spot price(rs.)_zinc'])
valve_df = valve_df.loc[valve_df_renamed.index]
valve_df['brass_index_poly'] = brass_index_pipeline.predict(
    valve_df_renamed[['spot price(rs.)_copper', 'spot price(rs.)_zinc']]
)

# --- 5. Model prediction ---
valve_features = ['month', 'year', 'rp_per_qty', 'amt_per_qty', 'quantity_clipped', 'last_bid_price_clipped', 'brass_index_poly']
model_q25 = joblib.load('../models/valve_proposed_rp_q25.joblib')
model_q50 = joblib.load('../models/valve_proposed_rp_q50.joblib')
model_q75 = joblib.load('../models/valve_proposed_rp_q75.joblib')

X_pred = valve_df[valve_features]
valve_df['pred_q25'] = model_q25.predict(X_pred)
valve_df['pred_q50'] = model_q50.predict(X_pred)
valve_df['pred_q75'] = model_q75.predict(X_pred)

# --- 6. Evaluation ---
valve_df['actual'] = valve_df['proposed_rp']
valve_df['abs_error'] = (valve_df['pred_q50'] - valve_df['actual']).abs()
valve_df['pct_error'] = 100 * valve_df['abs_error'] / valve_df['actual']
valve_df['in_range'] = valve_df.apply(
    lambda row: min(row['pred_q25'], row['pred_q75']) <= row['actual'] <= max(row['pred_q25'], row['pred_q75']),
    axis=1
)

# --- 7. Output sample and metrics ---
print('Sample predictions:')
print(valve_df[['auction_date', 'quantity', 'actual', 'pred_q25', 'pred_q50', 'pred_q75', 'abs_error', 'pct_error', 'in_range']].head(10))

mae = valve_df['abs_error'].mean()
rmse = np.sqrt(((valve_df['pred_q50'] - valve_df['actual']) ** 2).mean())
coverage = valve_df['in_range'].mean() * 100
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"Coverage (actual in Q25-Q75): {coverage:.1f}%")

# --- 8. Save results ---
valve_df[['auction_date', 'quantity', 'actual', 'pred_q25', 'pred_q50', 'pred_q75', 'abs_error', 'pct_error', 'in_range']].to_excel('valve_rp_eval_results.xlsx', index=False)
print("Results saved to valve_rp_eval_results.xlsx") 