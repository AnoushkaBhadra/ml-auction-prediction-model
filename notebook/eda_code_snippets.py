# EDA CODE SNIPPETS FOR WINNER PREDICTION
# Copy these code blocks into your notebook cells

# ============================================================================
# STEP 1: BASIC DATA OVERVIEW
# ============================================================================

print("=== DATA SHAPE ===")
print(f"Shape: {df.shape}")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n=== COLUMN NAMES ===")
print(df.columns.tolist())

print("\n=== DATA TYPES ===")
print(df.dtypes)

print("\n=== DATA INFO ===")
df.info()

# ============================================================================
# STEP 2: TARGET VARIABLE ANALYSIS
# ============================================================================

print("=== TARGET VARIABLE: H1 BUYER NAME ===")
print(f"Unique H1 buyers: {df['h1_buyer_name'].nunique()}")

# Win counts per buyer
buyer_counts = df['h1_buyer_name'].value_counts()
print(f"\nTop 10 buyers by win count:")
print(buyer_counts.head(10))

print(f"\nBottom 10 buyers by win count:")
print(buyer_counts.tail(10))

# Basic statistics about buyer distribution
print(f"\nBuyer distribution statistics:")
print(f"Mean wins per buyer: {buyer_counts.mean():.2f}")
print(f"Median wins per buyer: {buyer_counts.median():.2f}")
print(f"Std wins per buyer: {buyer_counts.std():.2f}")
print(f"Min wins: {buyer_counts.min()}")
print(f"Max wins: {buyer_counts.max()}")

# ============================================================================
# STEP 3: CONTEXT FEATURE ANALYSIS
# ============================================================================

print("=== CONTEXT FEATURES ===")

# State analysis
print(f"Unique states: {df['state'].nunique()}")
print(f"\nAuctions by state:")
state_counts = df['state'].value_counts()
print(state_counts)

# Month analysis (convert auction_date to datetime first)
df['auction_date'] = pd.to_datetime(df['auction_date'])
df['month'] = df['auction_date'].dt.month
df['year'] = df['auction_date'].dt.year

print(f"\nAuctions by month:")
month_counts = df['month'].value_counts().sort_index()
print(month_counts)

print(f"\nAuctions by year:")
year_counts = df['year'].value_counts().sort_index()
print(year_counts)

# Product analysis
print(f"\nUnique products: {df['productdescription'].nunique()}")
print(f"\nAuctions by product:")
product_counts = df['productdescription'].value_counts()
print(product_counts)

# ============================================================================
# STEP 4: NUMERICAL FEATURE ANALYSIS
# ============================================================================

print("=== NUMERICAL FEATURES ===")

# Select numerical columns
num_cols = ['proposed_rp', 'quantity', 'last_bid_price', 'total_amt']
print("Numerical columns summary:")
print(df[num_cols].describe())

# Check for outliers using IQR method
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"\n{col} - Outliers: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")

# ============================================================================
# STEP 5: BUYER-CONTEXT RELATIONSHIPS
# ============================================================================

print("=== BUYER-CONTEXT RELATIONSHIPS ===")

# Buyers vs. state
print("Top buyers by state (showing first 5 states):")
buyer_state = pd.crosstab(df['h1_buyer_name'], df['state'])
for state in buyer_state.columns[:5]:
    print(f"\n{state}:")
    print(buyer_state[state].sort_values(ascending=False).head(3))

# Buyers vs. category
print("\nTop buyers by product category:")
buyer_cat = pd.crosstab(df['h1_buyer_name'], df['productdescription'])
for cat in buyer_cat.columns:
    print(f"\n{cat}:")
    print(buyer_cat[cat].sort_values(ascending=False).head(3))

# Buyers vs. month
print("\nTop buyers by month:")
buyer_month = pd.crosstab(df['h1_buyer_name'], df['month'])
for month in buyer_month.columns:
    print(f"\nMonth {month}:")
    print(buyer_month[month].sort_values(ascending=False).head(3))

# ============================================================================
# STEP 6: TRENDS AND PATTERNS
# ============================================================================

print("=== TRENDS AND PATTERNS ===")

# Auctions over time
print("Number of auctions by month:")
monthly_auctions = df.groupby(df['auction_date'].dt.to_period('M')).size()
print(monthly_auctions)

# Average price over time
print("\nAverage last bid price by month:")
monthly_avg_price = df.groupby(df['auction_date'].dt.to_period('M'))['last_bid_price'].mean()
print(monthly_avg_price)

# Price trends by category
print("\nAverage price by product category:")
cat_avg_price = df.groupby('productdescription')['last_bid_price'].agg(['mean', 'std', 'count'])
print(cat_avg_price)

# ============================================================================
# STEP 7: SUMMARY STATISTICS
# ============================================================================

print("=== SUMMARY STATISTICS ===")
print(df.describe(include='all'))

print("\n=== KEY INSIGHTS ===")
print(f"1. Total auctions: {len(df)}")
print(f"2. Unique buyers: {df['h1_buyer_name'].nunique()}")
print(f"3. Unique states: {df['state'].nunique()}")
print(f"4. Unique products: {df['productdescription'].nunique()}")
print(f"5. Date range: {df['auction_date'].min()} to {df['auction_date'].max()}")
print(f"6. Most frequent buyer: {buyer_counts.index[0]} ({buyer_counts.iloc[0]} wins)")
print(f"7. Most frequent state: {state_counts.index[0]} ({state_counts.iloc[0]} auctions)")
print(f"8. Most frequent product: {product_counts.index[0]} ({product_counts.iloc[0]} auctions)")

# ============================================================================
# STEP 8: VISUALIZATIONS (OPTIONAL)
# ============================================================================

# Set up the plotting style
plt.style.use('default')
sns.set_palette("husl")

# 1. Buyer win distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
buyer_counts.head(20).plot(kind='bar')
plt.title('Top 20 Buyers by Win Count')
plt.xlabel('Buyer')
plt.ylabel('Number of Wins')
plt.xticks(rotation=45, ha='right')

plt.subplot(1, 2, 2)
sns.histplot(buyer_counts, bins=50, log=True)
plt.title('Distribution of Win Counts (Log Scale)')
plt.xlabel('Number of Wins')
plt.ylabel('Number of Buyers')
plt.tight_layout()
plt.show()

# 2. State and product distributions
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
state_counts.plot(kind='bar')
plt.title('Auctions by State')
plt.xlabel('State')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')

plt.subplot(1, 3, 2)
month_counts.plot(kind='bar')
plt.title('Auctions by Month')
plt.xlabel('Month')
plt.ylabel('Count')

plt.subplot(1, 3, 3)
product_counts.plot(kind='bar')
plt.title('Auctions by Product')
plt.xlabel('Product')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 3. Price distributions
plt.figure(figsize=(15, 5))
for i, col in enumerate(['proposed_rp', 'last_bid_price', 'total_amt']):
    plt.subplot(1, 3, i+1)
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
plt.tight_layout()
plt.show() 