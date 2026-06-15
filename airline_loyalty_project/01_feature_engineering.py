import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── Load data ──────────────────────────────────────────────────────────────
loyalty  = pd.read_csv(r'/Users/watankumar/Downloads/airline_loyalty_project/Customer Loyalty History.csv')
flight   = pd.read_csv(r'/Users/watankumar/Downloads/airline_loyalty_project/Customer Flight Activity.csv')
calendar = pd.read_csv(r'/Users/watankumar/Downloads/airline_loyalty_project/Calendar.csv')

print(f"Loyalty rows: {len(loyalty)}  |  Flight rows: {len(flight)}")

# ══════════════════════════════════════════════════════════════════════════
# PART 1 — DATA CLEANING
# ══════════════════════════════════════════════════════════════════════════

# --- Fix negative salaries (data error → treat as NaN) ---
loyalty.loc[loyalty['Salary'] < 0, 'Salary'] = np.nan

# --- Impute missing salary with median by Education + Loyalty Card tier ---
salary_median = loyalty.groupby(['Education','Loyalty Card'])['Salary'].transform('median')
loyalty['Salary'] = loyalty['Salary'].fillna(salary_median)
loyalty['Salary'] = loyalty['Salary'].fillna(loyalty['Salary'].median())  # fallback

# --- Create churn flag ---
# Churn Definition 1 (formal): member has a recorded cancellation
loyalty['churn_formal'] = loyalty['Cancellation Year'].notna().astype(int)

# Churn Definition 2 (behavioral): member enrolled before 2018 AND has zero
# total flights in the last 12 months of available data (2018 Jan-Dec).
# This captures silent churners who never officially cancelled.
last_12 = flight[flight['Year'] == 2018].groupby('Loyalty Number')['Total Flights'].sum().reset_index()
last_12.columns = ['Loyalty Number', 'flights_2018']

loyalty = loyalty.merge(last_12, on='Loyalty Number', how='left')
loyalty['flights_2018'] = loyalty['flights_2018'].fillna(0)

# Silent churn: enrolled before 2018, no flights in 2018, NOT formally cancelled
loyalty['churn_silent'] = (
    (loyalty['Enrollment Year'] < 2018) &
    (loyalty['flights_2018'] == 0) &
    (loyalty['churn_formal'] == 0)
).astype(int)

# Combined churn label (union of formal + silent)
loyalty['churned'] = ((loyalty['churn_formal'] == 1) | (loyalty['churn_silent'] == 1)).astype(int)

print(f"\nChurn breakdown:")
print(f"  Formal cancellations : {loyalty['churn_formal'].sum():,}")
print(f"  Silent churners      : {loyalty['churn_silent'].sum():,}")
print(f"  Total churned        : {loyalty['churned'].sum():,} ({loyalty['churned'].mean()*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════
# PART 2 — FEATURE ENGINEERING (flight-level aggregations)
# ══════════════════════════════════════════════════════════════════════════

# Aggregate flight activity per customer
agg = flight.groupby('Loyalty Number').agg(
    total_flights          = ('Total Flights', 'sum'),
    total_distance         = ('Distance', 'sum'),
    total_points_acc       = ('Points Accumulated', 'sum'),
    total_points_red       = ('Points Redeemed', 'sum'),
    total_dollar_redeemed  = ('Dollar Cost Points Redeemed', 'sum'),
    active_months          = ('Month', 'nunique'),       # months with any record
    avg_flights_per_month  = ('Total Flights', 'mean'),
    avg_distance_per_month = ('Distance', 'mean'),
    max_flights_month      = ('Total Flights', 'max'),
    flight_months_nonzero  = ('Total Flights', lambda x: (x > 0).sum()),
).reset_index()

# -- Behavioral features --

# Redemption ratio: how actively does customer spend points vs accumulate?
agg['redemption_ratio'] = np.where(
    agg['total_points_acc'] > 0,
    agg['total_points_red'] / agg['total_points_acc'],
    0
)

# Flight consistency: fraction of recorded months where customer actually flew
agg['flight_consistency'] = np.where(
    agg['active_months'] > 0,
    agg['flight_months_nonzero'] / agg['active_months'],
    0
)

# Points burn rate (dollar value redeemed per flight)
agg['value_per_flight'] = np.where(
    agg['total_flights'] > 0,
    agg['total_dollar_redeemed'] / agg['total_flights'],
    0
)

# Activity recency: flights in 2017 vs 2018 (trend signal)
y17 = flight[flight['Year']==2017].groupby('Loyalty Number')['Total Flights'].sum().rename('flights_2017')
y18 = flight[flight['Year']==2018].groupby('Loyalty Number')['Total Flights'].sum().rename('flights_2018_agg')
agg = agg.merge(y17, on='Loyalty Number', how='left').merge(y18, on='Loyalty Number', how='left')
agg['flights_2017'] = agg['flights_2017'].fillna(0)
agg['flights_2018_agg'] = agg['flights_2018_agg'].fillna(0)

# Trend: is the customer flying MORE or LESS in 2018 vs 2017?
agg['activity_trend'] = agg['flights_2018_agg'] - agg['flights_2017']  # positive = growing

# Seasonal spread: std dev of monthly flights across calendar months (higher = more variable)
monthly_std = flight.groupby('Loyalty Number')['Total Flights'].std().rename('flight_monthly_std').fillna(0)
agg = agg.merge(monthly_std, on='Loyalty Number', how='left')

# -- Merge with loyalty demographics --
df = loyalty.merge(agg, on='Loyalty Number', how='left')

# Fill customers with zero flight history
flight_cols = ['total_flights','total_distance','total_points_acc','total_points_red',
               'total_dollar_redeemed','active_months','avg_flights_per_month',
               'avg_distance_per_month','max_flights_month','flight_months_nonzero',
               'redemption_ratio','flight_consistency','value_per_flight',
               'flights_2017','flights_2018_agg','activity_trend','flight_monthly_std']
df[flight_cols] = df[flight_cols].fillna(0)

# -- Tenure feature (months since enrollment, capped at data end Dec 2018) --
df['enrollment_date_num'] = df['Enrollment Year'] * 12 + df['Enrollment Month']
end_date = 2018 * 12 + 12
df['tenure_months'] = end_date - df['enrollment_date_num']

# -- Card tier as ordinal --
card_map = {'Star': 1, 'Nova': 2, 'Aurora': 3}
df['card_tier_num'] = df['Loyalty Card'].map(card_map)

# -- Education ordinal --
edu_map = {'High School or Below': 1, 'College': 2, 'Bachelor': 3, 'Master': 4, 'Doctor': 5}
df['education_num'] = df['Education'].map(edu_map).fillna(2)

# -- Salary band --
df['salary_band'] = pd.qcut(df['Salary'], q=4, labels=['Low','Mid-Low','Mid-High','High'])

# -- CLV tier (for segmentation) --
df['clv_tier'] = pd.qcut(df['CLV'], q=4, labels=['Bronze','Silver','Gold','Platinum'])

# -- Engagement score (composite, 0-100) --
# Normalise 4 signals and average them
def norm(s):
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn + 1e-9)

df['engagement_score'] = (
    norm(df['total_flights']) * 30 +
    norm(df['flight_consistency']) * 25 +
    norm(df['redemption_ratio']) * 20 +
    norm(df['activity_trend'].clip(lower=-10)) * 25
)

print(f"\nFinal dataset shape: {df.shape}")
print(f"\nKey engineered features:")
for f in ['tenure_months','total_flights','flight_consistency','redemption_ratio',
          'activity_trend','engagement_score','churn_formal','churn_silent','churned']:
    print(f"  {f}: mean={df[f].mean():.2f}  std={df[f].std():.2f}")

df.to_csv('/Users/watankumar/Downloads/airline_loyalty_project/cleaned_features.csv', index=False)
print("\nSaved → cleaned_features.csv")
