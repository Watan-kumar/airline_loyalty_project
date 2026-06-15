import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, roc_auc_score,
                              roc_curve, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings; warnings.filterwarnings('ignore')

df = pd.read_csv('/Users/watankumar/Downloads/airline_loyalty_project/cleaned_features.csv')

# CHURN PREDICTION MODEL


FEATURES = [
    'tenure_months', 'card_tier_num', 'education_num', 'Salary',
    'total_flights', 'flights_2017',           
    'total_distance', 'total_points_acc', 'total_points_red',
    'redemption_ratio', 'flight_consistency',
    'avg_flights_per_month', 'flight_monthly_std',
    'value_per_flight', 'CLV', 'engagement_score',
    
]

TARGET = 'churned'

# Encode categoricals
df['gender_enc'] = (df['Gender'] == 'Male').astype(int)
df['married_enc'] = (df['Marital Status'] == 'Married').astype(int)
df['promo_enc']  = (df['Enrollment Type'] == '2018 Promotion').astype(int)
FEATURES += ['gender_enc', 'married_enc', 'promo_enc']

X = df[FEATURES].copy()
y = df[TARGET]

# Impute any remaining NaN
X = X.fillna(X.median())

#  Cross-validation comparison 
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'Random Forest':       RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                                   random_state=42, n_jobs=-1),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=150, max_depth=4,
                                                       learning_rate=0.08, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}
print("Cross-validation AUC scores:")
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    results[name] = scores
    print(f"  {name}: {scores.mean():.4f} ± {scores.std():.4f}")

#  Train best model (Random Forest) on full data for feature importance 
rf = RandomForestClassifier(n_estimators=300, class_weight='balanced',
                             random_state=42, n_jobs=-1)
rf.fit(X, y)
df['churn_probability'] = rf.predict_proba(X)[:, 1]

#  Feature importance 
fi = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nTop 10 churn predictors:")
print(fi.head(10).to_string())


# CUSTOMER SEGMENTATION (K-Means on behavioral features)

SEG_FEATURES = ['CLV', 'total_flights', 'flight_consistency',
                'redemption_ratio', 'tenure_months', 'churn_probability',
                'activity_trend', 'card_tier_num']

Xs = df[SEG_FEATURES].fillna(0)
scaler = StandardScaler()
Xs_scaled = scaler.fit_transform(Xs)

# Elbow test
inertias = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(Xs_scaled)
    inertias.append(km.inertia_)

# Fit with k=5 (chosen after elbow — 5 meaningful airline segments)
km = KMeans(n_clusters=5, random_state=42, n_init=20)
df['segment'] = km.fit_predict(Xs_scaled)

# Label segments by CLV + flight behavior
seg_profile = df.groupby('segment').agg(
    n            = ('Loyalty Number', 'count'),
    avg_clv      = ('CLV', 'mean'),
    avg_flights  = ('total_flights', 'mean'),
    avg_tenure   = ('tenure_months', 'mean'),
    churn_rate   = ('churned', 'mean'),
    churn_prob   = ('churn_probability', 'mean'),
    avg_consistency = ('flight_consistency', 'mean'),
    avg_redemption  = ('redemption_ratio', 'mean'),
    pct_aurora   = ('card_tier_num', lambda x: (x == 3).mean()),
).sort_values('avg_clv', ascending=False).reset_index()

print("\nSegment profiles:")
print(seg_profile.to_string())

# Map segments to meaningful names
seg_order = seg_profile['segment'].tolist()
seg_names = {
    seg_order[0]: 'Champions',        # highest CLV
    seg_order[1]: 'Loyal Actives',    # high CLV, consistent
    seg_order[2]: 'At-Risk Valuables',# mid CLV, higher churn
    seg_order[3]: 'Occasional Flyers',# low flights, low CLV
    seg_order[4]: 'Dormant / Lost',   # lowest engagement
}
df['segment_name'] = df['segment'].map(seg_names)

print("\nSegment name mapping:")
for k,v in seg_names.items():
    print(f"  Cluster {k} → {v}")


# VISUALISATIONS — Figure 1: Churn Model

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Churn Prediction Model — Airline Loyalty Program', fontsize=14, fontweight='bold')

# 1a. CV AUC comparison
names = list(results.keys())
means = [results[n].mean() for n in names]
stds  = [results[n].std()  for n in names]
colors = ['#4e8cff','#ff6b6b','#51cf66']
axes[0].barh(names, means, xerr=stds, color=colors, capsize=4, height=0.5)
axes[0].set_xlabel('ROC-AUC (5-fold CV)')
axes[0].set_title('Model Comparison')
axes[0].set_xlim(0.5, 1.0)
for i, (m, s) in enumerate(zip(means, stds)):
    axes[0].text(m + 0.005, i, f'{m:.3f}', va='center', fontsize=10)

# 1b. Feature importance (top 12)
fi12 = fi.head(12)
axes[1].barh(fi12.index[::-1], fi12.values[::-1], color='#4e8cff')
axes[1].set_xlabel('Importance')
axes[1].set_title('Top 12 Churn Predictors')
axes[1].tick_params(axis='y', labelsize=8)

# 1c. Churn probability distribution by actual churn
axes[2].hist(df[df['churned']==0]['churn_probability'], bins=40,
             alpha=0.6, label='Active', color='#51cf66', density=True)
axes[2].hist(df[df['churned']==1]['churn_probability'], bins=40,
             alpha=0.6, label='Churned', color='#ff6b6b', density=True)
axes[2].set_xlabel('Predicted Churn Probability')
axes[2].set_ylabel('Density')
axes[2].set_title('Churn Score Distribution')
axes[2].legend()

plt.tight_layout()
plt.savefig('/Users/watankumar/Downloads/airline_loyalty_project/fig1_churn_model.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig1_churn_model.png")

# VISUALISATIONS — Figure 2: Segmentation

seg_named = df.groupby('segment_name').agg(
    n            = ('Loyalty Number', 'count'),
    avg_clv      = ('CLV', 'mean'),
    avg_flights  = ('total_flights', 'mean'),
    churn_rate   = ('churned', 'mean'),
    churn_prob   = ('churn_probability', 'mean'),
    avg_tenure   = ('tenure_months', 'mean'),
    avg_consistency = ('flight_consistency', 'mean'),
).reset_index()

seg_order_named = ['Champions','Loyal Actives','At-Risk Valuables','Occasional Flyers','Dormant / Lost']
seg_named['segment_name'] = pd.Categorical(seg_named['segment_name'], categories=seg_order_named, ordered=True)
seg_named = seg_named.sort_values('segment_name')

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle('Customer Segmentation — 5 Behavioral Segments', fontsize=14, fontweight='bold')
palette = ['#2ecc71','#3498db','#e67e22','#9b59b6','#e74c3c']

# 2a. Segment size
axes2[0,0].bar(seg_named['segment_name'], seg_named['n'], color=palette)
axes2[0,0].set_title('Segment Size (# Members)')
axes2[0,0].tick_params(axis='x', rotation=20)
for i, v in enumerate(seg_named['n']):
    axes2[0,0].text(i, v + 50, f'{v:,}', ha='center', fontsize=9)

# 2b. Average CLV
axes2[0,1].bar(seg_named['segment_name'], seg_named['avg_clv'], color=palette)
axes2[0,1].set_title('Average CLV (CAD)')
axes2[0,1].tick_params(axis='x', rotation=20)

# 2c. Churn rate
axes2[1,0].bar(seg_named['segment_name'], seg_named['churn_rate']*100, color=palette)
axes2[1,0].set_title('Churn Rate (%)')
axes2[1,0].set_ylabel('%')
axes2[1,0].tick_params(axis='x', rotation=20)
for i, v in enumerate(seg_named['churn_rate']):
    axes2[1,0].text(i, v*100 + 0.3, f'{v*100:.1f}%', ha='center', fontsize=9)

# 2d. CLV vs Churn Probability scatter
sc_colors = [palette[seg_order_named.index(s)] for s in df['segment_name'].fillna('Dormant / Lost')]
axes2[1,1].scatter(df['CLV'].clip(upper=30000), df['churn_probability'],
                   c=sc_colors, alpha=0.15, s=5)
axes2[1,1].set_xlabel('CLV (CAD, capped 30k)')
axes2[1,1].set_ylabel('Churn Probability')
axes2[1,1].set_title('CLV vs Churn Risk by Segment')
from matplotlib.patches import Patch
legend_els = [Patch(color=palette[i], label=s) for i, s in enumerate(seg_order_named)]
axes2[1,1].legend(handles=legend_els, fontsize=8, loc='upper right')

plt.tight_layout()
plt.savefig('/Users/watankumar/Downloads/airline_loyalty_project/fig2_segmentation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig2_segmentation.png")

# Save enriched dataset
df.to_csv('/Users/watankumar/Downloads/airline_loyalty_project/final_dataset.csv', index=False)
print("Saved final_dataset.csv")

# Save segment summary
seg_named.to_csv('/Users/watankumar/Downloads/airline_loyalty_project/segment_summary.csv', index=False)
print("Saved segment_summary.csv")
