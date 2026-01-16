# Step 3: Feature Engineering and Preprocessing
# Prepare the data for machine learning models with clear, interpretable features

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("Starting feature engineering and preprocessing...")

# Load the dataset
df = pd.read_csv("D:\ppp\ghana_fintech_credit\data\ghana_fintech_credit_data.csv")

print("Dataset loaded successfully")
print(f"Original dataset shape: {df.shape}")

# Display current features
print("\nOriginal features:")
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")

# Create clear, business-friendly feature names for stakeholders
feature_rename_map = {
    'transaction_consistency': 'transaction_regularity',
    'bill_payment_punctuality': 'bill_payment_history',
    'savings_accumulation_rate': 'savings_habit',
    'income_stability': 'income_reliability',
    'emergency_fund_coverage': 'emergency_savings',
    'avg_monthly_transactions': 'monthly_activity',
    'avg_transaction_value': 'typical_transaction_size',
    'digital_services_used': 'digital_engagement'
}

df = df.rename(columns=feature_rename_map)

print("\nCreating new features that make business sense...")

# 1. Create Financial Health Score (combining multiple positive behaviors)
df['financial_health_score'] = (
    df['transaction_regularity'] * 0.25 +
    df['bill_payment_history'] * 0.25 +
    (df['savings_habit'] * 100) * 0.20 +  # Scale savings rate to be more meaningful
    df['income_reliability'] * 0.20 +
    (df['emergency_savings'] / 4) * 0.10   # Normalize emergency savings
)

# 2. Create Digital Engagement Score
df['digital_engagement_score'] = (
    (df['digital_engagement'] / 5) * 0.4 +        # Services used (0-1 scale)
    (df['months_active'] / 48) * 0.3 +           # Tenure (0-1 scale)
    (df['monthly_activity'] / 50) * 0.3          # Activity level (0-1 scale)
)

# 3. Create Risk Flags (easy for stakeholders to understand)
df['low_savings_flag'] = (df['savings_habit'] < 0.05).astype(int)
df['irregular_income_flag'] = (df['income_reliability'] < 0.6).astype(int)
df['poor_bill_payment_flag'] = (df['bill_payment_history'] < 0.7).astype(int)

# 4. Create Customer Value Score
df['customer_value_score'] = (
    (df['typical_transaction_size'] / 300) * 0.4 +    # Transaction size importance
    (df['monthly_activity'] / 40) * 0.3 +             # Activity importance
    (df['months_active'] / 48) * 0.3                  # Loyalty importance
)

# 5. Create Urban Advantage Score (captures urban benefits)
df['urban_advantage'] = (
    (df['economic_index'] - 0.5) * 2 +               # Economic advantage
    (df['typical_transaction_size'] / 200) * 0.5     # Spending power
)

print("New features created:")
new_features = ['financial_health_score', 'digital_engagement_score', 
                'low_savings_flag', 'irregular_income_flag', 'poor_bill_payment_flag',
                'customer_value_score', 'urban_advantage']

for feature in new_features:
    print(f"  - {feature}")

# Prepare features for modeling
print("\nPreparing features for machine learning...")

# Separate target variable
target = 'will_default'
y = df[target]

# Select features for the model (focus on interpretable features)
base_features = [
    'economic_index', 'age', 'months_active', 'monthly_activity',
    'transaction_regularity', 'bill_payment_history', 'savings_habit',
    'emergency_savings', 'income_reliability', 'typical_transaction_size',
    'digital_engagement'
]

engineered_features = [
    'financial_health_score', 'digital_engagement_score', 
    'low_savings_flag', 'irregular_income_flag', 'poor_bill_payment_flag',
    'customer_value_score', 'urban_advantage'
]

categorical_features = ['area_type', 'gender', 'sector', 'mobile_provider']

# Combine all features
all_features = base_features + engineered_features + categorical_features

print(f"Total features considered: {len(all_features)}")
print(f"Base features: {len(base_features)}")
print(f"Engineered features: {len(engineered_features)}")
print(f"Categorical features: {len(categorical_features)}")

# Create feature set
X = df[all_features]

# Handle categorical variables (one-hot encoding)
print("Encoding categorical variables...")
X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)

print(f"After encoding: {X_encoded.shape[1]} features")

# Split the data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, 
    y, 
    test_size=0.3, 
    random_state=42, 
    stratify=y  # Maintain same default rate in both sets
)

print(f"Training set: {X_train.shape[0]:,} records")
print(f"Testing set: {X_test.shape[0]:,} records")
print(f"Training default rate: {y_train.mean():.3f}")
print(f"Testing default rate: {y_test.mean():.3f}")

# Scale the numerical features for better model performance
print("Scaling numerical features...")
scaler = StandardScaler()

# Fit scaler on training data and transform both sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrames for better handling
feature_columns = X_encoded.columns.tolist()
X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_columns, index=X_test.index)

print("Feature scaling completed")

# Create a feature importance preview based on correlation with target
print("\nFeature correlation with loan default:")
correlations = X_encoded.copy()
correlations['will_default'] = y
feature_corrs = correlations.corr()['will_default'].abs().sort_values(ascending=False)

# Show top 15 features correlated with default
top_features = feature_corrs[1:16]  # Exclude target itself
print("Top 15 features most related to default risk:")
for feature, corr in top_features.items():
    print(f"  {feature:30}: {corr:.3f}")

# Save processed datasets and important objects
print("\nSaving processed data...")
X_train_scaled.to_csv('X_train_processed.csv', index=False)
X_test_scaled.to_csv('X_test_processed.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

# Save feature names and scaler for future use
import joblib

joblib.dump(scaler, 'feature_scaler.pkl')
joblib.dump(feature_columns, 'feature_columns.pkl')
joblib.dump(top_features.index.tolist(), 'important_features.pkl')

print("Data processing artifacts saved")

# Display final dataset summary
print("\n" + "="*60)
print("FEATURE ENGINEERING COMPLETED")
print("="*60)
print(f"Original dataset: {len(df.columns)} features")
print(f"After engineering: {X_encoded.shape[1]} features")
print(f"Training set: {X_train_scaled.shape[0]:,} records")
print(f"Testing set: {X_test_scaled.shape[0]:,} records")

print("\nKey engineered features for stakeholders:")
stakeholder_features = {
    'financial_health_score': 'Overall financial responsibility score (0-100 scale)',
    'digital_engagement_score': 'How actively they use digital services (0-1 scale)',
    'customer_value_score': 'Potential value as a long-term customer (0-1 scale)',
    'urban_advantage': 'Benefits from being in urban areas (economic + spending)',
    'low_savings_flag': 'Red flag for customers with very low savings',
    'irregular_income_flag': 'Red flag for unstable income patterns',
    'poor_bill_payment_flag': 'Red flag for late bill payments'
}

for feature, description in stakeholder_features.items():
    print(f"  • {feature:25} - {description}")

print("\nMost important predictors of default risk:")
print("  1. Transaction regularity")
print("  2. Bill payment history") 
print("  3. Income reliability")
print("  4. Financial health score")
print("  5. Savings habits")

print("\nReady for model training in Step 4")