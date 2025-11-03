# Step 1: Generate Realistic Ghana Mobile Money Dataset
# This creates data reflecting actual market patterns: MTN dominance and urban-rural divide

import pandas as pd
import numpy as np
import random
from faker import Faker

print("Starting realistic synthetic data generation for Ghana FinTech credit scoring...")

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
fake = Faker()

# Ghana configuration with realistic market shares
GHANA_REGIONS = [
    'Greater Accra', 'Ashanti', 'Western', 'Central',  # Urban regions
    'Eastern', 'Volta', 'Bono', 'Ahafo',               # Semi-urban regions  
    'Northern', 'Upper East', 'Upper West',            # Rural regions
    'Savannah', 'North East', 'Oti', 'Bono East', 'Western North'  # Rural regions
]

# Define area types for realistic urban-rural split
URBAN_REGIONS = ['Greater Accra', 'Ashanti', 'Western', 'Central']
RURAL_REGIONS = ['Northern', 'Upper East', 'Upper West', 'Savannah', 'North East', 'Oti']

# Mobile providers with realistic market shares (MTN ~70%)
MOBILE_PROVIDERS = ['MTN Mobile Money', 'Telecel Cash', 'AirtelTigo Money']
PROVIDER_PROBABILITIES = [0.72, 0.18, 0.10]  # MTN dominance

ECONOMIC_SECTORS = [
    'Small-scale Trade', 'Agriculture', 'Transportation', 
    'Personal Services', 'Artisan Work', 'Food Services'
]

# Regional economic indicators based on actual Ghana data
REGIONAL_ECONOMIC_INDEX = {
    'Greater Accra': 0.85, 'Ashanti': 0.78, 'Western': 0.72, 'Central': 0.68,
    'Eastern': 0.65, 'Volta': 0.60, 'Bono': 0.63, 'Ahafo': 0.62,
    'Northern': 0.52, 'Upper East': 0.48, 'Upper West': 0.45, 
    'Savannah': 0.47, 'North East': 0.46, 'Oti': 0.49, 
    'Bono East': 0.61, 'Western North': 0.58
}

# Generate 10,000 records
n_samples = 10000
data = []

print(f"Generating {n_samples} realistic records...")
print("Reflecting MTN market dominance and urban-rural patterns...")

for i in range(n_samples):
    # Assign region with urban bias (70% urban, 30% rural)
    if random.random() < 0.7:
        region = random.choice(URBAN_REGIONS)
        area_type = 'Urban'
    else:
        region = random.choice(RURAL_REGIONS)
        area_type = 'Rural'
    
    economic_index = REGIONAL_ECONOMIC_INDEX[region]
    
    # Generate demographic data
    age = random.randint(18, 65)
    gender = random.choice(['Male', 'Female'])
    sector = random.choice(ECONOMIC_SECTORS)
    
    # Assign mobile provider with MTN dominance
    mobile_provider = np.random.choice(MOBILE_PROVIDERS, p=PROVIDER_PROBABILITIES)
    
    # Generate mobile money behavior with urban-rural differences
    months_active = random.randint(6, 48)
    
    # Urban areas have more transactions and higher values
    if area_type == 'Urban':
        avg_monthly_transactions = max(10, int(np.random.normal(30, 10)))
        avg_transaction_value = max(50, int(np.random.normal(200, 80)))
        transaction_consistency = max(0.3, min(0.95, np.random.normal(0.75, 0.15)))
    else:  # Rural areas
        avg_monthly_transactions = max(5, int(np.random.normal(15, 8)))
        avg_transaction_value = max(20, int(np.random.normal(80, 40)))
        transaction_consistency = max(0.2, min(0.90, np.random.normal(0.65, 0.20)))
    
    # Generate financial behavior features  
    bill_payment_punctuality = max(0.1, min(0.99, np.random.normal(0.70, 0.20)))
    savings_accumulation_rate = max(0.01, min(0.3, np.random.normal(0.08, 0.04)))
    emergency_fund_coverage = random.randint(1, 4)
    income_stability = max(0.1, min(0.99, np.random.normal(0.65, 0.20) * economic_index))
    digital_services_used = random.randint(1, 5)
    
    # Calculate default probability - urban areas have slightly higher defaults
    base_default_probability = (
        0.3 * (1 - transaction_consistency) +
        0.25 * (1 - bill_payment_punctuality) +
        0.2 * (1 - savings_accumulation_rate * 10) +
        0.15 * (1 - income_stability) +
        0.1 * (1 - emergency_fund_coverage / 4)
    )
    
    # Urban areas have slightly higher risk due to more credit usage
    if area_type == 'Urban':
        base_default_probability *= 1.1
    
    # Add randomness and ensure valid probability
    default_probability = max(0, min(1, base_default_probability + np.random.normal(0, 0.1)))
    will_default = 1 if default_probability > 0.5 else 0
    
    record = {
        'customer_id': f'GH_{i:06d}',
        'region': region,
        'area_type': area_type,
        'economic_index': round(economic_index, 3),
        'age': age,
        'gender': gender,
        'sector': sector,
        'mobile_provider': mobile_provider,
        'months_active': months_active,
        'avg_monthly_transactions': avg_monthly_transactions,
        'transaction_consistency': round(transaction_consistency, 3),
        'bill_payment_punctuality': round(bill_payment_punctuality, 3),
        'savings_accumulation_rate': round(savings_accumulation_rate, 3),
        'emergency_fund_coverage': emergency_fund_coverage,
        'income_stability': round(income_stability, 3),
        'avg_transaction_value': round(avg_transaction_value, 2),
        'digital_services_used': digital_services_used,
        'will_default': will_default,
        'default_probability': round(default_probability, 3)
    }
    
    data.append(record)

# Create DataFrame
df = pd.DataFrame(data)

print("Realistic dataset created successfully!")
print(f"Dataset shape: {df.shape}")

# Verify realistic patterns
print("\n" + "="*50)
print("REALITY CHECK - KEY MARKET PATTERNS")
print("="*50)

# MTN Market Share
mtn_share = (df['mobile_provider'] == 'MTN Mobile Money').mean()
print(f"MTN Market Share: {mtn_share:.1%}")

# Urban-Rural Split
urban_share = (df['area_type'] == 'Urban').mean()
print(f"Urban Customer Share: {urban_share:.1%}")

# Default Rates by Area
urban_default = df[df['area_type'] == 'Urban']['will_default'].mean()
rural_default = df[df['area_type'] == 'Rural']['will_default'].mean()
print(f"Urban Default Rate: {urban_default:.3f}")
print(f"Rural Default Rate: {rural_default:.3f}")

# Transaction Values by Area
urban_trans = df[df['area_type'] == 'Urban']['avg_transaction_value'].mean()
rural_trans = df[df['area_type'] == 'Rural']['avg_transaction_value'].mean()
print(f"Urban Avg Transaction: GHS {urban_trans:.2f}")
print(f"Rural Avg Transaction: GHS {rural_trans:.2f}")

# Display first few records
print("\nFirst 3 records:")
print(df.head(3))

# Save to CSV file
csv_filename = 'ghana_fintech_credit_data.csv'
df.to_csv(csv_filename, index=False)
print(f"\nDataset saved as '{csv_filename}'")

print("\n" + "="*50)
print("STEP 1 COMPLETED")
print("="*50)
print("Dataset reflects real Ghana mobile money patterns:")
print("- MTN has dominant market share (~72%)")
print("- More customers in urban areas (~70%)") 
print("- Higher transaction values in urban areas")
print("- Realistic regional economic variations")