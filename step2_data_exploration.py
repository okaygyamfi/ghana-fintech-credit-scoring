# Step 2: Simplified Data Exploration for Non-Technical Audience
# Focus on clear, easy-to-understand visualizations of key market patterns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Starting simplified data exploration...")

# Load the dataset
df = pd.read_csv("D:\ppp\ghana_fintech_credit\data\ghana_fintech_credit_data.csv")

print("Dataset loaded successfully")
print(f"Total customers: {len(df):,}")

# Set up clean, professional styling
plt.style.use('default')
sns.set_palette("colorblind")

# Create a summary report
print("\n" + "="*50)
print("GHANA MOBILE MONEY MARKET SUMMARY")
print("="*50)

# Key statistics for stakeholders
total_customers = len(df)
mtn_customers = len(df[df['mobile_provider'] == 'MTN Mobile Money'])
urban_customers = len(df[df['area_type'] == 'Urban'])
overall_default_rate = df['will_default'].mean()

print(f"Total Customers: {total_customers:,}")
print(f"MTN Market Share: {mtn_customers/total_customers:.1%}")
print(f"Urban Customers: {urban_customers/total_customers:.1%}")
print(f"Overall Default Rate: {overall_default_rate:.1%}")

# VISUALIZATION 1: Mobile Money Market Share
print("\nCreating Visualization 1: Mobile Money Market Share...")

plt.figure(figsize=(10, 6))
provider_counts = df['mobile_provider'].value_counts()

# Simple pie chart
plt.pie(provider_counts.values, 
        labels=provider_counts.index, 
        autopct='%1.0f%%',
        startangle=90,
        colors=['yellow', 'red', 'blue'])

plt.title('Mobile Money Market Share in Ghana\nMTN Dominates with 72% of Customers', 
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('market_share.png', dpi=300, bbox_inches='tight')
plt.show()

# VISUALIZATION 2: Urban vs Rural Comparison
print("Creating Visualization 2: Urban vs Rural Comparison...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Customer Distribution
urban_rural_counts = df['area_type'].value_counts()
axes[0].pie(urban_rural_counts.values, 
           labels=urban_rural_counts.index, 
           autopct='%1.0f%%',
           colors=['lightblue', 'lightgreen'])
axes[0].set_title('Customer Distribution:\nUrban vs Rural')

# Default Rates Comparison
default_rates = df.groupby('area_type')['will_default'].mean()
axes[1].bar(default_rates.index, default_rates.values, 
           color=['lightblue', 'lightgreen'])
axes[1].set_title('Loan Default Rates:\nUrban vs Rural')
axes[1].set_ylabel('Default Rate')
for i, v in enumerate(default_rates.values):
    axes[1].text(i, v + 0.005, f'{v:.1%}', ha='center', fontweight='bold')

# Average Transaction Value
transaction_values = df.groupby('area_type')['avg_transaction_value'].mean()
axes[2].bar(transaction_values.index, transaction_values.values,
           color=['lightblue', 'lightgreen'])
axes[2].set_title('Average Transaction Value:\nUrban vs Rural')
axes[2].set_ylabel('GHS')
for i, v in enumerate(transaction_values.values):
    axes[2].text(i, v + 10, f'GHS {v:.0f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('urban_rural_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# VISUALIZATION 3: Key Factors Affecting Loan Default
print("Creating Visualization 3: Key Risk Factors...")

# Calculate average feature values for defaulters vs non-defaulters
defaulters = df[df['will_default'] == 1]
non_defaulters = df[df['will_default'] == 0]

key_factors = {
    'Transaction Consistency': {
        'Good Customers': non_defaulters['transaction_consistency'].mean(),
        'Defaulters': defaulters['transaction_consistency'].mean()
    },
    'Bill Payment Punctuality': {
        'Good Customers': non_defaulters['bill_payment_punctuality'].mean(),
        'Defaulters': defaulters['bill_payment_punctuality'].mean()
    },
    'Savings Rate': {
        'Good Customers': non_defaulters['savings_accumulation_rate'].mean() * 100,
        'Defaulters': defaulters['savings_accumulation_rate'].mean() * 100
    },
    'Income Stability': {
        'Good Customers': non_defaulters['income_stability'].mean(),
        'Defaulters': defaulters['income_stability'].mean()
    }
}

# Create comparison chart
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i, (factor, values) in enumerate(key_factors.items()):
    categories = list(values.keys())
    scores = list(values.values())
    
    bars = axes[i].bar(categories, scores, color=['green', 'red'], alpha=0.7)
    axes[i].set_title(f'{factor}\nComparison', fontweight='bold')
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        if factor == 'Savings Rate':
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                       f'{score:.1f}%', ha='center', fontweight='bold')
        else:
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{score:.2f}', ha='center', fontweight='bold')
    
    # Set y-axis limits appropriately
    if factor == 'Savings Rate':
        axes[i].set_ylim(0, max(scores) + 2)
    else:
        axes[i].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('risk_factors.png', dpi=300, bbox_inches='tight')
plt.show()

# VISUALIZATION 4: Regional Performance Overview
print("Creating Visualization 4: Regional Performance...")

# Select top 8 regions for clarity
regional_data = df.groupby('region').agg({
    'will_default': 'mean',
    'avg_transaction_value': 'mean',
    'customer_id': 'count'
}).round(3)

regional_data = regional_data.rename(columns={'customer_id': 'customer_count'})
regional_data = regional_data.nlargest(8, 'customer_count')  # Top 8 by customer count

plt.figure(figsize=(12, 8))

# Create subplots
plt.subplot(2, 1, 1)
plt.bar(regional_data.index, regional_data['avg_transaction_value'], 
        color='skyblue', alpha=0.7)
plt.title('Average Transaction Value by Region\n(Top 8 Regions by Customer Count)', 
          fontweight='bold')
plt.ylabel('GHS')
plt.xticks(rotation=45)

# Add value labels
for i, v in enumerate(regional_data['avg_transaction_value']):
    plt.text(i, v + 10, f'GHS {v:.0f}', ha='center', fontsize=9)

plt.subplot(2, 1, 2)
plt.bar(regional_data.index, regional_data['will_default'], 
        color='lightcoral', alpha=0.7)
plt.title('Default Rates by Region', fontweight='bold')
plt.ylabel('Default Rate')
plt.xticks(rotation=45)

# Add value labels
for i, v in enumerate(regional_data['will_default']):
    plt.text(i, v + 0.005, f'{v:.1%}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('regional_performance.png', dpi=300, bbox_inches='tight')
plt.show()

# Summary for Stakeholders
print("\n" + "="*50)
print("KEY INSIGHTS FOR STAKEHOLDERS")
print("="*50)

print("1. MARKET STRUCTURE:")
print(f"   • MTN dominates with {mtn_customers/total_customers:.0%} market share")
print(f"   • {urban_customers/total_customers:.0%} of customers are in urban areas")

print("\n2. RISK PATTERNS:")
print(f"   • Overall default rate: {overall_default_rate:.1%}")
print(f"   • Urban default rate: {df[df['area_type']=='Urban']['will_default'].mean():.1%}")
print(f"   • Rural default rate: {df[df['area_type']=='Rural']['will_default'].mean():.1%}")

print("\n3. BEHAVIORAL DIFFERENCES:")
print(f"   • Urban transactions: GHS {df[df['area_type']=='Urban']['avg_transaction_value'].mean():.0f}")
print(f"   • Rural transactions: GHS {df[df['area_type']=='Rural']['avg_transaction_value'].mean():.0f}")

print("\n4. KEY RISK INDICATORS:")
print(f"   • Good customers have {non_defaulters['transaction_consistency'].mean():.0%} transaction consistency")
print(f"   • Defaulters have {defaulters['transaction_consistency'].mean():.0%} transaction consistency")

print("\n" + "="*50)
print("EXPLORATION COMPLETED")
print("="*50)
print("Generated 4 clear visualizations for stakeholder presentations:")
print("1. market_share.png - MTN dominance in mobile money")
print("2. urban_rural_comparison.png - Key differences by area type") 
print("3. risk_factors.png - Behavioral differences between good and risky customers")
print("4. regional_performance.png - Top regions by transaction value and default rates")