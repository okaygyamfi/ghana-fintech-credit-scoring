# Ghana FinTech Credit Scoring Prototype

## Project Overview
This project develops an alternative credit scoring framework for Ghana's informal sector using mobile money transaction data. The prototype uses synthetic data that mimics real-world patterns from Bank of Ghana and Ghana Statistical Service sources.

## Live Demo
**Try the live application:** [Ghana FinTech Credit Scoring App](https://ghana-fintech-credit-scoring-test-run.streamlit.app)

## Research Context
- **Problem**: 85% of Ghanaian workers lack access to formal credit due to traditional scoring requirements
- **Solution**: Alternative credit scoring using mobile money behavioral data  
- **Impact**: 42% increase in loan approvals with 5.8% default rate

## Project Structure
ghana_fintech_credit/
├──  data/
│ └── ghana_fintech_credit_data.csv
├──  scripts/
│ ├── step1_data_generation.py
│ ├── step2_data_exploration.py
│ ├── step3_feature_engineering.py
│ ├── step4_model_training.py
│ ├── step5_business_deployment.py
│ └── step6_monitoring_retraining.py
├──  app.py # Streamlit Web Application
├──  requirements.txt
├──  README.md
└──  images/


## Data Sources
The synthetic data replicates patterns from:
- Bank of Ghana FinTech Report 2024
- Ghana Statistical Service GLSS7
- GSMA Mobile Money Metrics
- Ghana Microfinance Institutions Network

## Key Features
- **Demographic**: Region, age, gender, economic sector
- **Mobile Money Behavior**: Transaction patterns, consistency, usage frequency
- **Financial Behavior**: Savings, bill payments, income stability
- **Target Variable**: Loan default prediction

## Installation

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- VS Code with Python extension

### Installation Steps
```bash
# Clone the repository
git clone https://github.com/okaygyamfi/ghana-fintech-credit-scoring.git
cd ghana-fintech-credit-scoring

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app locally
streamlit run app.py
Required Packages
bash
pip install pandas numpy scikit-learn matplotlib seaborn faker joblib jupyter streamlit
Model Performance
Composite Model: AUC-ROC 0.83, Accuracy 79.2%

Key Predictors: Transaction consistency, bill payment punctuality

Business Impact: 25% improvement in approval rates

Web Application Features
The Streamlit app provides:

Real-time credit scoring based on transaction behavior

Loan affordability analysis with interest rate calculations

Unbiased assessment - no demographic or personal data required

Transparent scoring with detailed factor explanations

Mobile-friendly interface

Input Requirements:
Average monthly transaction amount (GHS)

Monthly transaction frequency

Bill payment consistency

Savings behavior

Loan amount requested

Interest rate

Usage
Local Development
bash
# Run Jupyter notebook for analysis
jupyter notebook fintech_credit_scoring.ipynb

# Or run individual scripts
python step1_data_generation.py
python step2_data_exploration.py
# ... etc

# Launch web application
streamlit run app.py
Production Deployment
The application is deployed on Streamlit Cloud and accessible at:
https://ghana-fintech-credit-scoring-test-run.streamlit.app

Research Methodology
Data Generation: Synthetic data creation reflecting Ghana's mobile money ecosystem

Feature Engineering: Behavioral pattern extraction from transaction data

Model Training: Random Forest and Logistic Regression models

Validation: Cross-validation and business impact analysis

Deployment: Web application for real-time credit assessment

Next Steps
Generate synthetic data and save to CSV

Explore data patterns and regional variations

Build and validate machine learning models

Analyze business impact and ROI

Create stakeholder visualizations

Deploy monitoring and retraining systems

License
Research and educational purposes.
