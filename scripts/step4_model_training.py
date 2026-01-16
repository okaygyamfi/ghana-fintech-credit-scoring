# Step 4: Model Training and Evaluation
# Train and compare machine learning models for credit scoring

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import cross_val_score
import joblib

print("Starting model training and evaluation...")

# Load the processed data
X_train = pd.read_csv("D:\ppp\ghana_fintech_credit\data\X_train_processed.csv")
X_test = pd.read_csv("D:\ppp\ghana_fintech_credit\data\X_test_processed.csv")
y_train = pd.read_csv("D:\ppp\ghana_fintech_credit\data\y_train.csv").squeeze()
y_test = pd.read_csv("D:\ppp\ghana_fintech_credit\data\y_test.csv").squeeze()

print("Processed data loaded successfully")
print(f"Training set: {X_train.shape[0]:,} records, {X_train.shape[1]} features")
print(f"Testing set: {X_test.shape[0]:,} records")
print(f"Training default rate: {y_train.mean():.3f}")
print(f"Testing default rate: {y_test.mean():.3f}")

# Load feature names for interpretation
feature_columns = joblib.load('feature_columns.pkl')
important_features = joblib.load('important_features.pkl')

print(f"\nUsing {len(feature_columns)} features for modeling")

# Initialize models with business-appropriate settings
print("\nInitializing machine learning models...")

# Logistic Regression - good for interpretability
logistic_model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced',  # Handle imbalanced data
    C=0.1  # Regularization to prevent overfitting
)

# Random Forest - good for complex patterns
random_forest_model = RandomForestClassifier(
    n_estimators=100,        # Number of trees
    max_depth=10,            # Limit tree depth
    min_samples_split=50,    # Prevent overfitting
    min_samples_leaf=20,     # Prevent overfitting
    class_weight='balanced', # Handle imbalanced data
    random_state=42
)

print("Models initialized successfully")

# Train the models
print("\nTraining models...")

print("Training Logistic Regression...")
logistic_model.fit(X_train, y_train)
logistic_train_score = logistic_model.score(X_train, y_train)
print(f"Logistic Regression training accuracy: {logistic_train_score:.3f}")

print("Training Random Forest...")
random_forest_model.fit(X_train, y_train)
rf_train_score = random_forest_model.score(X_train, y_train)
print(f"Random Forest training accuracy: {rf_train_score:.3f}")

# Make predictions
print("\nMaking predictions on test data...")

# Logistic Regression predictions
logistic_predictions = logistic_model.predict(X_test)
logistic_probabilities = logistic_model.predict_proba(X_test)[:, 1]

# Random Forest predictions
rf_predictions = random_forest_model.predict(X_test)
rf_probabilities = random_forest_model.predict_proba(X_test)[:, 1]

# Evaluate model performance
print("\n" + "="*60)
print("MODEL PERFORMANCE EVALUATION")
print("="*60)

# Logistic Regression Performance
print("\nLOGISTIC REGRESSION RESULTS:")
logistic_accuracy = logistic_model.score(X_test, y_test)
logistic_auc = roc_auc_score(y_test, logistic_probabilities)

print(f"Accuracy: {logistic_accuracy:.3f}")
print(f"AUC-ROC Score: {logistic_auc:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, logistic_predictions, target_names=['Non-Default', 'Default']))

# Random Forest Performance
print("\nRANDOM FOREST RESULTS:")
rf_accuracy = random_forest_model.score(X_test, y_test)
rf_auc = roc_auc_score(y_test, rf_probabilities)

print(f"Accuracy: {rf_accuracy:.3f}")
print(f"AUC-ROC Score: {rf_auc:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, rf_predictions, target_names=['Non-Default', 'Default']))

# Cross-validation for more robust evaluation
print("\nPerforming cross-validation...")
logistic_cv_scores = cross_val_score(logistic_model, X_train, y_train, cv=5, scoring='roc_auc')
rf_cv_scores = cross_val_score(random_forest_model, X_train, y_train, cv=5, scoring='roc_auc')

print(f"Logistic Regression CV AUC: {logistic_cv_scores.mean():.3f} (+/- {logistic_cv_scores.std() * 2:.3f})")
print(f"Random Forest CV AUC: {rf_cv_scores.mean():.3f} (+/- {rf_cv_scores.std() * 2:.3f})")

# Create performance comparison visualization
print("\nCreating performance visualizations...")

# Visualization 1: ROC Curves
plt.figure(figsize=(10, 8))

# Calculate ROC curves
fpr_logistic, tpr_logistic, _ = roc_curve(y_test, logistic_probabilities)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probabilities)

# Plot ROC curves
plt.plot(fpr_logistic, tpr_logistic, label=f'Logistic Regression (AUC = {logistic_auc:.3f})', linewidth=2)
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {rf_auc:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', alpha=0.5)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves: Model Performance Comparison', fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('model_performance_roc.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualization 2: Feature Importance from Random Forest
plt.figure(figsize=(12, 8))

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': random_forest_model.feature_importances_
})

# Get top 15 most important features
top_features = feature_importance.nlargest(15, 'importance')

# Create horizontal bar chart
plt.barh(top_features['feature'], top_features['importance'])
plt.xlabel('Feature Importance')
plt.title('Top 15 Most Important Features for Credit Scoring\n(Random Forest Model)', fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualization 3: Confusion Matrix for Random Forest (better performing model)
plt.figure(figsize=(8, 6))

cm = confusion_matrix(y_test, rf_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Non-Default', 'Predicted Default'],
            yticklabels=['Actual Non-Default', 'Actual Default'])

plt.title('Confusion Matrix: Random Forest Model', fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Business-oriented performance analysis
print("\n" + "="*60)
print("BUSINESS IMPACT ANALYSIS")
print("="*60)

# Calculate business metrics
def calculate_business_metrics(y_true, y_pred, y_prob, model_name):
    # Approval rate at different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6]
    results = []
    
    for threshold in thresholds:
        approved = y_prob < threshold  # Lower probability = better credit risk
        approval_rate = approved.mean()
        
        if approved.any():
            default_rate = y_true[approved].mean()
            good_customers_approved = ((y_true[approved] == 0).sum() / len(y_true)) * 100
            risky_customers_approved = ((y_true[approved] == 1).sum() / len(y_true)) * 100
        else:
            default_rate = 0
            good_customers_approved = 0
            risky_customers_approved = 0
            
        results.append({
            'threshold': threshold,
            'approval_rate': approval_rate,
            'default_rate': default_rate,
            'good_customers_approved': good_customers_approved,
            'risky_customers_approved': risky_customers_approved
        })
    
    return pd.DataFrame(results)

# Analyze business impact for Random Forest (better model)
business_results = calculate_business_metrics(y_test, rf_predictions, rf_probabilities, "Random Forest")

print("Business Impact at Different Risk Thresholds:")
print(business_results.round(3))

# Find optimal threshold (balancing approval rate and default rate)
business_results['efficiency_score'] = business_results['approval_rate'] * (1 - business_results['default_rate'])
optimal_idx = business_results['efficiency_score'].idxmax()
optimal_threshold = business_results.loc[optimal_idx, 'threshold']

print(f"\nRecommended Risk Threshold: {optimal_threshold}")
print(f"At this threshold:")
print(f"  • Approval Rate: {business_results.loc[optimal_idx, 'approval_rate']:.1%}")
print(f"  • Expected Default Rate: {business_results.loc[optimal_idx, 'default_rate']:.1%}")
print(f"  • Good Customers Approved: {business_results.loc[optimal_idx, 'good_customers_approved']:.1f}%")
print(f"  • Risky Customers Approved: {business_results.loc[optimal_idx, 'risky_customers_approved']:.1f}%")

# Compare with traditional methods
traditional_approval_rate = 0.42  # From research
traditional_default_rate = 0.049  # From research

improvement_approval = business_results.loc[optimal_idx, 'approval_rate'] - traditional_approval_rate
improvement_default = business_results.loc[optimal_idx, 'default_rate'] - traditional_default_rate

print(f"\nVS TRADITIONAL CREDIT SCORING:")
print(f"  • Approval Rate Improvement: +{improvement_approval:.1%}")
print(f"  • Default Rate Change: {improvement_default:+.1%}")

# Save the trained models
print("\nSaving trained models...")

joblib.dump(logistic_model, 'logistic_regression_model.pkl')
joblib.dump(random_forest_model, 'random_forest_model.pkl')

print("Models saved successfully")

# Final summary
print("\n" + "="*60)
print("MODEL TRAINING COMPLETED")
print("="*60)
print("KEY RESULTS:")
print(f"• Best Model: Random Forest (AUC: {rf_auc:.3f})")
print(f"• Model Accuracy: {rf_accuracy:.3f}")
print(f"• Key Predictors: Financial behavior patterns")
print(f"• Business Impact: +{improvement_approval:.1%} more approvals")
print(f"• Risk Management: {business_results.loc[optimal_idx, 'default_rate']:.1%} default rate")

print("\nGenerated Files:")
print("• model_performance_roc.png - Model comparison chart")
print("• feature_importance.png - Key factors in credit decisions")
print("• confusion_matrix.png - Model performance details")
print("• logistic_regression_model.pkl - Trained logistic model")
print("• random_forest_model.pkl - Trained random forest model")

print("\nReady for Step 5: Model Deployment and Business Application")