# Step 6: Model Monitoring and Retraining
# Establish monitoring system and retraining pipeline for long-term model performance

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime, timedelta
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

print("Starting model monitoring and retraining system...")

# Load current model and deployment configuration
print("Loading current production model...")

try:
    current_model = joblib.load('random_forest_model.pkl')
    deployment_config = joblib.load('deployment_config.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    scaler = joblib.load('feature_scaler.pkl')
    print("Production model and configuration loaded successfully")
except FileNotFoundError as e:
    print(f"Error loading production files: {e}")
    print("Please ensure Steps 1-5 were completed successfully")
    exit()

print(f"Current model version: {deployment_config['model_version']}")
print(f"Deployment date: {deployment_config['deployment_date']}")

# Create monitoring system
class ModelMonitor:
    def __init__(self, model, feature_columns, scaler):
        self.model = model
        self.feature_columns = feature_columns
        self.scaler = scaler
        self.performance_history = []
        self.data_drift_alerts = []
        
    def monitor_performance(self, X_new, y_true, period_name):
        """Monitor model performance on new data"""
        
        # Ensure features match training
        X_aligned = self._align_features(X_new)
        X_scaled = self.scaler.transform(X_aligned)
        
        # Make predictions
        y_pred = self.model.predict(X_scaled)
        y_prob = self.model.predict_proba(X_scaled)[:, 1]
        
        # Calculate metrics
        auc = roc_auc_score(y_true, y_prob)
        accuracy = (y_pred == y_true).mean()
        default_rate = y_true.mean()
        
        # Store performance
        performance_record = {
            'period': period_name,
            'date': datetime.now().strftime("%Y-%m-%d"),
            'auc_score': auc,
            'accuracy': accuracy,
            'default_rate': default_rate,
            'sample_size': len(y_true)
        }
        
        self.performance_history.append(performance_record)
        
        return performance_record
    
    def detect_data_drift(self, X_new, reference_data, threshold=0.1):
        """Detect significant changes in data distribution"""
        
        drift_detected = False
        drift_report = {}
        
        # Compare feature distributions (simplified approach)
        for feature in reference_data.columns:
            if feature in ['will_default', 'customer_id']:
                continue
                
            if reference_data[feature].dtype in ['float64', 'int64']:
                # For numerical features, compare means
                ref_mean = reference_data[feature].mean()
                new_mean = X_new[feature].mean()
                change_pct = abs((new_mean - ref_mean) / ref_mean)
                
                if change_pct > threshold:
                    drift_detected = True
                    drift_report[feature] = {
                        'change_percentage': change_pct,
                        'reference_mean': ref_mean,
                        'current_mean': new_mean
                    }
        
        return drift_detected, drift_report
    
    def check_retraining_need(self, current_performance, drift_detected):
        """Determine if model retraining is needed"""
        
        retraining_needed = False
        reasons = []
        
        # Performance degradation check
        if current_performance['auc_score'] < 0.78:  # Below acceptable threshold
            retraining_needed = True
            reasons.append(f"Performance degradation: AUC dropped to {current_performance['auc_score']:.3f}")
        
        # Data drift check
        if drift_detected:
            retraining_needed = True
            reasons.append("Significant data drift detected")
        
        # Time-based retraining (every 6 months as per research)
        deployment_date = datetime.strptime(deployment_config['deployment_date'], "%Y-%m-%d")
        months_since_deployment = (datetime.now() - deployment_date).days / 30
        
        if months_since_deployment >= 6:
            retraining_needed = True
            reasons.append(f"Scheduled retraining: {months_since_deployment:.1f} months since deployment")
        
        return retraining_needed, reasons
    
    def retrain_model(self, X_train, y_train, X_val, y_val):
        """Retrain model with new data"""
        
        print("Starting model retraining...")
        
        # Use same parameters as original model
        new_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=50,
            min_samples_leaf=20,
            class_weight='balanced',
            random_state=42
        )
        
        # Train new model
        new_model.fit(X_train, y_train)
        
        # Validate performance
        val_predictions = new_model.predict(X_val)
        val_probabilities = new_model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_probabilities)
        
        print(f"New model validation AUC: {val_auc:.3f}")
        
        return new_model, val_auc
    
    def _align_features(self, X_new):
        """Ensure new data has same features as training data"""
        
        X_aligned = X_new.copy()
        
        # Add missing columns
        for col in self.feature_columns:
            if col not in X_aligned.columns:
                X_aligned[col] = 0
        
        # Remove extra columns and reorder
        X_aligned = X_aligned[self.feature_columns]
        
        return X_aligned

# Initialize monitoring system
monitor = ModelMonitor(current_model, feature_columns, scaler)

# Simulate monitoring over time with different scenarios
print("\n" + "="*60)
print("MODEL MONITORING SIMULATION")
print("="*60)

# Load original data for reference
original_data = pd.read_csv("D:\ppp\ghana_fintech_credit\data\ghana_fintech_credit_data.csv")
X_original = pd.read_csv("D:\ppp\ghana_fintech_credit\data\X_train_processed.csv")
y_original = pd.read_csv("D:\ppp\ghana_fintech_credit\data\y_train.csv").squeeze()

# Scenario 1: Initial Performance (Month 0)
print("\nScenario 1: Initial Performance (Month 0)")
initial_performance = monitor.monitor_performance(X_original, y_original, "Initial")
print(f"AUC: {initial_performance['auc_score']:.3f}")
print(f"Accuracy: {initial_performance['accuracy']:.3f}")
print(f"Default Rate: {initial_performance['default_rate']:.3f}")

# Scenario 2: Simulate Performance After 3 Months
print("\nScenario 2: Performance After 3 Months")
# Simulate slight performance degradation
X_3month = X_original.copy()
y_3month = y_original.copy()

# Simulate some data drift (slight changes in feature distributions)
drift_factor = 1.05  # 5% drift
for col in X_3month.select_dtypes(include=[np.number]).columns:
    if 'flag' not in col:  # Don't drift binary flags
        X_3month[col] = X_3month[col] * np.random.normal(1, 0.02, len(X_3month))

month3_performance = monitor.monitor_performance(X_3month, y_3month, "Month 3")
print(f"AUC: {month3_performance['auc_score']:.3f}")
print(f"Accuracy: {month3_performance['accuracy']:.3f}")

# Scenario 3: Simulate Performance After 6 Months (Time for retraining)
print("\nScenario 3: Performance After 6 Months")
X_6month = X_original.copy()
y_6month = y_original.copy()

# Simulate more significant drift
for col in X_6month.select_dtypes(include=[np.number]).columns:
    if 'flag' not in col:
        X_6month[col] = X_6month[col] * np.random.normal(1, 0.05, len(X_6month))

month6_performance = monitor.monitor_performance(X_6month, y_6month, "Month 6")
print(f"AUC: {month6_performance['auc_score']:.3f}")
print(f"Accuracy: {month6_performance['accuracy']:.3f}")

# Check if retraining is needed
print("\n" + "="*60)
print("RETRAINING ASSESSMENT")
print("="*60)

# Detect data drift
drift_detected, drift_report = monitor.detect_data_drift(X_6month, X_original)

if drift_detected:
    print("Data Drift Detected in Features:")
    for feature, details in list(drift_report.items())[:5]:  # Show top 5
        print(f"  {feature}: {details['change_percentage']:.1%} change")

retraining_needed, reasons = monitor.check_retraining_need(month6_performance, drift_detected)

if retraining_needed:
    print("\nRETRAINING RECOMMENDED")
    for reason in reasons:
        print(f"  • {reason}")
    
    # Simulate retraining process
    print("\nInitiating retraining process...")
    
    # Split data for retraining (80% train, 20% validation)
    split_idx = int(len(X_6month) * 0.8)
    X_retrain = X_6month[:split_idx]
    y_retrain = y_6month[:split_idx]
    X_validate = X_6month[split_idx:]
    y_validate = y_6month[split_idx:]
    
    # Retrain model
    new_model, new_auc = monitor.retrain_model(X_retrain, y_retrain, X_validate, y_validate)
    
    # Compare with old model
    improvement = new_auc - month6_performance['auc_score']
    print(f"Performance improvement: {improvement:+.3f}")
    
    if improvement > 0:
        print("New model performs better. Updating production model...")
        
        # Update model and configuration
        monitor.model = new_model
        deployment_config['model_version'] = '1.1'
        deployment_config['retraining_date'] = datetime.now().strftime("%Y-%m-%d")
        deployment_config['previous_version'] = '1.0'
        deployment_config['performance_improvement'] = improvement
        
        # Save updated model
        joblib.dump(new_model, 'random_forest_model_v1.1.pkl')
        joblib.dump(deployment_config, 'deployment_config_v1.1.pkl')
        print("New model saved as version 1.1")
        
else:
    print("\nNo retraining needed at this time")
    print("Model performance remains within acceptable limits")

# Create monitoring dashboard
print("\nCreating monitoring dashboard...")

# Performance trends visualization
performance_df = pd.DataFrame(monitor.performance_history)

plt.figure(figsize=(15, 10))

# Plot 1: AUC Score Trend
plt.subplot(2, 3, 1)
plt.plot(performance_df['period'], performance_df['auc_score'], 'bo-', linewidth=2, markersize=8)
plt.axhline(y=0.78, color='red', linestyle='--', label='Minimum Acceptable AUC')
plt.ylabel('AUC Score')
plt.title('Model Performance Trend\n(AUC Score Over Time)', fontweight='bold')
plt.xticks(rotation=45)
plt.legend()
plt.grid(alpha=0.3)

# Plot 2: Accuracy Trend
plt.subplot(2, 3, 2)
plt.plot(performance_df['period'], performance_df['accuracy'], 'go-', linewidth=2, markersize=8)
plt.ylabel('Accuracy')
plt.title('Accuracy Trend Over Time', fontweight='bold')
plt.xticks(rotation=45)
plt.grid(alpha=0.3)

# Plot 3: Default Rate Trend
plt.subplot(2, 3, 3)
plt.plot(performance_df['period'], performance_df['default_rate'], 'ro-', linewidth=2, markersize=8)
plt.axhline(y=0.07, color='red', linestyle='--', label='Maximum Acceptable Default Rate')
plt.ylabel('Default Rate')
plt.title('Portfolio Default Rate Trend', fontweight='bold')
plt.xticks(rotation=45)
plt.legend()
plt.grid(alpha=0.3)

# Plot 4: Data Drift Analysis (simulated)
plt.subplot(2, 3, 4)
if drift_detected:
    drift_features = list(drift_report.keys())[:5]
    drift_magnitudes = [drift_report[feature]['change_percentage'] for feature in drift_features]
    
    plt.barh(drift_features, drift_magnitudes, color='orange')
    plt.xlabel('Change Percentage')
    plt.title('Top 5 Features with Data Drift', fontweight='bold')
else:
    plt.text(0.5, 0.5, 'No Significant Data Drift Detected', 
             ha='center', va='center', transform=plt.gca().transAxes, fontweight='bold')
    plt.title('Data Drift Status', fontweight='bold')

# Plot 5: Retraining Schedule
plt.subplot(2, 3, 5)
months = [0, 3, 6, 9, 12]
status = ['Completed', 'Completed', 'Scheduled' if retraining_needed else 'Monitoring', 'Scheduled', 'Scheduled']
colors = ['green', 'green', 'orange' if retraining_needed else 'blue', 'gray', 'gray']

plt.bar(months, [1, 1, 1, 1, 1], color=colors, alpha=0.7)
plt.xlabel('Months Since Deployment')
plt.title('Model Retraining Schedule\n(Every 6 Months)', fontweight='bold')
plt.xticks(months)

# Add status labels
for i, (month, stat) in enumerate(zip(months, status)):
    plt.text(month, 0.5, stat, ha='center', va='center', fontweight='bold')

# Plot 6: Alert Summary
plt.subplot(2, 3, 6)
alert_types = ['Performance', 'Data Drift', 'Fairness', 'System']
alert_counts = [1 if month6_performance['auc_score'] < 0.78 else 0, 
                1 if drift_detected else 0, 0, 0]  # Simplified

colors = ['red' if count > 0 else 'green' for count in alert_counts]
plt.bar(alert_types, alert_counts, color=colors, alpha=0.7)
plt.ylabel('Active Alerts')
plt.title('Monitoring Alert Summary', fontweight='bold')

# Add values on bars
for i, v in enumerate(alert_counts):
    plt.text(i, v + 0.05, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('model_monitoring_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()

# Create automated monitoring report
print("\n" + "="*60)
print("AUTOMATED MONITORING REPORT")
print("="*60)

print(f"Report Date: {datetime.now().strftime('%Y-%m-%d')}")
print(f"Model Version: {deployment_config['model_version']}")

print("\nPERFORMANCE SUMMARY:")
latest_perf = monitor.performance_history[-1]
print(f"Current AUC: {latest_perf['auc_score']:.3f} ({'✓ ACCEPTABLE' if latest_perf['auc_score'] >= 0.78 else '✗ NEEDS ATTENTION'})")
print(f"Current Accuracy: {latest_perf['accuracy']:.3f}")
print(f"Current Default Rate: {latest_perf['default_rate']:.3f}")

print("\nDATA QUALITY:")
print(f"Data Drift: {'DETECTED' if drift_detected else 'NONE'}")
if drift_detected:
    print(f"Features with drift: {len(drift_report)}")

print("\nRETRAINING STATUS:")
if retraining_needed:
    print("STATUS: RETRAINING REQUIRED")
    print("Reasons:")
    for reason in reasons:
        print(f"  - {reason}")
else:
    print("STATUS: NO RETRAINING NEEDED")
    deployment_date = datetime.strptime(deployment_config['deployment_date'], "%Y-%m-%d")
    days_until_retraining = 180 - (datetime.now() - deployment_date).days
    print(f"Next scheduled retraining in: {days_until_retraining} days")

print("\nRECOMMENDED ACTIONS:")
if retraining_needed:
    print("1. Schedule model retraining immediately")
    print("2. Review data drift patterns")
    print("3. Validate new model performance")
    print("4. Plan production deployment")
else:
    print("1. Continue monitoring performance")
    print("2. Review feature distributions monthly")
    print("3. Prepare for next scheduled retraining")

# Save monitoring history
monitoring_history = {
    'performance_history': monitor.performance_history,
    'data_drift_alerts': monitor.data_drift_alerts,
    'last_retraining_check': datetime.now().strftime("%Y-%m-%d"),
    'retraining_recommended': retraining_needed
}

joblib.dump(monitoring_history, 'monitoring_history.pkl')

print("\n" + "="*60)
print("MONITORING SYSTEM DEPLOYED SUCCESSFULLY")
print("="*60)
print("Key monitoring components established:")
print("✓ Performance tracking (AUC, accuracy, default rates)")
print("✓ Data drift detection")
print("✓ Automated retraining assessment")
print("✓ Visual monitoring dashboard")
print("✓ Scheduled retraining every 6 months")

print("\nMonitoring files created:")
print("• model_monitoring_dashboard.png - Visual monitoring overview")
print("• monitoring_history.pkl - Historical performance data")
print("• random_forest_model_v1.1.pkl - Retrained model (if applicable)")

print("\nNext steps:")
print("1. Integrate with production data pipeline")
print("2. Set up automated alert system")
print("3. Establish model governance process")
print("4. Define rollback procedures for model updates")