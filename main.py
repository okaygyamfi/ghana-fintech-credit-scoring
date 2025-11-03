# main.py - Ghana FinTech Credit Scoring Prototype
# Main execution script that runs the complete prototype pipeline

import os
import sys
from datetime import datetime

def run_step_1():
    """Step 1: Generate realistic synthetic data"""
    print("\n" + "="*50)
    print("STEP 1: GENERATING REALISTIC SYNTHETIC DATA")
    print("="*50)
    
    try:
        from step1_data_generation import generate_ghana_fintech_data
        df = generate_ghana_fintech_data(10000)
        print("✓ Data generation completed successfully")
        return True
    except Exception as e:
        print(f"Error in Step 1: {e}")
        return False

def run_step_2():
    """Step 2: Simplified data exploration"""
    print("\n" + "="*50)
    print("STEP 2: DATA EXPLORATION AND VISUALIZATION")
    print("="*50)
    
    try:
        from step2_data_exploration import explore_data_for_stakeholders
        explore_data_for_stakeholders()
        print("✓ Data exploration completed successfully")
        return True
    except Exception as e:
        print(f"Error in Step 2: {e}")
        return False

def run_step_3():
    """Step 3: Feature engineering and preprocessing"""
    print("\n" + "="*50)
    print("STEP 3: FEATURE ENGINEERING AND PREPROCESSING")
    print("="*50)
    
    try:
        from step3_feature_engineering import prepare_features_for_modeling
        results = prepare_features_for_modeling()
        print("✓ Feature engineering completed successfully")
        return True
    except Exception as e:
        print(f"Error in Step 3: {e}")
        return False

def run_step_4():
    """Step 4: Model training and evaluation"""
    print("\n" + "="*50)
    print("STEP 4: MODEL TRAINING AND EVALUATION")
    print("="*50)
    
    try:
        from step4_model_training import train_and_evaluate_models
        results = train_and_evaluate_models()
        print("✓ Model training completed successfully")
        return True
    except Exception as e:
        print(f"Error in Step 4: {e}")
        return False

def run_step_5():
    """Step 5: Model deployment and business application"""
    print("\n" + "="*50)
    print("STEP 5: MODEL DEPLOYMENT AND BUSINESS APPLICATION")
    print("="*50)
    
    try:
        from step5_business_deployment import deploy_model_and_analyze_impact
        impact = deploy_model_and_analyze_impact()
        print("✓ Business deployment analysis completed")
        return True
    except Exception as e:
        print(f"Error in Step 5: {e}")
        return False

def run_step_6():
    """Step 6: Model monitoring and retraining setup"""
    print("\n" + "="*50)
    print("STEP 6: MODEL MONITORING AND RETRAINING SYSTEM")
    print("="*50)
    
    try:
        from step6_monitoring_retraining import setup_monitoring_system
        monitoring_system = setup_monitoring_system()
        print("✓ Monitoring system setup completed")
        return True
    except Exception as e:
        print(f"Error in Step 6: {e}")
        return False

def main():
    """Main execution function"""
    print("=" * 70)
    print("GHANA FINTECH CREDIT SCORING PROTOTYPE")
    print("=" * 70)
    print("A comprehensive solution for credit scoring in Ghana's informal sector")
    print("Using mobile money data to enable financial inclusion")
    print("=" * 70)
    
    try:
        print("Starting Ghana FinTech Credit Scoring Prototype...")
        print(f"Execution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run all steps
        run_step_1()
        run_step_2() 
        run_step_3()
        run_step_4()
        run_step_5()
        run_step_6()
        
        print("\n" + "="*70)
        print("PROTOTYPE EXECUTION COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\nExecution completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\nError in prototype execution: {e}")
        print("Please ensure all required files and dependencies are available.")
        sys.exit(1)

if __name__ == "__main__":
    main()