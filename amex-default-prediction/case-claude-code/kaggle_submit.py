#!/usr/bin/env python3
"""
Kaggle Submission Script for Credit Default Prediction
"""

import subprocess
import sys
from pathlib import Path

def submit_to_kaggle(competition_name: str = "amex-default-prediction", 
                    submission_file: str = "submission.csv",
                    message: str = "LightGBM ensemble model with advanced feature engineering"):
    """
    Submit to Kaggle competition
    
    Args:
        competition_name: Name of the Kaggle competition
        submission_file: Path to submission CSV file
        message: Submission message
    """
    
    # Check if submission file exists
    if not Path(submission_file).exists():
        print(f"Error: Submission file {submission_file} not found!")
        return False
    
    # Check if kaggle CLI is installed
    try:
        subprocess.run(["kaggle", "--version"], check=True, capture_output=True)
        print("Kaggle CLI is installed ✓")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: Kaggle CLI not installed. Install with: pip install kaggle")
        print("Also make sure to configure your Kaggle API credentials.")
        return False
    
    # Submit to competition
    try:
        print(f"Submitting {submission_file} to {competition_name}...")
        
        cmd = [
            "kaggle", "competitions", "submit",
            "-c", competition_name,
            "-f", submission_file,
            "-m", message
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Submission successful!")
            print(result.stdout)
            return True
        else:
            print("❌ Submission failed!")
            print(result.stderr)
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Submission failed with error: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_submission_format(submission_file: str = "submission.csv"):
    """Validate submission file format"""
    
    import pandas as pd
    
    try:
        df = pd.read_csv(submission_file)
        
        print(f"Submission file validation:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        
        # Check required columns
        if 'customer_ID' not in df.columns or 'prediction' not in df.columns:
            print("❌ Missing required columns: customer_ID, prediction")
            return False
        
        # Check prediction range
        pred_min, pred_max = df['prediction'].min(), df['prediction'].max()
        print(f"  Prediction range: [{pred_min:.6f}, {pred_max:.6f}]")
        
        if pred_min < 0 or pred_max > 1:
            print("⚠️  Warning: Predictions should be probabilities between 0 and 1")
        
        # Check for missing values
        missing_customers = df['customer_ID'].isnull().sum()
        missing_predictions = df['prediction'].isnull().sum()
        
        if missing_customers > 0 or missing_predictions > 0:
            print(f"❌ Missing values: {missing_customers} customer_IDs, {missing_predictions} predictions")
            return False
        
        print("✅ Submission format is valid!")
        return True
        
    except Exception as e:
        print(f"❌ Error validating submission: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("KAGGLE SUBMISSION SCRIPT")
    print("="*60)
    
    # Validate submission format
    if check_submission_format():
        print()
        
        # Ask user for confirmation
        response = input("Proceed with Kaggle submission? (y/n): ").lower().strip()
        
        if response == 'y' or response == 'yes':
            # Submit to Kaggle
            success = submit_to_kaggle(
                competition_name="amex-default-prediction",
                submission_file="submission.csv",
                message="Advanced LightGBM model with comprehensive feature engineering - AUC 0.9554"
            )
            
            if success:
                print("\n🎉 Successfully submitted to Kaggle!")
                print("Check your submissions at: https://www.kaggle.com/competitions/amex-default-prediction/submissions")
            else:
                print("\n❌ Submission failed. Please check error messages above.")
        else:
            print("Submission cancelled.")
    else:
        print("Please fix submission format issues before submitting.")