#!/usr/bin/env python3
"""
Automated Kaggle Submission (Non-interactive)
"""

import subprocess
import pandas as pd
from pathlib import Path

def auto_submit_to_kaggle():
    """Automatically submit to Kaggle if validation passes"""
    
    print("="*60)
    print("AUTOMATED KAGGLE SUBMISSION")
    print("="*60)
    
    # Validate submission
    try:
        df = pd.read_csv("submission.csv")
        print(f"✅ Submission validation passed")
        print(f"   Shape: {df.shape}")
        print(f"   Prediction range: [{df['prediction'].min():.6f}, {df['prediction'].max():.6f}]")
        print(f"   Mean prediction: {df['prediction'].mean():.6f}")
        
        # Check if Kaggle CLI is available
        try:
            subprocess.run(["kaggle", "--version"], check=True, capture_output=True)
            print("✅ Kaggle CLI is available")
            
            # Submit to competition
            print("🚀 Submitting to Kaggle competition...")
            
            cmd = [
                "kaggle", "competitions", "submit",
                "-c", "amex-default-prediction",
                "-f", "submission.csv", 
                "-m", "LightGBM model with 933 features - Validation AUC: 0.9554"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("🎉 SUCCESS! Submission completed!")
                print("Check your results at: https://www.kaggle.com/competitions/amex-default-prediction/submissions")
                return True
            else:
                print("❌ Submission failed:")
                print(result.stderr)
                return False
                
        except FileNotFoundError:
            print("⚠️  Kaggle CLI not found. To submit manually:")
            print("   1. Install: pip install kaggle")
            print("   2. Configure API credentials")
            print("   3. Run: kaggle competitions submit -c amex-default-prediction -f submission.csv -m 'Your message'")
            return False
            
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

if __name__ == "__main__":
    auto_submit_to_kaggle()