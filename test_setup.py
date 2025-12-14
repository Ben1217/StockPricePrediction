"""
Test script to verify all libraries are installed correctly
"""

print("=" * 50)
print("TESTING STOCK PREDICTION ENVIRONMENT")
print("=" * 50)

passed = 0
failed = 0

# Test 1: Data Libraries
print("\n[1/8] Testing Data Libraries...")
try:
    import pandas as pd
    import numpy as np
    import scipy
    print(f"✅ pandas {pd.__version__}, numpy {np.__version__}, scipy {scipy.__version__}")
    passed += 1
except ImportError as e:
    print(f"❌ Error: {e}")
    failed += 1

# Test 2: Data Acquisition
print("\n[2/8] Testing Data Acquisition...")
try:
    import yfinance as yf
    print("✅ yfinance installed")
    passed += 1
except ImportError as e:
    print(f"❌ Error: {e}")
    failed += 1

# Test 3: Visualization
print("\n[3/8] Testing Visualization...")
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly
    print("✅ matplotlib, seaborn, plotly installed")
    passed += 1
except ImportError as e:
    print(f"❌ Error: {e}")
    failed += 1

# Test 4: Technical Indicators
print("\n[4/8] Testing Technical Indicators...")
try:
    import ta
    print("✅ ta library installed")
    passed += 1
except ImportError as e:
    print(f"❌ Error: {e}")
    print("⚠️  Install with: pip install ta")
    failed += 1

# Test 5: Machine Learning
print("\n[5/8] Testing Machine Learning...")
try:
    from sklearn.ensemble import RandomForestRegressor
    import xgboost as xgb
    print("✅ scikit-learn, xgboost installed")
    passed += 1
except ImportError as e:
    print(f"❌ Error: {e}")
    failed += 1

# Test 6: Deep Learning
print("\n[6/8] Testing Deep Learning...")
try:
    import tensorflow as tf
    print(f"✅ TensorFlow {tf.__version__} installed")
    passed += 1
except Exception as e:
    print(f"❌ TensorFlow Error (DLL issue)")
    print("⚠️  Solutions:")
    print("   1. Install: https://aka.ms/vs/17/release/vc_redist.x64.exe")
    print("   2. Or use: pip uninstall tensorflow && pip install tensorflow-cpu")
    print("   3. Or skip TensorFlow and use XGBoost/Random Forest")
    failed += 1

# Test 7: Optimization
print("\n[7/8] Testing Optimization...")
try:
    import cvxpy as cp
    import shap
    print("✅ cvxpy, shap installed")
    passed += 1
except Exception as e:
    print(f"❌ Version conflict detected")
    print("⚠️  Fix with: pip install --upgrade numpy scipy shap cvxpy")
    failed += 1

# Test 8: Dashboard
print("\n[8/8] Testing Dashboard...")
try:
    import dash
    print("✅ Dash installed")
    passed += 1
except ImportError as e:
    print(f"❌ Error: {e}")
    failed += 1

print("\n" + "=" * 50)
print(f"SETUP TEST COMPLETE! {passed}/8 passed, {failed}/8 failed")
print("=" * 50)

# Bonus: Download sample data
print("\n[BONUS] Testing data download...")
try:
    import yfinance as yf
    data = yf.download('SPY', start='2024-01-01', end='2024-01-10', progress=False)
    print(f"✅ Successfully downloaded {len(data)} days of SPY data")
    print(data.head())
    passed += 1
except Exception as e:
    print(f"❌ Data download failed: {e}")

print("\n" + "=" * 50)
if failed == 0:
    print("🎉 ALL TESTS PASSED! You're ready to start!")
elif passed >= 6:
    print("✅ MOSTLY WORKING! You can start with basic features.")
    print("   Fix remaining issues as needed.")
else:
    print("⚠️  Several issues detected. Run fixes above.")
print("=" * 50)