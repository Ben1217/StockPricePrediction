"""
Lightweight environment smoke test for the current QuantVision stack.
"""

print("=" * 60)
print("TESTING QUANTVISION ENVIRONMENT")
print("=" * 60)

passed = 0
failed = 0


def ok(message):
    global passed
    print(f"[OK] {message}")
    passed += 1


def fail(message):
    global failed
    print(f"[FAIL] {message}")
    failed += 1


print("\n[1/8] Core data libraries")
try:
    import pandas as pd
    import numpy as np
    import scipy

    ok(f"pandas {pd.__version__}, numpy {np.__version__}, scipy {scipy.__version__}")
except ImportError as exc:
    fail(str(exc))

print("\n[2/8] Data access libraries")
try:
    import yfinance  # noqa: F401
    import requests  # noqa: F401

    ok("yfinance and requests are installed")
except ImportError as exc:
    fail(str(exc))

print("\n[3/8] API runtime")
try:
    import fastapi  # noqa: F401
    import uvicorn  # noqa: F401
    import pydantic  # noqa: F401

    ok("fastapi, uvicorn, and pydantic are installed")
except ImportError as exc:
    fail(str(exc))

print("\n[4/8] Technical analysis stack")
try:
    import ta  # noqa: F401
    import cachetools  # noqa: F401

    ok("ta and cachetools are installed")
except ImportError as exc:
    fail(str(exc))

print("\n[5/8] Machine learning stack")
try:
    import sklearn  # noqa: F401
    import xgboost  # noqa: F401

    ok("scikit-learn and xgboost are installed")
except ImportError as exc:
    fail(str(exc))

print("\n[6/8] PyTorch LSTM stack")
try:
    import torch

    ok(f"torch {torch.__version__} is installed")
except Exception as exc:
    fail(str(exc))

print("\n[7/8] Quant and explainability stack")
try:
    import cvxpy  # noqa: F401
    import shap  # noqa: F401
    import hmmlearn  # noqa: F401

    ok("cvxpy, shap, and hmmlearn are installed")
except Exception as exc:
    fail(str(exc))

print("\n[8/8] Optional helpers")
try:
    import httpx  # noqa: F401
    import sqlalchemy  # noqa: F401
    import streamlit  # noqa: F401

    ok("httpx, sqlalchemy, and streamlit are installed")
except ImportError as exc:
    fail(str(exc))

print("\n" + "=" * 60)
print(f"SETUP TEST COMPLETE: {passed} passed, {failed} failed")
print("=" * 60)

print("\n[BONUS] Sample market-data request")
try:
    import yfinance as yf

    data = yf.download("SPY", start="2024-01-01", end="2024-01-10", progress=False)
    if data.empty:
        fail("Sample yfinance download returned no rows")
    else:
        ok(f"Downloaded {len(data)} SPY rows successfully")
except Exception as exc:
    fail(f"Sample download failed: {exc}")

print("\n" + "=" * 60)
if failed == 0:
    print("Environment looks ready for the current FastAPI + React project.")
elif passed >= 6:
    print("Most dependencies are available. Missing packages are likely optional.")
else:
    print("Several important dependencies are missing. Reinstall from requirements.txt.")
print("=" * 60)
