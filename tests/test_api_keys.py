"""
Environment and data-source configuration check for the current project.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("[WARN] python-dotenv not installed. .env values will not be loaded automatically.")


def test_yfinance():
    """Test yfinance access."""
    try:
        import yfinance as yf

        data = yf.download("AAPL", period="1d", progress=False)
        if not data.empty:
            price = float(data["Close"].iloc[-1])
            return "[OK] Working", f"AAPL: ${price:.2f} - No API key required"
        return "[FAIL] Failed", "Unable to fetch data"
    except ImportError:
        return "[FAIL] Not Installed", "pip install yfinance"
    except Exception as exc:
        return "[FAIL] Error", str(exc)


def test_wikipedia():
    """Test S&P 500 constituent lookup."""
    try:
        import pandas as pd
        import requests
        from io import StringIO

        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        tables = pd.read_html(StringIO(response.text))

        if tables and len(tables[0]) > 400:
            return "[OK] Working", f"Found {len(tables[0])} S&P 500 companies - No API key required"
        return "[FAIL] Failed", "Could not parse the S&P 500 table"
    except Exception as exc:
        return "[FAIL] Error", str(exc)


def test_alpha_vantage():
    """Test Alpha Vantage when configured."""
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

    if not api_key or api_key in ["your_alpha_vantage_key", "your_key_here", ""]:
        return "[SKIP] Not Configured", "Optional - set ALPHA_VANTAGE_API_KEY only if you want fallback endpoints"

    try:
        import requests

        url = (
            "https://www.alphavantage.co/query"
            f"?function=GLOBAL_QUOTE&symbol=AAPL&apikey={api_key}"
        )
        response = requests.get(url, timeout=10)
        data = response.json()

        if "Error Message" in data:
            return "[FAIL] Invalid Key", "Check your Alpha Vantage key"
        if "Note" in data:
            return "[WARN] Rate Limited", "Alpha Vantage rate limit reached"
        if "Global Quote" in data:
            return "[OK] Working", f"Key valid ({api_key[:8]}...)"
        return "[UNKNOWN]", str(data)[:120]
    except Exception as exc:
        return "[FAIL] Error", str(exc)


def test_anthropic_env():
    """Check whether Anthropic support is configured for optional agent flows."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key in ["your_anthropic_api_key", "your_key_here", ""]:
        return "[SKIP] Not Configured", "Optional - set ANTHROPIC_API_KEY only if you want agent workflows"
    return "[OK] Present", f"Anthropic key detected ({api_key[:8]}...)"


def check_env_variables():
    """Print the key environment values used by the project."""
    print("\n" + "=" * 70)
    print("ENVIRONMENT VARIABLES CHECK")
    print("=" * 70)

    env_vars = {
        "ALPHA_VANTAGE_API_KEY": os.getenv("ALPHA_VANTAGE_API_KEY", "Not set"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", "Not set"),
        "QUANTVISION_API_URL": os.getenv("QUANTVISION_API_URL", "http://localhost:8000"),
        "POSTGRES_HOST": os.getenv("POSTGRES_HOST", "localhost"),
        "POSTGRES_DB": os.getenv("POSTGRES_DB", "stock_data"),
    }

    for key, value in env_vars.items():
        if "KEY" in key and value not in ["Not set", "your_alpha_vantage_key", "your_anthropic_api_key"]:
            display = f"{value[:4]}...{value[-4:]}" if len(value) > 8 else "****"
        else:
            display = value
        print(f"  {key:24} = {display}")


def main():
    print("=" * 70)
    print("QUANTVISION DATA-SOURCE CHECK")
    print("=" * 70)
    print("Testing the services used by the current project...")

    tests = [
        ("yfinance", test_yfinance, "Primary market data source"),
        ("Wikipedia", test_wikipedia, "S&P 500 constituent lookup"),
        ("Alpha Vantage", test_alpha_vantage, "Optional fallback data source"),
        ("Anthropic", test_anthropic_env, "Optional CrewAI agent support"),
    ]

    results = []
    for name, test_func, purpose in tests:
        status, message = test_func()
        results.append((name, status, message, purpose))
        print(f"\n{name:20} {status:20}")
        print(f"  Purpose: {purpose}")
        print(f"  Status:  {message}")

    check_env_variables()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    working = sum(1 for _, status, _, _ in results if "[OK]" in status)
    skipped = sum(1 for _, status, _, _ in results if "[SKIP]" in status)
    failed = sum(1 for _, status, _, _ in results if "[FAIL]" in status)

    print(f"  [OK]   Working:        {working}")
    print(f"  [SKIP] Not Configured: {skipped}")
    print(f"  [FAIL] Failed:         {failed}")

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print(
        """
  Required for the main app:
    - yfinance
    - Wikipedia S&P 500 lookup

  Optional:
    - Alpha Vantage for fallback/live quote endpoints
    - Anthropic for CrewAI agent workflows

  The FastAPI backend and React frontend do not require Alpha Vantage or Anthropic
  unless you explicitly want those optional features.
        """
    )

    if working >= 2:
        print("  [SUCCESS] Core services are available for the current project.")
    else:
        print("  [WARNING] A core data source failed. Review the errors above.")


if __name__ == "__main__":
    main()
