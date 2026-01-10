"""
API Configuration Test Script
Tests all API connections and key validity
"""
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("[WARN] python-dotenv not installed. Environment variables from .env won't be loaded.")


def test_yfinance():
    """Test yfinance (no key needed)"""
    try:
        import yfinance as yf
        data = yf.download('AAPL', period='1d', progress=False)
        if not data.empty:
            price = float(data['Close'].iloc[-1])
            return "[OK] Working", f"AAPL: ${price:.2f} - No API key required"
        return "[FAIL] Failed", "Unable to fetch data"
    except ImportError:
        return "[FAIL] Not Installed", "pip install yfinance"
    except Exception as e:
        return "[FAIL] Error", str(e)


def test_wikipedia():
    """Test Wikipedia S&P 500 list (no key needed)"""
    try:
        import requests
        import pandas as pd
        from io import StringIO
        
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        response = requests.get(url, headers=headers, timeout=10)
        tables = pd.read_html(StringIO(response.text))
        
        if len(tables) > 0 and len(tables[0]) > 400:
            return "[OK] Working", f"Found {len(tables[0])} S&P 500 companies - No API key required"
        return "[FAIL] Failed", "Could not parse S&P 500 table"
    except Exception as e:
        return "[FAIL] Error", str(e)


def test_alpha_vantage():
    """Test Alpha Vantage API (optional - not currently used in code)"""
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    
    if not api_key or api_key in ['your_alpha_vantage_key', 'your_key_here', '']:
        return "[SKIP] Not Configured", "Optional - Add key to .env if needed"
    
    try:
        import requests
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=AAPL&interval=5min&apikey={api_key}"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if "Error Message" in data:
            return "[FAIL] Invalid Key", "Check your API key"
        elif "Note" in data:
            return "[WARN] Rate Limited", "Free tier limit reached (5 calls/min)"
        elif "Time Series (5min)" in data:
            return "[OK] Working", f"Key valid ({api_key[:8]}...)"
        else:
            return "[UNKNOWN]", str(data)[:100]
    except Exception as e:
        return "[FAIL] Error", str(e)


def test_polygon():
    """Test Polygon.io API (optional - not currently used in code)"""
    api_key = os.getenv('POLYGON_API_KEY')
    
    if not api_key or api_key in ['your_polygon_key', 'your_key_here', '']:
        return "[SKIP] Not Configured", "Optional - Add key to .env if needed"
    
    try:
        import requests
        url = f"https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2023-01-09/2023-01-09?apiKey={api_key}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            return "[OK] Working", f"Key valid ({api_key[:8]}...)"
        elif response.status_code == 401:
            return "[FAIL] Invalid Key", "Authentication failed"
        elif response.status_code == 429:
            return "[WARN] Rate Limited", "Too many requests"
        else:
            return "[FAIL] Error", f"HTTP {response.status_code}"
    except Exception as e:
        return "[FAIL] Error", str(e)


def check_env_variables():
    """Check which environment variables are set"""
    print("\n" + "=" * 70)
    print("ENVIRONMENT VARIABLES CHECK")
    print("=" * 70)
    
    env_vars = {
        'USE_YFINANCE': ('Data Source', os.getenv('USE_YFINANCE', 'True')),
        'ALPHA_VANTAGE_API_KEY': ('API Key', os.getenv('ALPHA_VANTAGE_API_KEY', 'Not set')),
        'POLYGON_API_KEY': ('API Key', os.getenv('POLYGON_API_KEY', 'Not set')),
    }
    
    for var, (category, value) in env_vars.items():
        # Mask API keys
        if 'KEY' in var and value not in ['Not set', 'your_alpha_vantage_key', 'your_polygon_key']:
            display_value = f"{value[:4]}...{value[-4:]}" if len(value) > 8 else "****"
        else:
            display_value = value
        
        print(f"  {var:30} = {display_value}")


def main():
    print("=" * 70)
    print("API CONFIGURATION TEST")
    print("=" * 70)
    print("Testing all data source APIs...")
    
    tests = [
        ("yfinance", test_yfinance, "PRIMARY - Stock data (no key)"),
        ("Wikipedia", test_wikipedia, "PRIMARY - S&P 500 list (no key)"),
        ("Alpha Vantage", test_alpha_vantage, "OPTIONAL - Backup data source"),
        ("Polygon.io", test_polygon, "OPTIONAL - Intraday data"),
    ]
    
    results = []
    for name, test_func, purpose in tests:
        status, message = test_func()
        results.append((name, status, message, purpose))
        print(f"\n{name:20} {status:20}")
        print(f"  Purpose: {purpose}")
        print(f"  Status:  {message}")
    
    # Check environment variables
    check_env_variables()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    working = sum(1 for _, s, _, _ in results if "[OK]" in s)
    not_configured = sum(1 for _, s, _, _ in results if "[SKIP]" in s)
    failed = sum(1 for _, s, _, _ in results if "[FAIL]" in s)
    
    print(f"  [OK]   Working:        {working}")
    print(f"  [SKIP] Not Configured: {not_configured}")
    print(f"  [FAIL] Failed:         {failed}")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    print("""
  REQUIRED APIs (Must Work):
    - yfinance: Primary stock data source
    - Wikipedia: S&P 500 constituent list
  
  OPTIONAL APIs (Nice to Have):
    - Alpha Vantage: Backup data source (25 calls/day free)
    - Polygon.io: Intraday data (paid plans available)
  
  NOTE: Alpha Vantage and Polygon.io are NOT currently used in code.
        They are listed as potential fallback options only.
        You can safely leave these unconfigured.
    """)
    
    if working >= 2:
        print("  [SUCCESS] Primary APIs working! Dashboard is fully functional.")
    else:
        print("  [WARNING] Some primary APIs failed. Check errors above.")


if __name__ == "__main__":
    main()
