"""
yfinance Data Freshness Test Script
Tests whether yfinance provides live or delayed data
"""

import yfinance as yf
from datetime import datetime
import pytz
import pandas as pd

def test_data_freshness():
    """
    Test if yfinance provides live or delayed data
    """
    print("=" * 60)
    print("YFINANCE DATA FRESHNESS TEST")
    print("=" * 60)
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}")
    
    # Test with major stocks
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY']
    
    results = []
    
    for ticker in test_tickers:
        print(f"\n[TEST] Testing {ticker}...")
        
        result = {
            'Ticker': ticker,
            'Fast_Price': None,
            'Hist_Price': None,
            'Last_Update': None,
            'Delay_Min': None,
            'Status': None
        }
        
        # Get current data
        stock = yf.Ticker(ticker)
        
        # Method 1: Fast info (usually real-time)
        try:
            fast_info = stock.fast_info
            current_price = getattr(fast_info, 'last_price', None)
            if current_price:
                result['Fast_Price'] = current_price
                print(f"  Fast Info Price: ${current_price:.2f}")
        except Exception as e:
            print(f"  Fast Info: Not available ({e})")
        
        # Method 2: Historical data with 1-minute intervals (last available)
        try:
            hist = stock.history(period='1d', interval='1m')
            if not hist.empty:
                last_timestamp = hist.index[-1]
                last_price = hist['Close'].iloc[-1]
                
                result['Hist_Price'] = last_price
                
                # Convert to Eastern timezone (NYSE)
                if last_timestamp.tzinfo is None:
                    last_timestamp = pytz.utc.localize(last_timestamp)
                
                local_tz = pytz.timezone('America/New_York')
                last_timestamp_local = last_timestamp.astimezone(local_tz)
                
                result['Last_Update'] = last_timestamp_local.strftime('%Y-%m-%d %I:%M:%S %p %Z')
                
                print(f"  Last Price: ${last_price:.2f}")
                print(f"  Last Update: {result['Last_Update']}")
                
                # Calculate delay
                current_time = datetime.now(local_tz)
                delay = (current_time - last_timestamp_local).total_seconds() / 60
                result['Delay_Min'] = round(delay, 1)
                
                print(f"  Data Delay: {delay:.1f} minutes")
                
                # Determine data type
                if delay < 1:
                    status = "[GREEN] REAL-TIME (< 1 min)"
                elif delay < 15:
                    status = "[YELLOW] NEAR REAL-TIME (< 15 min)"
                elif delay < 30:
                    status = "[ORANGE] DELAYED (15-30 min)"
                else:
                    status = "[RED] SIGNIFICANTLY DELAYED (> 30 min)"
                
                result['Status'] = status
                print(f"  Status: {status}")
            else:
                print(f"  [X] No historical data available")
        except Exception as e:
            print(f"  [X] Error getting history: {e}")
        
        results.append(result)
    
    # Check market status
    print("\n" + "=" * 60)
    print("MARKET STATUS CHECK")
    print("=" * 60)
    
    market_status = "UNKNOWN"
    try:
        spy = yf.Ticker('SPY')
        info = spy.info
        market_status = info.get('marketState', 'UNKNOWN')
        print(f"Current Market State: {market_status}")
        print(f"(REGULAR = Market Open, CLOSED = Market Closed, PRE/POST = Extended Hours)")
        
        # Additional info
        if 'regularMarketTime' in info:
            market_time = datetime.fromtimestamp(info['regularMarketTime'])
            print(f"Regular Market Time: {market_time}")
    except Exception as e:
        print(f"Could not get market status: {e}")
    
    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    # Calculate average delay
    valid_delays = [r['Delay_Min'] for r in results if r['Delay_Min'] is not None]
    if valid_delays:
        avg_delay = sum(valid_delays) / len(valid_delays)
        print(f"\nAverage Delay: {avg_delay:.1f} minutes")
    else:
        avg_delay = None
    
    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    if market_status == 'CLOSED':
        print("""
[OK] Market is CLOSED - End-of-day data is expected and acceptable.
   -> Show: "Market Closed - Showing latest closing prices"
        """)
    elif market_status in ['REGULAR', 'PRE', 'POST']:
        if valid_delays and avg_delay and avg_delay < 15:
            print(f"""
[OK] Data appears NEAR REAL-TIME (avg delay: {avg_delay:.1f} min)
   -> Show: "Live Data - Updated every 5 minutes"
   -> Cache TTL of 5 minutes is appropriate
            """)
        elif avg_delay:
            print(f"""
[WARN] Data appears DELAYED (avg delay: {avg_delay:.1f} min)
   -> Show: "Delayed Data - Approximately 15-minute delay"
   -> Consider upgrading to paid API for real-time data
            """)
    else:
        print("""
[WARN] Could not determine market status
   -> Show last update timestamp for transparency
        """)
    
    print("""
GENERAL RECOMMENDATIONS:
1. Always display "Last Updated: [timestamp]" in the dashboard
2. Add disclaimer: "Data provided by Yahoo Finance. May be delayed."
3. For real-time needs, consider:
   - Polygon.io (paid) - True real-time
   - Alpha Vantage (free tier) - 5 API calls/min
   - IEX Cloud (paid) - Good for small projects
    """)
    
    return results, market_status


def check_market_hours():
    """Check if US market is currently open"""
    try:
        # Simple check using SPY market state
        spy = yf.Ticker('SPY')
        info = spy.info
        market_state = info.get('marketState', 'UNKNOWN')
        
        return {
            'state': market_state,
            'is_open': market_state == 'REGULAR',
            'is_extended': market_state in ['PRE', 'POST'],
            'is_closed': market_state == 'CLOSED'
        }
    except Exception as e:
        return {
            'state': 'ERROR',
            'is_open': False,
            'is_extended': False,
            'is_closed': True,
            'error': str(e)
        }


if __name__ == "__main__":
    results, market_status = test_data_freshness()

