import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta  # For easy EMA calculation
from datetime import datetime
import numpy as np

# --- 1. CONFIGURATION ---
# Strategy Parameters
SYMBOL = "USDJPY"
TIMEFRAME = mt5.TIMEFRAME_H1
EMA_FAST_PERIOD = 30
EMA_SLOW_PERIOD = 70
ENTRY_MARGIN_PIPS = 1.0
EXIT_MARGIN_PIPS = 5.0
EMERGENCY_SL_PERCENT = 0.03  # 3%

# Backtest Parameters
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2025, 10, 19)  # 1 year of data
INITIAL_BALANCE = 10000.0
LOT_SIZE = 0.1  # Fixed lot size for this example

# --- 2. HELPER FUNCTIONS ---

def connect_to_mt5():
    """Initializes and checks the connection to MetaTrader 5."""
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        return False
    print("MetaTrader 5 Initialized Successfully")
    return True

def get_symbol_info(symbol):
    """Fetches and validates symbol information."""
    info = mt5.symbol_info(symbol)
    if info is None:
        print(f"Failed to get symbol info for {symbol}")
        return None

    if not info.visible:
        print(f"Symbol {symbol} is not visible, trying to select...")
        if not mt5.symbol_select(symbol, True):
            print(f"symbol_select({symbol}) failed, error code =", mt5.last_error())
            return None
    return info

def get_pip_value(info):
    """Calculates the price value of a single pip."""
    # A 'pip' is 0.0001 for 5-digit pairs (EURUSD) and 0.01 for 3-digit pairs (USDJPY)
    if info.digits in (3, 5):
        return 10 * info.point
    else:
        return info.point

def get_historical_data(symbol, timeframe, start, end):
    """Fetches OHLC data from MT5 and returns a pandas DataFrame."""
    rates = mt5.copy_rates_range(symbol, timeframe, start, end)
    if rates is None or len(rates) == 0:
        print(f"No data found for {symbol} in the specified range.")
        return pd.DataFrame()
    
    data = pd.DataFrame(rates)
    data['time'] = pd.to_datetime(data['time'], unit='s')
    data.set_index('time', inplace=True)
    return data

def calculate_indicators(data, fast, slow):
    """Adds EMA and helper columns to the DataFrame."""
    data['EMA_30'] = ta.ema(data['close'], length=fast)
    data['EMA_70'] = ta.ema(data['close'], length=slow)
    
    # Create columns for the 'higher' and 'lower' of the two EMAs
    data['max_ema'] = data[['EMA_30', 'EMA_70']].max(axis=1)
    data['min_ema'] = data[['EMA_30', 'EMA_70']].min(axis=1)
    
    data.dropna(inplace=True)  # Remove rows with NaN (from EMA calculation)
    return data

# --- 3. BACKTESTING ENGINE ---

def run_backtest(data, info, pip_value):
    """Runs the main event-driven backtest loop."""
    print(f"\nRunning backtest on {SYMBOL} from {START_DATE.date()} to {END_DATE.date()}...")
    
    balance = INITIAL_BALANCE
    lot_size = LOT_SIZE
    contract_size = info.trade_contract_size
    
    # Convert pip margins to price margins
    entry_margin_price = ENTRY_MARGIN_PIPS * pip_value
    exit_margin_price = EXIT_MARGIN_PIPS * pip_value
    
    trades = []
    in_position = False  # Can be False, 'LONG', or 'SHORT'
    entry_price = 0.0
    trade_open_balance = 0.0  # Balance at the time of opening the trade

    # Iterate through each candle (bar)
    for i in range(1, len(data)):
        # --- Get current and previous candle data ---
        curr = data.iloc[i]
        prev = data.iloc[i-1]
        
        # Current prices
        price = curr['close']
        max_ema = curr['max_ema']
        min_ema = curr['min_ema']
        
        # Previous prices
        prev_price = prev['close']
        prev_max_ema = prev['max_ema']
        prev_min_ema = prev['min_ema']

        # --- 1. CHECK FOR EXITS (if in a position) ---
        if in_position:
            pnl = 0.0
            is_emergency_sl = False
            is_strategy_exit = False
            
            if in_position == 'LONG':
                pnl = (price - entry_price) * lot_size * contract_size
                is_emergency_sl = (pnl < 0) and (abs(pnl) > (trade_open_balance * EMERGENCY_SL_PERCENT))
                is_strategy_exit = (price < min_ema - exit_margin_price)
                exit_reason = 'Emergency SL' if is_emergency_sl else 'Strategy Exit'

            elif in_position == 'SHORT':
                pnl = (entry_price - price) * lot_size * contract_size
                is_emergency_sl = (pnl < 0) and (abs(pnl) > (trade_open_balance * EMERGENCY_SL_PERCENT))
                is_strategy_exit = (price > max_ema + exit_margin_price)
                exit_reason = 'Emergency SL' if is_emergency_sl else 'Strategy Exit'

            # Close position if any exit condition is met
            if is_emergency_sl or is_strategy_exit:
                balance += pnl
                trades.append({
                    'type': in_position,
                    'exit_time': curr.name,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'pnl': pnl,
                    'balance': balance,
                    'exit_reason': exit_reason
                })
                in_position = False
                entry_price = 0.0

        # --- 2. CHECK FOR ENTRIES (if not in a position) ---
        if not in_position:
            # LONG Entry Condition:
            # 1. Price was at or below the EMAs
            # 2. Price is now above the highest EMA + margin
            if (prev_price <= prev_max_ema) and (price > max_ema + entry_margin_price):
                in_position = 'LONG'
                entry_price = price
                trade_open_balance = balance  # Record balance for 3% SL rule
                # print(f"{curr.name} - OPEN LONG @ {price}") # Uncomment for verbose logging

            # SHORT Entry Condition:
            # 1. Price was at or above the EMAs
            # 2. Price is now below the lowest EMA - margin
            elif (prev_price >= prev_min_ema) and (price < min_ema - entry_margin_price):
                in_position = 'SHORT'
                entry_price = price
                trade_open_balance = balance  # Record balance for 3% SL rule
                # print(f"{curr.name} - OPEN SHORT @ {price}") # Uncomment for verbose logging

    return pd.DataFrame(trades), balance

# --- 4. RESULTS ANALYSIS ---

def analyze_results(trades_df, final_balance):
    """Prints a summary of the backtest performance."""
    if trades_df.empty:
        print("No trades were executed.")
        return

    print("\n--- Backtest Results ---")
    print(f"Initial Balance: ${INITIAL_BALANCE:,.2f}")
    print(f"Final Balance:   ${final_balance:,.2f}")
    
    total_pnl = trades_df['pnl'].sum()
    print(f"Total Net P/L:   ${total_pnl:,.2f}")
    
    total_trades = len(trades_df)
    print(f"Total Trades:    {total_trades}")
    
    wins = trades_df[trades_df['pnl'] > 0]
    losses = trades_df[trades_df['pnl'] <= 0]
    
    win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
    print(f"Win Rate:        {win_rate:.2f}%")
    
    avg_win = wins['pnl'].mean()
    avg_loss = losses['pnl'].mean()
    print(f"Average Win:     ${avg_win:,.2f}")
    print(f"Average Loss:    ${avg_loss:,.2f}")
    
    profit_factor = abs(wins['pnl'].sum() / losses['pnl'].sum()) if losses['pnl'].sum() != 0 else np.inf
    print(f"Profit Factor:   {profit_factor:.2f}")

    sl_exits = trades_df[trades_df['exit_reason'] == 'Emergency SL']
    print(f"Emergency SL Hits: {len(sl_exits)}")
    
    print("\n--- Trades List ---")
    # Display max 20 trades for brevity
    print(trades_df.to_string(index=False, max_rows=20))

# --- 5. EXECUTION ---

def main():
    if not connect_to_mt5():
        return

    symbol_info = get_symbol_info(SYMBOL)
    if symbol_info is None:
        mt5.shutdown()
        return
    
    pip = get_pip_value(symbol_info)
    print(f"Symbol: {SYMBOL}, Pip Value: {pip}, Contract Size: {symbol_info.trade_contract_size}")
    
    data = get_historical_data(SYMBOL, TIMEFRAME, START_DATE, END_DATE)
    if data.empty:
        mt5.shutdown()
        return

    data_with_indicators = calculate_indicators(data, EMA_FAST_PERIOD, EMA_SLOW_PERIOD)
    
    trades_df, final_balance = run_backtest(data_with_indicators, symbol_info, pip)
    
    analyze_results(trades_df, final_balance)
    
    mt5.shutdown()
    print("\nMetaTrader 5 Shutdown")

if __name__ == "__main__":
    main()