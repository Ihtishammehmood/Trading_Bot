"""
EMA30 / EMA70 H1 crossover backtester using MetaTrader5 Python API.

How to run:
1. pip install MetaTrader5 pandas numpy
2. Make sure MT5 desktop is running and you are logged into the demo account.
3. python ema_crossover_backtest_mt5.py

Change parameters below: SYMBOL, CAPITAL_N, RISK_PCT, LOTS (Q), START/END dates.
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np

# -------------------------
# USER PARAMETERS (edit)
# -------------------------
SYMBOL = "GBPUSD"  # change to desired symbol
TIMEFRAME = mt5.TIMEFRAME_D1
BARS = 3500  # number of H1 bars to fetch (increase for longer tests)
CAPITAL_N = 10000.0  # N (account capital)
RISK_PCT = 0.03  # 3% risk
LOTS = 0.1  # Q = lots (example 0.1 lots = mini lot)
BE_BUFFER_PIPS = 1.5  # buffer in pips when moving SL to break-even
TAKE_PARTIAL_PCT = 0.30  # take 30% at RR 1:2
PRINT_PROGRESS = True
OUTFILE = "ema_backtest_trades.csv"
# -------------------------


def connect_mt5():
    if not mt5.initialize():
        raise RuntimeError(
            "Failed to initialize MT5, is terminal running and logged in?"
        )


def shutdown_mt5():
    mt5.shutdown()


def fetch_rates(symbol, timeframe, nbars):
    # fetch most recent nbars
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, nbars)
    if rates is None or len(rates) == 0:
        raise RuntimeError(
            f"No data for {symbol}. Check symbol name and that it's available."
        )
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    return df


def compute_adx(df, period=14):
    """Compute Average Directional Index (ADX) to identify trending/ranging markets"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Calculate True Range (TR)
    df['prev_close'] = close.shift(1)
    df['tr1'] = high - low
    df['tr2'] = abs(high - df['prev_close'])
    df['tr3'] = abs(low - df['prev_close'])
    df['TR'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Calculate +DM and -DM
    df['up_move'] = high - high.shift(1)
    df['down_move'] = low.shift(1) - low
    
    df['+DM'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['-DM'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    
    # Calculate smoothed TR and DM
    df['TR' + str(period)] = df['TR'].rolling(window=period).sum()
    df['+DM' + str(period)] = df['+DM'].rolling(window=period).sum()
    df['-DM' + str(period)] = df['-DM'].rolling(window=period).sum()
    
    # Calculate +DI and -DI
    df['+DI' + str(period)] = 100 * df['+DM' + str(period)] / df['TR' + str(period)]
    df['-DI' + str(period)] = 100 * df['-DM' + str(period)] / df['TR' + str(period)]
    
    # Calculate DX and ADX with handling division by zero
    denominator = df['+DI' + str(period)] + df['-DI' + str(period)]
    df['DX'] = np.where(denominator != 0,
                        100 * abs(df['+DI' + str(period)] - df['-DI' + str(period)]) / denominator,
                        0)  # Default to 0 when denominator is 0
    df['ADX'] = df['DX'].rolling(window=period).mean()
    df['ADX'] = df['ADX'].fillna(0)  # Replace NaN values with 0
    
    # Clean up intermediate columns
    cols_to_drop = ['prev_close', 'tr1', 'tr2', 'tr3', 'TR', 'up_move', 'down_move', 
                    '+DM', '-DM', 'TR' + str(period), '+DM' + str(period), '-DM' + str(period),
                    '+DI' + str(period), '-DI' + str(period), 'DX']
    df.drop(columns=cols_to_drop, inplace=True)
    
    return df

def compute_emas(df, fast=20, slow=50):
    df["EMA_fast"] = df["close"].ewm(span=fast, adjust=False).mean()
    df["EMA_slow"] = df["close"].ewm(span=slow, adjust=False).mean()
    return df


def symbol_info_params(symbol):
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"Symbol info not found for {symbol}")
    point = info.point  # minimal price change
    contract_size = (
        getattr(info, "trade_contract_size", None)
        or getattr(info, "contract_size", None)
        or 100000
    )
    # try to get tick value/size if available
    tick_value = getattr(info, "trade_tick_value", None) or getattr(
        info, "tick_value", None
    )
    tick_size = (
        getattr(info, "trade_tick_size", None)
        or getattr(info, "tick_size", None)
        or point
    )
    return {
        "point": point,
        "contract_size": contract_size,
        "tick_value": tick_value,
        "tick_size": tick_size,
        "lot_min": info.volume_min if hasattr(info, "volume_min") else None,
        "lot_step": info.volume_step if hasattr(info, "volume_step") else None,
    }


def pip_to_price(pips, point, pip_multiple=10):
    """
    Convert number of pips to price units.
    For most forex pairs, 1 pip = 10 * point (if point is 0.00001 for 5-digit brokers).
    """
    return pips * point * pip_multiple


def compute_stop_distance_price(capital, risk_pct, lots, info, price):
    """
    Compute stop distance according to strategy formula:
    STOP DISTANCE = (N × risk_pct) ÷ (100 × Q)
    where N = capital, Q = lots
    """
    point = info["point"]

    # Strategy formula: stop_distance = (N × risk_pct) ÷ (100 × Q)
    monetary_risk = capital * risk_pct  # N × risk_pct
    pips_affordable = (monetary_risk) / (100 * lots)  # divide by (100 × Q)

    # Convert pips to price units (1 pip = 10 points in 5-digit broker)
    price_distance = pips_affordable * point * 10
    return price_distance, pips_affordable


def bar_within(bar, low_target, high_target):
    """Check if bar low..high crosses target(s). Return True if within range"""
    return bar["low"] <= high_target and bar["high"] >= low_target


def simulate(
    df, symbol, capital, risk_pct, lots, be_buffer_pips=1.5, take_partial_pct=0.3
):
    info = symbol_info_params(symbol)
    point = info["point"]
    be_buffer_price = pip_to_price(be_buffer_pips, point)
    monetary_risk = capital * risk_pct

    trades = []
    position = None  # None or dict with keys: side, entry_idx, entry_price, sl_price, lots_total, lots_remaining, pips_risk
    equity = capital
    equity_curve = []
    peak_equity = equity
    max_drawdown = 0.0

    df = df.copy()
    for i in range(1, len(df) - 1):
        bar = df.iloc[i]
        next_bar = df.iloc[i + 1]  # we often enter on next bar open
        # detect crossovers using current and previous EMA difference
        prev = df.iloc[i - 1]
        ema_prev_diff = prev["EMA_fast"] - prev["EMA_slow"]
        ema_curr_diff = bar["EMA_fast"] - bar["EMA_slow"]

        # CROSS UP
        cross_up = (ema_prev_diff <= 0) and (ema_curr_diff > 0)
        # CROSS DOWN
        cross_down = (ema_prev_diff >= 0) and (ema_curr_diff < 0)

        # If no position and a signal occurs, open at next_bar.open
        if position is None and cross_up:
            entry_price = next_bar["open"]
            # compute stop distance in price units using strategy formula
            price_distance, pips_risk = compute_stop_distance_price(
                capital, risk_pct, lots, info, entry_price
            )
            sl_price = entry_price - price_distance
            # record position
            position = {
                "side": "long",
                "entry_idx": i + 1,
                "entry_time": next_bar.name,
                "entry_price": entry_price,
                "sl_price": sl_price,
                "lots_total": lots,
                "lots_remaining": lots,
                "pips_risk": pips_risk,
                "price_distance": price_distance,
                "monetary_risk": monetary_risk,
                "tp1_price": entry_price + 2 * price_distance,  # RR 1:2
                "partial_taken": False,
                "sl_moved_to_be": False,
            }
            if PRINT_PROGRESS:
                print(
                    f"OPEN LONG @ {entry_price:.5f} sl {sl_price:.5f} pips_risk {pips_risk:.1f}"
                )
            # move to next bar processing
            continue

        if position is None and cross_down:
            entry_price = next_bar["open"]
            # compute stop distance using strategy formula
            price_distance, pips_risk = compute_stop_distance_price(
                capital, risk_pct, lots, info, entry_price
            )
            sl_price = entry_price + price_distance
            position = {
                "side": "short",
                "entry_idx": i + 1,
                "entry_time": next_bar.name,
                "entry_price": entry_price,
                "sl_price": sl_price,
                "lots_total": lots,
                "lots_remaining": lots,
                "pips_risk": pips_risk,
                "price_distance": price_distance,
                "monetary_risk": monetary_risk,
                "tp1_price": entry_price - 2 * price_distance,  # RR 1:2
                "partial_taken": False,
                "sl_moved_to_be": False,
            }
            if PRINT_PROGRESS:
                print(
                    f"OPEN SHORT @ {entry_price:.5f} sl {sl_price:.5f} pips_risk {pips_risk:.1f}"
                )
            continue

        # If position exists, manage it using current bar's high/low
        if position is not None:
            # helper vars
            side = position["side"]
            entry = position["entry_price"]
            tp1 = position["tp1_price"]

            # compute current unrealized profit in pips using close price
            close_price = bar["close"]
            if side == "long":
                unreal_pips = (close_price - entry) / (point * 10)  # in pips
            else:
                unreal_pips = (entry - close_price) / (point * 10)

            # Move SL to break-even if profit in pips >= initial risk in pips (1:1)
            if (not position["sl_moved_to_be"]) and abs(unreal_pips) >= position[
                "pips_risk"
            ]:
                # Move SL to entry ± buffer (small buffer to avoid being stopped due to spread)
                if side == "long":
                    new_sl = entry + be_buffer_price
                else:
                    new_sl = entry - be_buffer_price
                position["sl_price"] = new_sl
                position["sl_moved_to_be"] = True
                if PRINT_PROGRESS:
                    print(
                        f"Move SL to BE for {side.upper()} at {new_sl:.5f} (time {bar.name})"
                    )

            # Position Management according to strategy:
            # 1. Take partial profit (30%) at RR 1:2
            # 2. Let remaining 70% run until opposite MA crossover
            bar_low = bar["low"]
            bar_high = bar["high"]
            closed_this_bar = False

            # Take 30% profit at RR 1:2 (tp1)
            if (not position["partial_taken"]) and (
                (side == "long" and bar_high >= tp1)
                or (side == "short" and bar_low <= tp1)
            ):
                # Strategy: take exactly 30% at RR 1:2
                take_lots = round(position["lots_total"] * take_partial_pct, 6)
                position["lots_remaining"] = round(
                    position["lots_remaining"] - take_lots, 6
                )
                position["partial_taken"] = True
                # Calculate profit for the partial
                if side == "long":
                    profit_pips = (tp1 - entry) / (point * 10)
                else:
                    profit_pips = (entry - tp1) / (point * 10)
                profit_value = profit_pips * 10 * take_lots
                equity += profit_value
                if PRINT_PROGRESS:
                    print(
                        f"Take 30% ({take_lots} lots) at RR 1:2 TP1 {tp1:.5f} profit {profit_value:.2f}"
                    )
                # Strategy: Continue managing remaining 70% until MA crossover
            # SL hit?
            if (side == "long" and bar_low <= position["sl_price"]) or (
                side == "short" and bar_high >= position["sl_price"]
            ):
                # SL hit: close remaining lots
                exit_price = position["sl_price"]
                if side == "long":
                    profit_pips = (exit_price - entry) / (point * 10)
                else:
                    profit_pips = (entry - exit_price) / (point * 10)
                profit_value = profit_pips * 10 * position["lots_remaining"]
                equity += profit_value
                closed_trade = {
                    "symbol": symbol,
                    "side": side,
                    "entry_time": position["entry_time"],
                    "entry_price": entry,
                    "exit_time": bar.name,
                    "exit_price": exit_price,
                    "lots": position["lots_remaining"],
                    "profit": profit_value,
                }
                trades.append(closed_trade)
                if PRINT_PROGRESS:
                    print(
                        f"SL HIT {side.upper()} exit {exit_price:.5f} profit {profit_value:.2f}"
                    )
                position = None
                closed_this_bar = True

            # Strategy: Close remaining position (70%) when MAs cross in opposite direction
            if (
                (not closed_this_bar)
                and position is not None
                and ((side == "long" and cross_down) or (side == "short" and cross_up))
            ):
                # Close remaining position at next bar's open price
                exit_price = next_bar["open"]
                if side == "long":
                    profit_pips = (exit_price - entry) / (point * 10)
                else:
                    profit_pips = (entry - exit_price) / (point * 10)
                profit_value = profit_pips * 10 * position["lots_remaining"]
                equity += profit_value
                closed_trade = {
                    "symbol": symbol,
                    "side": side,
                    "entry_time": position["entry_time"],
                    "entry_price": entry,
                    "exit_time": next_bar.name,
                    "exit_price": exit_price,
                    "lots": position["lots_remaining"],
                    "profit": profit_value,
                    "exit_reason": "opposite_ma_cross",  # add reason for analysis
                }
                trades.append(closed_trade)
                if PRINT_PROGRESS:
                    print(
                        f"Strategy: Close remaining {position['lots_remaining']} lots (70%) on opposite MA cross at {exit_price:.5f} profit {profit_value:.2f}"
                    )
                position = None

        # record equity curve per bar
        equity_curve.append({"time": bar.name, "equity": equity})
        if equity > peak_equity:
            peak_equity = equity
        drawdown = peak_equity - equity
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    # If a position still open at end, close at last close
    if position is not None:
        last = df.iloc[-1]
        exit_price = last["close"]
        side = position["side"]
        entry = position["entry_price"]
        if side == "long":
            profit_pips = (exit_price - entry) / (point * 10)
        else:
            profit_pips = (entry - exit_price) / (point * 10)
        profit_value = profit_pips * 10 * position["lots_remaining"]
        equity += profit_value
        trades.append(
            {
                "symbol": symbol,
                "side": side,
                "entry_time": position["entry_time"],
                "entry_price": entry,
                "exit_time": last.name,
                "exit_price": exit_price,
                "lots": position["lots_remaining"],
                "profit": profit_value,
            }
        )
        if PRINT_PROGRESS:
            print(f"Closed remaining at end {exit_price:.5f} profit {profit_value:.2f}")
        position = None

    # Summaries
    trades_df = pd.DataFrame(trades)
    total_trades = len(trades_df)
    net_profit = trades_df["profit"].sum() if total_trades > 0 else 0.0
    wins = trades_df[trades_df["profit"] > 0]
    win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
    avg_win = wins["profit"].mean() if len(wins) > 0 else 0.0
    avg_loss = (
        trades_df[trades_df["profit"] < 0]["profit"].mean()
        if len(trades_df[trades_df["profit"] < 0]) > 0
        else 0.0
    )

    results = {
        "trades_df": trades_df,
        "total_trades": total_trades,
        "net_profit": net_profit,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "final_equity": equity,
        "max_drawdown": max_drawdown,
        "equity_curve": pd.DataFrame(equity_curve),
    }
    return results


def save_trades(trades_df, outfile):
    if trades_df is None or trades_df.empty:
        print("No trades to save.")
        return
    trades_df.to_csv(outfile, index=False)
    print(f"Saved trades to {outfile}")


def validate_inputs():
    """Validate strategy inputs match requirements"""
    if CAPITAL_N <= 0:
        raise ValueError("Capital (N) must be positive")
    if RISK_PCT <= 0 or RISK_PCT > 0.05:  # max 5% risk
        raise ValueError("Risk percentage must be between 0% and 5%")
    if LOTS <= 0:
        raise ValueError("Lots (Q) must be positive")
    if TAKE_PARTIAL_PCT != 0.30:
        raise ValueError("Take partial percentage must be exactly 30% (0.30)")
    if BE_BUFFER_PIPS < 0:
        raise ValueError("Break-even buffer must be non-negative")
    if BARS < 100:  # Minimum bars needed for reliable EMAs and ADX
        raise ValueError("Minimum 100 bars required for reliable signals")
        
def validate_data(df):
    """Validate price data quality"""
    if df.empty:
        raise ValueError("No data available")
        
    # Check for gaps
    time_diff = df.index.to_series().diff()
    expected_diff = pd.Timedelta(minutes=30)  # for M30 timeframe
    if (time_diff > expected_diff * 2).any():
        print("Warning: Data contains gaps larger than expected timeframe")
        
    # Check for abnormal price movements
    price_change_pct = abs(df['close'].pct_change())
    if (price_change_pct > 0.05).any():  # 5% price change
        print("Warning: Detected abnormal price movements (>5% change)")


def analyze_market_conditions(trades_df, df):
    """Analyze strategy performance in different market conditions"""
    if trades_df.empty:
        return pd.DataFrame()  # Return empty DataFrame if no trades
        
    try:
        trades_df = trades_df.copy()
        
        # Add ADX values at entry for each trade
        trades_df['entry_adx'] = trades_df['entry_time'].map(df['ADX'])
        trades_df['entry_adx'] = trades_df['entry_adx'].fillna(0)  # Handle any missing ADX values
        
        # Classify market conditions
        trades_df['market_condition'] = 'ranging'  # default
        trades_df.loc[trades_df['entry_adx'] > 25, 'market_condition'] = 'trending'
        
        # Analysis by market condition
        conditions = trades_df.groupby('market_condition').agg({
            'profit': ['count', 'sum', 'mean'],
            'side': 'count'
        }).round(2)
        
        conditions.columns = ['trades', 'total_profit', 'avg_profit', 'total_trades']
        conditions['win_rate'] = trades_df[trades_df['profit'] > 0].groupby('market_condition').size() / conditions['trades']
        
        return conditions
    except Exception as e:
        print(f"Warning: Could not analyze market conditions: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on error
    
    # Analysis by market condition
    conditions = trades_df.groupby('market_condition').agg({
        'profit': ['count', 'sum', 'mean'],
        'side': 'count'
    }).round(2)
    
    conditions.columns = ['trades', 'total_profit', 'avg_profit', 'total_trades']
    conditions['win_rate'] = trades_df[trades_df['profit'] > 0].groupby('market_condition').size() / conditions['trades']
    
    return conditions

def main():
    connect_mt5()
    try:
        # Validate inputs match strategy requirements
        validate_inputs()

        df = fetch_rates(SYMBOL, TIMEFRAME, BARS)
        validate_data(df)  # Validate data quality
        
        df = compute_emas(df, fast=20, slow=50)
        df = compute_adx(df, period=14)  # Add ADX calculation
        res = simulate(
            df,
            SYMBOL,
            CAPITAL_N,
            RISK_PCT,
            LOTS,
            be_buffer_pips=BE_BUFFER_PIPS,
            take_partial_pct=TAKE_PARTIAL_PCT,
        )
        trades_df = res["trades_df"]
        save_trades(trades_df, OUTFILE)

        # Print summary
        print("\n=== BACKTEST SUMMARY ===")
        print(f"Symbol: {SYMBOL}")
        print(f"Capital start: {CAPITAL_N:.2f}")
        print(
            f"Risk per trade: {RISK_PCT * 100:.2f}% => {CAPITAL_N * RISK_PCT:.2f} currency units"
        )
        print(f"Total trades: {res['total_trades']}")
        print(f"Net profit: {res['net_profit']:.2f}")
        print(f"Win rate: {res['win_rate'] * 100:.2f}%")
        print(f"Avg win: {res['avg_win']:.2f}, Avg loss: {res['avg_loss']:.2f}")
        print(
            f"Final equity: {res['final_equity']:.2f}, Max drawdown: {res['max_drawdown']:.2f}"
        )

        # Analyze market conditions
        if not trades_df.empty:
            print("\n=== MARKET CONDITIONS ANALYSIS ===")
            conditions = analyze_market_conditions(trades_df, df)
            
            # Only proceed with market condition analysis if we have valid conditions
            if not conditions.empty:
                print("\nPerformance by Market Condition:")
                print(conditions.round(4))
                
                # Get the trades with market conditions
                trades_with_conditions = trades_df.copy()
                trades_with_conditions['entry_adx'] = trades_with_conditions['entry_time'].map(df['ADX'])
                trades_with_conditions['market_condition'] = 'ranging'
                trades_with_conditions.loc[trades_with_conditions['entry_adx'] > 25, 'market_condition'] = 'trending'
                
                # Additional insights
                trend_trades = trades_with_conditions[trades_with_conditions['market_condition'] == 'trending']
                range_trades = trades_with_conditions[trades_with_conditions['market_condition'] == 'ranging']
            
            print("\nTrending Market Stats:")
            if not trend_trades.empty:
                print(f"Average profit: {trend_trades['profit'].mean():.2f}")
                print(f"Largest win: {trend_trades['profit'].max():.2f}")
                print(f"Largest loss: {trend_trades['profit'].min():.2f}")
            
            print("\nRanging Market Stats:")
            if not range_trades.empty:
                print(f"Average profit: {range_trades['profit'].mean():.2f}")
                print(f"Largest win: {range_trades['profit'].max():.2f}")
                print(f"Largest loss: {range_trades['profit'].min():.2f}")

        # show top 5 trades
        if not trades_df.empty:
            print("\nTop 5 trades by profit:")
            print(
                trades_df.sort_values("profit", ascending=False)
                .head(5)
                .to_string(index=False)
            )
    finally:
        shutdown_mt5()


if __name__ == "__main__":
    main()