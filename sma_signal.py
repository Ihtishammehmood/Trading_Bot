from strategy_utils import MyBuySell
from datetime import datetime
import backtrader as bt
import yfinance as yf

aapl_df = yf.download("AAPL",
                      start="2022-01-01",
                      end=datetime.today(),
                      progress=False,
                      auto_adjust=True)

# Flatten the multi-level column index
aapl_df.columns = aapl_df.columns.get_level_values(0)

aapl_df.head()

# create the strategy using a signal
class SmaSignal(bt.Signal):
    params = (("period", 20), )

    def __init__(self):
        self.lines.signal = self.data - bt.ind.SMA(period=self.p.period)

# download data
data = bt.feeds.PandasData(dataname=aapl_df)

# create a Cerebro entity
cerebro = bt.Cerebro(stdstats = False)

# set up the backtest
cerebro.adddata(data)
cerebro.broker.setcash(100000.0)
cerebro.add_signal(bt.SIGNAL_LONG, SmaSignal)
cerebro.addobserver(MyBuySell)
cerebro.addobserver(bt.observers.Value)

# run backtest
print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")
x = cerebro.run()
print(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")

# plot results
cerebro.plot(iplot=False, volume=False)