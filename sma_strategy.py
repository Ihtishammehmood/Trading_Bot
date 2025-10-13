from datetime import datetime
import backtrader as bt
import yfinance as yf
from strategy_utils import MyBuySell, get_action_log_string, get_result_log_string



# Create a Strategy
class SmaStrategy(bt.Strategy):
    params = (("ma_period", 20), )

    def __init__(self):
        # keep track of close price in the series
        self.data_close = self.datas[0].close

        # keep track of pending orders
        self.order = None

        # add a simple moving average indicator
        self.sma = bt.ind.SMA(self.datas[0],
                              period=self.params.ma_period)

    def log(self, txt):
        dt = self.datas[0].datetime.date(0).isoformat()
        print(f"{dt}: {txt}")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # order already submitted/accepted - no action required
            return

        # report executed order
        if order.status in [order.Completed]:

            direction = "b" if order.isbuy() else "s"
            log_str = get_action_log_string(
                    dir=direction,
                    action="e",
                    price=order.executed.price,
                    size=order.executed.size,
                    cost=order.executed.value,
                    commission=order.executed.comm
                )
            self.log(log_str)

        # report failed order
        elif order.status in [order.Canceled, order.Margin,
                              order.Rejected]:
            self.log("Order Failed")

        # reset order -> no pending order
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log(get_result_log_string(gross=trade.pnl, net=trade.pnlcomm))

    def next(self):
        # do nothing if an order is pending
        if self.order:
            return

        # check if there is already a position
        if not self.position:
            # buy condition
            if self.data_close[0] > self.sma[0]:
                self.log(get_action_log_string("b", "c", self.data_close[0], 1))
                self.order = self.buy()
        else:
            # sell condition
            if self.data_close[0] < self.sma[0]:
                self.log(get_action_log_string("s", "c", self.data_close[0], 1))
                self.order = self.sell()

    def start(self):
        print(f"Initial Portfolio Value: {self.broker.get_value():.2f}")

    def stop(self):
        print(f"Final Portfolio Value: {self.broker.get_value():.2f}")

# Download AAPL data
df = yf.download("ORCL", start="2022-01-01", end=datetime.today(), auto_adjust=False)

# Ensure proper column names for Backtrader
df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']

# create a Cerebro entity
cerebro = bt.Cerebro(stdstats = False)

# set up the backtest
data = bt.feeds.PandasData(dataname=df)
cerebro.adddata(data)
cerebro.broker.setcash(100000.0)
cerebro.addstrategy(SmaStrategy)
cerebro.addobserver(MyBuySell)
cerebro.addobserver(bt.observers.Value)

# run backtest
cerebro.run()

cerebro.plot(iplot=False, volume=False)
