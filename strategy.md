# MOVING AVERAGE CROSSOVER STRATEGY

**Timeframe**
Signal on H1

**Indicators**
* **EMA30** on H1 — **fast**
* **EMA70** on H1 — **slow**

**Entry**
* **LONG**: when EMA30 crosses above EMA70 on H1
* **SHORT**: opposite — EMA30 crosses below EMA70

**Monetary Risk** = Capital $\times$ % of risk $\div 100$
*Example: $10000 \times 3\% \div 100$

**Capital** = N
**Risk percentage** = 3%

**Stop Distance** (maximum distance allowed between entry and stop loss)
**Q** = (UNITS OR LOTS)

**STOP DISTANCE** = MONETARY RISK $\div$ Q
**STOP DISTANCE** = (N $\times$ 3\%) $\div$ (100 $\times$ Q)

**STOP LOSS** = ENTRY PRICE – STOP DISTANCE

**Position Management**
When the favorable move reaches **1:1** (price advances equal to the initial risk), move **SL to break-even**: **SL = entry\_price $\pm$ BE\_buffer** (small buffer to avoid being stopped out due to spread).
*Example: if long and entry=1.1000 and SL=1.0968 (risk = 32 pips), when price $\geq$ 1.1032, move SL to $1.1000 + BE\_buffer$.

Take partial **30% at Risk Reward 1:2**.
Let the remaining **70%** run until the moving averages cross in the opposite direction, then close the position.

**Step-by-step Rules:**
1. On H1, calculate EMA30, EMA70.
2. When EMA30 crosses EMA70, entry signal.
3. Open position with calculated SL.
4. Monitor, if profit $>= $ risk (1:1), move SL to BE.
5. Take 30% at 1:2.
6. Close remaining position when the MAs cross opposite.

**Explanation in Words:**
Check two moving averages (**EMA30** and **EMA70**). When the fast crosses the slow, enter long or short. Stop loss is calculated from the percentage of capital risked, converted into pip distance. When the trade reaches **1:1**, move **SL to break-even**, take **30% profit at 1:2**, and let the rest run until the next opposite crossover.

**Indicators and Timeframe:**
* **Timeframe**: H1
* **Indicators**: EMA30 (fast), EMA70 (slow)

**Entry Rules:**
* **LONG**: enter when EMA30 crosses above EMA70 on H1.