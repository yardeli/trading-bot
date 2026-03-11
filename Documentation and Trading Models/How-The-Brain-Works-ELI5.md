# How the Trading Brain Works (ELI5)

## The Big Idea

Imagine you have 8 friends who each give you stock tips. Over time, you notice some friends are right more often than others. You start trusting the good ones more and ignoring the bad ones. That's basically what the brain does.

---

## The 8 Friends (Strategies)

Each "friend" looks at prices differently to decide if something is worth buying:

| Friend | What They Do | Plain English |
|--------|-------------|---------------|
| **RSI** | Watches if something is "too cheap" or "too expensive" relative to recent prices | "This dropped a lot recently, it'll probably bounce back" |
| **MA (Moving Average)** | Compares the short-term trend to the long-term trend | "The recent prices just crossed above the older average — it's trending up" |
| **Momentum** | Measures how fast the price is moving | "This thing is flying — jump on before it goes higher" |
| **MeanRev** | Measures how far price is from its average | "This is way below normal, it should snap back" |
| **VWAP** | Compares price to a volume-weighted average | "People are paying more for this than we'd have to — it's a deal" |
| **Composite** | Asks ALL the friends and averages their opinions | "Most of the group thinks this is a buy" |
| **Stochastic** | Similar to RSI but uses the high-low range | "Price is near the bottom of its recent range" |
| **Bollinger** | Checks if price is outside its normal bands | "Price just dropped below the lower boundary — that's unusual" |

---

## How the Brain Learns

### Step 1: Every friend starts equal

When you first open the app, all 8 strategies have a **weight of 1.0x** — the brain trusts them all equally.

### Step 2: Watch what happens

Every time a trade closes (sell), the brain records:
- **Which friend suggested the buy** (e.g., "RSI told me to buy this")
- **Did we make money?** (win or loss)
- **How much?** (percentage gain/loss)

### Step 3: Update trust scores

After each trade, the brain recalculates how much it trusts each friend using a simple formula:

```
trust = (how often they're right) mixed with (how much money they make)
```

It uses something called **exponential moving average** — which is a fancy way of saying "recent results matter more than old ones." The last ~12-15 trades have the most influence. Ancient history fades away.

- A friend who keeps winning gets a **higher weight** (up to 3.0x)
- A friend who keeps losing gets a **lower weight** (down to 0.2x)
- A friend with fewer than 5 trades stays at 1.0x (not enough data to judge)

### Step 4: Use the trust scores

Next time the brain is deciding what to buy, it takes each friend's suggestion and **multiplies the signal strength by their trust score**:

```
RSI says "buy BTC" with strength 0.5
RSI's trust score is 2.3x (it's been winning a lot)
Adjusted strength = 0.5 × 2.3 = 1.15 (strong buy!)

MeanRev says "buy ETH" with strength 0.6
MeanRev's trust score is 0.4x (it's been losing)
Adjusted strength = 0.6 × 0.4 = 0.24 (weak, probably skip)
```

The bot then buys in order of adjusted strength — best signals first.

---

## The Other Smart Stuff

### Regime Detection: "What kind of market is this?"

Every few seconds, the brain looks at ALL the assets and asks: "Is this a trending market or a choppy market?"

- **Trending** = prices keep moving in the same direction (up or down)
- **Choppy** = prices bounce around randomly with no clear direction

Why does this matter? Because different friends work better in different markets:

| Market | Winners | Losers |
|--------|---------|--------|
| Trending | MA, Momentum | MeanRev |
| Choppy | MeanRev, RSI | MA, Momentum |

So the brain gives bonus points to the right friends for the current market type.

### Streak Tracking: "Are we hot or cold?"

The brain tracks consecutive wins and losses:

- **On a winning streak (3+):** The brain gets slightly more aggressive — wider take-profit targets, easier entry requirements
- **On a losing streak (3+):** The brain gets slightly more cautious — tighter stops, higher bar for new trades
- **Between streaks:** Everything gradually returns to normal

This is gentle — the circuit breaker handles real emergencies.

### GARCH Volatility: "How wild are things right now?"

Instead of just looking at how much prices moved recently (standard deviation), the brain uses a formula called GARCH that's better at detecting when volatility is *increasing*:

```
new_volatility = a tiny base amount
               + 10% of (latest price move squared)
               + 85% of (previous volatility estimate)
```

This means: if there's suddenly a huge price swing, the brain quickly raises its volatility estimate. If things calm down, it slowly lowers it. This helps with:
- **Sizing positions** (smaller when things are wild)
- **Setting stops** (wider when volatility is high so you don't get stopped out by noise)

### Kelly Criterion: "How much should I bet?"

Once a strategy has 30+ trades of history, the brain can calculate the mathematically optimal bet size:

```
Kelly % = win_rate - (1 - win_rate) / (avg_win / avg_loss)
```

Example: If RSI wins 60% of the time and wins are 1.5x bigger than losses:
```
Kelly = 0.60 - 0.40 / 1.5 = 0.60 - 0.27 = 0.33 (bet 33% of portfolio)
```

The brain uses **half-Kelly** (so 16.5% in this example) because the full Kelly is too aggressive — one bad streak could wipe you out.

### ATR Stops: "How far should my safety net be?"

Instead of fixed stop-losses (like "sell if it drops 2%"), the brain measures how much each asset *normally* moves (called ATR — Average True Range) and sets stops based on that:

- **Stop-loss** = 2.5 × normal movement below entry
- **Take-profit** = 7.5 × normal movement above entry (3:1 reward-to-risk)
- **Trailing stop** = 2 × normal movement below the highest price reached

This means: a volatile asset like BTC gets wider stops (so normal swings don't trigger a sell), while a stable asset like GLD gets tighter stops (because a 2% move actually means something).

### Circuit Breaker: "Pull the emergency brake"

If the portfolio drops too much from its peak:

| Drawdown | What Happens |
|----------|-------------|
| -10% | Warning: slightly tighter stops, slightly smaller positions |
| -15% | De-risk: noticeably smaller positions |
| -20% | Reduce: positions cut in half |
| -25% | HALT: no new trades until recovery |

This prevents catastrophic losses. It's the last line of defense.

---

## Where Does the Data Live?

All of the brain's learning is saved to your browser's **localStorage** (like a cookie but bigger). This means:

- **Closing the tab doesn't lose anything** — it auto-saves after every trade
- **Opening the page again restores everything** — win rates, weights, streaks, regime history
- **You can export/import** the brain data as a JSON file (backup or transfer between computers)
- **Reset** wipes the brain clean if you want a fresh start

The storage key is `czwm_brain_v1` and it's about 2KB — tiny.

---

## The Learning Loop (TL;DR)

```
1. Price comes in
2. Each strategy looks at it and says "buy" or "not now"
3. Brain multiplies each suggestion by the strategy's trust score
4. Brain checks the market regime and adjusts further
5. Best signals get executed (if circuit breaker allows)
6. Position gets ATR-based stop-loss and take-profit
7. When position closes → brain records win/loss
8. Trust scores update → better strategies get more influence
9. Repeat forever, getting smarter each cycle
```

The longer it runs, the more trades it sees, the better it knows which strategies work in which conditions. It's not magic — it's just paying attention and remembering what worked.
