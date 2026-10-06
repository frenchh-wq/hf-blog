# Autonomous Trading Agent

A working build of the "OpenAI Dots + Jev + Opus 5.5 trading agent" prompt. It researches markets,
generates and backtests strategies, trades them (paper by default), manages risk, and re-evaluates
its own strategy on a schedule.

> **About the names in the original prompt:** "OpenAI Dots" and "Jev" aren't SDKs I could find,
> so each layer is built from real, installable pieces:
>
> | Prompt layer | What it is here |
> |---|---|
> | OpenAI Dots: orchestration, memory, tools | `agent/trader.py` loop + SQLite memory (`agent/storage.py`) |
> | Claude Opus 5.5: research, strategy generation, reasoning | `agent/llm.py` (Anthropic SDK, `claude-opus-5-5`, structured JSON output) |
> | Jev: execution, risk, position control | `agent/execution.py` (paper broker / ccxt live broker) + `agent/risk.py` |
> | Exchanges / data | `agent/data.py` via [ccxt](https://github.com/ccxt/ccxt): Binance, Bybit, Kraken, OKX, 100+ more |

**Read this first:** no backtest guarantees future returns, and most retail algo strategies lose
money after fees. The agent is built to be skeptical: it trades nothing it hasn't validated
out-of-sample, it stands aside when nothing validates, and it can't trade real money until you
explicitly unlock it.

## Quick start

```bash
cd trading-agent
pip install -r requirements.txt
cp .env.example .env          # add ANTHROPIC_API_KEY (optional) and settings

python main.py check          # final check: tests + full backtest + paper replay
python main.py backtest       # research + walk-forward backtest on live exchange data
python main.py paper          # run 24/7 against live prices with a paper broker
python main.py dashboard      # http://localhost:8050 (run alongside paper/live)
python main.py report         # write today's report to results/
```

Add `--synthetic` to any command to run offline on generated data, or `--no-claude` to skip the
LLM and use grid search only.

## How it works

```
 every new candle            every 168 bars (configurable)        daily
 ────────────────            ─────────────────────────────        ─────
 fetch OHLCV + price         grid search (168 configs)            report → results/
 stop / take-profit check    + Claude proposals (3-5 configs)     Claude reviews trades
 strategy signals            walk-forward: train 70% / test 30%   proposes fixes → re-validated
 risk gates → order          adopt only if it beats the incumbent
 equity snapshot → SQLite    out-of-sample; otherwise stand aside
```

**Modules** (`agent/`)

- `analysis.py` (market_analysis): regime (trending / ranging / volatile), ADX trend strength, ATR volatility percentile, RSI, swing support and resistance
- `strategies.py` + `backtest.py` (strategy_engine): EMA cross, RSI reversion, Donchian breakout, Bollinger reversion; an event-driven backtester with fees, slippage, intrabar stops, and the same risk sizing used live; walk-forward optimisation
- `llm.py`: Claude reads the market snapshot, leaderboard and recent performance, and returns schema-validated strategy proposals. Its suggestions are only adopted after they pass the same out-of-sample backtest as everything else.
- `risk.py` (risk_management): position sizing, loss limits, kill switch
- `execution.py` (trade_execution): `PaperBroker`, and `CcxtBroker` for live spot market orders
- `notify.py` + `dashboard.py` + `report.py` (monitoring): Telegram/Discord alerts, a live dashboard, daily JSON reports
- `trader.py` (self_improvement + orchestration): the 24/7 loop, scheduled re-research, and the daily Claude review

## Risk rules (from the prompt, enforced in both backtest and live)

| Rule | Default | Behaviour |
|---|---|---|
| Max position risk | 2% per trade | size = equity × 2% ÷ (entry − stop); never leveraged |
| Daily loss limit | 5% | no new entries for the rest of the UTC day |
| Max drawdown | 15% from peak | **kill switch**: flatten and stop |
| Min confidence | > 1.5 | each strategy reports a normalised signal strength; weaker signals are skipped |
| Kill switch | — | 5 consecutive API errors, a >15% single-candle move, a `KILL` file in this folder, or max drawdown |

Emergency stop: `touch trading-agent/KILL`. The agent closes its position and exits on the next tick.

## Going live

1. Run `python main.py check` and paper trade for a few weeks first.
2. Create an exchange API key with **trade-only permission (no withdrawals)**, ideally IP-restricted.
3. In `.env`, set `MODE=live`, `EXCHANGE_API_KEY`, `EXCHANGE_API_SECRET`, and
   `LIVE_CONFIRM=I understand this trades real money`.
4. `python main.py live`

Live mode trades **spot only, long only**. Futures and shorting are deliberately left out.

## Output

- `trading.db`: trades, equity curve, signals, logs, strategy history
- `results/backtest.json`, `results/backtest_equity.csv`, `results/report_YYYY-MM-DD.json`

## Tests

`python -m pytest -q tests` covers sizing, daily-loss and drawdown limits, the API-error and
abnormal-move kill switches, the `KILL` file, lookahead bias, the stand-aside behaviour, Claude
response parsing, and an end-to-end paper replay.
