# Grok Trader

Automated trading system skeleton that integrates MetaTrader5 with xAI's Grok 4
model for idea generation, risk management, and execution. The project is
designed for Pepperstone demo accounts and should only be used for paper trading
until thoroughly validated.

## Features

- Periodic polling of market data (default EURUSD & USDJPY, hourly bars)
- Grok-powered screening workflow with chart-based trade level extraction
- Risk-aware position sizing (2% balance risk cap) and drawdown guard
- Automated order placement through MetaTrader5 with SL/TP
- Logging of prompts, responses, and trades for auditing
- Simplified historical backtest mode that reuses the Grok prompts

## Prerequisites

- Python 3.10+
- Installed packages: `MetaTrader5`, `requests`, `pandas`, `matplotlib`,
  `mplfinance`
- Pepperstone MetaTrader5 terminal with demo credentials
- xAI API key (Grok 4 access)

## Configuration

Edit the placeholders in `trading_bot.py` or provide environment variables:

- `AccountConfig.login`: Pepperstone demo account number
- `AccountConfig.password`: demo account password (prefer secure retrieval)
- `AccountConfig.server`: e.g., `Pepperstone-Demo`
- `XAI_API_KEY`: environment variable containing your xAI API key
- `BOT_MODE`: `live` (default) or `backtest`

Optional configuration (via code or environment overrides) includes watchlist
symbols, polling interval, history depth, and risk parameters.

## Running the Bot

1. Ensure the MetaTrader5 terminal is installed and logged into the demo account.
2. Export your xAI key:

   ```bash
   export XAI_API_KEY="sk-your-key"
   ```

3. Launch the bot:

   ```bash
   python trading_bot.py
   ```

The bot runs continuously, checking for trades every five minutes. Logs and
prompt/trade histories are stored in the `logs/` directory. Charts shared with
Grok are written to `charts/`.

## Backtesting

To perform a slow, Grok-in-the-loop backtest using historical data:

```bash
export BOT_MODE=backtest
python trading_bot.py
```

This mode queries Grok for each step, which may incur latency and API usage.
Results are saved to `logs/results.csv`.

## Safety Notes

- The system enforces a 10% equity drawdown cap and 2% risk per trade.
- Demo mode is strongly recommended until the workflow and prompts are
  thoroughly verified.
- Review all orders and API responses in the logs before enabling live trading.
