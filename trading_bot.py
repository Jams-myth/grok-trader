"""Automated trading system integrating MetaTrader5 with Grok 4 (xAI).

This module implements a demo-oriented automated trading workflow that
combines MetaTrader5 for execution/data with Grok 4 for higher level
analysis. The implementation emphasises clarity, observability, and risk
controls suitable for paper-trading and incremental enhancement.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import MetaTrader5 as mt5
import mplfinance as mpf
import pandas as pd
import requests


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class AccountConfig:
    """Credentials and connection data for the MetaTrader5 account."""

    login: int = 0  # Replace with your Pepperstone demo account ID
    password: str = "YOUR_PASSWORD"  # Replace with secure retrieval
    server: str = "Pepperstone-Demo"  # Update to live server when ready
    path: Optional[str] = None  # Optional terminal path if multiple installs


@dataclasses.dataclass
class RiskConfig:
    """Defines account level risk constraints."""

    risk_per_trade: float = 0.02  # 2% of balance per trade
    max_drawdown: float = 0.10  # Close all if balance equity drawdown > 10%
    max_trades_per_symbol: int = 1


@dataclasses.dataclass
class BotConfig:
    """Top level configuration."""

    account: AccountConfig = dataclasses.field(default_factory=AccountConfig)
    risk: RiskConfig = dataclasses.field(default_factory=RiskConfig)
    watchlist: Tuple[str, ...] = ("EURUSD", "USDJPY")
    timeframe: int = mt5.TIMEFRAME_H1
    history_bars: int = 100
    loop_interval_seconds: int = 300
    log_directory: Path = Path("logs")
    prompts_log: Path = Path("logs/prompts.jsonl")
    trades_log: Path = Path("logs/trades.jsonl")
    results_log: Path = Path("logs/results.csv")
    api_base_url: str = "https://x.ai/api"
    grok_model: str = "grok-4"
    grok_api_key_env: str = "XAI_API_KEY"
    chart_directory: Path = Path("charts")
    demo_mode: bool = True  # Only run on demo until validated
    retry_attempts: int = 3
    retry_delay: float = 5.0


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def ensure_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix:
            path.touch(exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)


def setup_logging(config: BotConfig) -> None:
    ensure_directories(
        [config.log_directory, config.prompts_log, config.trades_log, config.results_log]
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(config.log_directory / "bot.log"),
            logging.StreamHandler(),
        ],
    )


logger = logging.getLogger("grok_trading_bot")


# ---------------------------------------------------------------------------
# Grok (xAI) integration
# ---------------------------------------------------------------------------


def call_grok(
    config: BotConfig,
    prompt: str,
    images: Optional[List[str]] = None,
    temperature: float = 0.0,
) -> Dict:
    """Send a prompt to the Grok API and return the parsed JSON response."""

    api_key = os.getenv(config.grok_api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing Grok API key. Set environment variable {config.grok_api_key_env}."
        )

    payload = {
        "model": config.grok_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert trading assistant. Always respond with strict JSON."
                    " Numeric values must be floats."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": temperature,
    }
    if images:
        payload["images"] = images

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for attempt in range(1, config.retry_attempts + 1):
        try:
            response = requests.post(
                f"{config.api_base_url}/v1/chat/completions",
                headers=headers,
                data=json.dumps(payload),
                timeout=60,
            )
            if response.status_code == 429 and attempt < config.retry_attempts:
                logger.warning("Grok rate limited. Retrying in %.1f seconds", config.retry_delay)
                time.sleep(config.retry_delay)
                continue
            response.raise_for_status()
            content = response.json()
            message = content.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            parsed = json.loads(message)
            return parsed
        except (requests.RequestException, json.JSONDecodeError):
            logger.exception("Grok call failed on attempt %s/%s", attempt, config.retry_attempts)
            if attempt == config.retry_attempts:
                raise
            time.sleep(config.retry_delay)
    raise RuntimeError("Unable to reach Grok API")


# ---------------------------------------------------------------------------
# MetaTrader5 utilities
# ---------------------------------------------------------------------------


def initialize_mt5(account: AccountConfig) -> None:
    logger.info("Initializing MetaTrader5 connection to %s", account.server)
    if not mt5.initialize(path=account.path, server=account.server):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    authorized = mt5.login(account.login, password=account.password, server=account.server)
    if not authorized:
        error = mt5.last_error()
        mt5.shutdown()
        raise RuntimeError(f"MT5 login failed: {error}")
    logger.info("MT5 connection established")


def shutdown_mt5() -> None:
    if mt5.initialize():
        mt5.shutdown()


# ---------------------------------------------------------------------------
# Data collection and charting
# ---------------------------------------------------------------------------


def fetch_positions() -> List[mt5.TradePosition]:
    positions = mt5.positions_get()
    if positions is None:
        error = mt5.last_error()
        raise RuntimeError(f"Failed to fetch positions: {error}")
    return list(positions)


def fetch_rates(symbol: str, timeframe: int, count: int) -> pd.DataFrame:
    utc_now = dt.datetime.utcnow()
    rates = mt5.copy_rates_from(symbol, timeframe, utc_now, count)
    if rates is None:
        error = mt5.last_error()
        raise RuntimeError(f"Failed to fetch rates for {symbol}: {error}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    return df


def generate_chart(symbol: str, rates: pd.DataFrame, config: BotConfig) -> Path:
    ensure_directories([config.chart_directory])
    chart_path = config.chart_directory / f"{symbol}_{int(time.time())}.png"

    mpf.plot(
        rates,
        type="candle",
        style="charles",
        title=f"{symbol} H1 Chart",
        volume=True,
        savefig=dict(fname=str(chart_path), dpi=150, bbox_inches="tight"),
    )
    logger.info("Saved chart for %s at %s", symbol, chart_path)
    return chart_path


# ---------------------------------------------------------------------------
# Risk management and order execution
# ---------------------------------------------------------------------------


def get_account_info() -> mt5.AccountInfo:
    info = mt5.account_info()
    if info is None:
        raise RuntimeError(f"Unable to retrieve account info: {mt5.last_error()}")
    return info


def estimate_pip_value(symbol: str) -> float:
    """Rudimentary pip value estimate per lot."""

    if symbol.endswith("JPY"):
        return 9.0
    if symbol.endswith("XAU") or symbol.endswith("XAG"):
        return 1.0
    return 10.0


def calculate_lot_size(
    balance: float,
    risk_config: RiskConfig,
    symbol: str,
    entry_price: float,
    stop_price: float,
) -> float:
    risk_amount = balance * risk_config.risk_per_trade
    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= 0:
        raise ValueError("Stop distance must be positive")

    pip_value = estimate_pip_value(symbol)
    pip_size = 0.01 if symbol.endswith("JPY") else 0.0001
    stop_pips = stop_distance / pip_size
    if stop_pips == 0:
        raise ValueError("Stop pips cannot be zero")

    lots = risk_amount / (stop_pips * pip_value)
    lots = max(round(lots, 2), 0.01)
    return lots


def has_open_trade(symbol: str, positions: List[mt5.TradePosition], max_trades: int) -> bool:
    count = sum(1 for pos in positions if pos.symbol == symbol)
    return count >= max_trades


def send_market_order(
    symbol: str,
    direction: str,
    volume: float,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    comment: str,
) -> Dict:
    order_type = mt5.ORDER_TYPE_BUY if direction.upper() == "BUY" else mt5.ORDER_TYPE_SELL

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"No tick data for {symbol}: {mt5.last_error()}")
    price = tick.ask if direction.upper() == "BUY" else tick.bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": stop_loss,
        "tp": take_profit,
        "deviation": 20,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }

    logger.info("Sending order: %s", request)
    result = mt5.order_send(request)
    if result is None:
        raise RuntimeError(f"Order send returned None: {mt5.last_error()}")
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        raise RuntimeError(f"Order failed: {result.retcode} {result.comment}")
    return {
        "order": result.order,
        "deal": result.deal,
        "price": result.price,
        "volume": result.volume,
        "symbol": symbol,
        "direction": direction,
    }


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def append_jsonl(path: Path, payload: Dict) -> None:
    ensure_directories([path])
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload) + "\n")


# ---------------------------------------------------------------------------
# Core trading workflow
# ---------------------------------------------------------------------------


class GrokTradingBot:
    def __init__(self, config: Optional[BotConfig] = None) -> None:
        self.config = config or BotConfig()
        setup_logging(self.config)
        self.lock = threading.Lock()

    def build_screening_prompt(
        self, positions: List[mt5.TradePosition], market_data: Dict[str, Dict]
    ) -> str:
        positions_payload = [
            {
                "ticket": pos.ticket,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL",
                "price_open": pos.price_open,
                "sl": pos.sl,
                "tp": pos.tp,
                "profit": pos.profit,
            }
            for pos in positions
        ]
        watchlist_payload = {
            symbol: data["rates"].tail(5).to_dict(orient="list")
            for symbol, data in market_data.items()
        }
        prompt = (
            "You are screening FX pairs and current positions."
            " Respond with JSON using key 'trades' listing exactly three recommended trades"
            " ordered by preference. Each entry must include 'symbol' and 'direction' (BUY/SELL)."
            " Only include symbols from the provided watchlist."
            "\nWatchlist recent data: "
            f"{json.dumps(watchlist_payload)}"
            "\nOpen positions: "
            f"{json.dumps(positions_payload)}"
        )
        return prompt

    def build_chart_prompt(self, symbol: str) -> str:
        return (
            "Analyze the provided chart for symbol {symbol}. Respond strictly in JSON with keys "
            "'entry', 'stop', and 'target'. Values must be floats with at least four decimals."
        ).format(symbol=symbol)

    def collect_market_data(self) -> Dict[str, Dict]:
        market_data = {}
        for symbol in self.config.watchlist:
            try:
                rates = fetch_rates(symbol, self.config.timeframe, self.config.history_bars)
                chart_path = generate_chart(symbol, rates, self.config)
                market_data[symbol] = {"rates": rates, "chart": chart_path}
            except Exception:
                logger.exception("Failed to collect data for %s", symbol)
        return market_data

    def should_skip_trading(self) -> bool:
        now = dt.datetime.utcnow()
        if now.weekday() >= 5:
            return True
        if now.weekday() == 4 and now.hour >= 21:
            return True
        if now.weekday() == 6 and now.hour < 22:
            return True
        return False

    def enforce_drawdown(self, info: mt5.AccountInfo) -> None:
        drawdown = (info.balance - info.equity) / info.balance if info.balance else 0
        if drawdown > self.config.risk.max_drawdown:
            logger.warning("Max drawdown exceeded (%.2f%%). Closing positions.", drawdown * 100)
            positions = fetch_positions()
            for pos in positions:
                order_type = (
                    mt5.ORDER_TYPE_SELL
                    if pos.type == mt5.POSITION_TYPE_BUY
                    else mt5.ORDER_TYPE_BUY
                )
                tick = mt5.symbol_info_tick(pos.symbol)
                if tick is None:
                    continue
                price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": pos.symbol,
                    "volume": pos.volume,
                    "type": order_type,
                    "position": pos.ticket,
                    "price": price,
                    "deviation": 20,
                    "comment": "Drawdown protection",
                }
                result = mt5.order_send(request)
                logger.info("Closed position %s with result %s", pos.ticket, result)
            raise RuntimeError("Trading halted due to drawdown limit")

    def parse_trades(self, grok_response: Dict) -> List[Dict[str, str]]:
        trades = grok_response.get("trades", [])
        if not isinstance(trades, list):
            raise ValueError("Invalid Grok response format: 'trades' missing")
        formatted = []
        for trade in trades:
            symbol = trade.get("symbol")
            direction = trade.get("direction", "").upper()
            if symbol and direction in {"BUY", "SELL"}:
                formatted.append({"symbol": symbol, "direction": direction})
        return formatted[:3]

    def process_trade_signal(
        self,
        symbol: str,
        direction: str,
        trade_levels: Dict[str, float],
        account_info: mt5.AccountInfo,
        positions: List[mt5.TradePosition],
    ) -> Optional[Dict]:
        if symbol not in self.config.watchlist:
            logger.info("Skipping %s - not in watchlist", symbol)
            return None
        if has_open_trade(symbol, positions, self.config.risk.max_trades_per_symbol):
            logger.info("Skipping %s - open trade exists", symbol)
            return None

        entry = trade_levels.get("entry")
        stop = trade_levels.get("stop")
        target = trade_levels.get("target")
        if not all(isinstance(v, (int, float)) for v in (entry, stop, target)):
            logger.warning("Invalid trade levels for %s: %s", symbol, trade_levels)
            return None

        try:
            volume = calculate_lot_size(
                balance=account_info.balance,
                risk_config=self.config.risk,
                symbol=symbol,
                entry_price=entry,
                stop_price=stop,
            )
            result = send_market_order(
                symbol=symbol,
                direction=direction,
                volume=volume,
                entry_price=entry,
                stop_loss=stop,
                take_profit=target,
                comment="Grok signal",
            )
            trade_log = {
                "timestamp": dt.datetime.utcnow().isoformat(),
                "symbol": symbol,
                "direction": direction,
                "entry": entry,
                "stop": stop,
                "target": target,
                "volume": volume,
                "order_result": result,
            }
            append_jsonl(self.config.trades_log, trade_log)
            logger.info("Executed trade: %s", trade_log)
            return trade_log
        except Exception:
            logger.exception("Failed to execute trade for %s", symbol)
            return None

    def run_once(self) -> None:
        if self.should_skip_trading():
            logger.info("Skipping trading window")
            return

        info = get_account_info()
        self.enforce_drawdown(info)

        positions = fetch_positions()
        market_data = self.collect_market_data()
        screening_prompt = self.build_screening_prompt(positions, market_data)
        append_jsonl(
            self.config.prompts_log,
            {"timestamp": dt.datetime.utcnow().isoformat(), "prompt": screening_prompt},
        )
        screening_response = call_grok(self.config, screening_prompt)
        append_jsonl(
            self.config.prompts_log,
            {
                "timestamp": dt.datetime.utcnow().isoformat(),
                "response": screening_response,
            },
        )

        trades = self.parse_trades(screening_response)
        if not trades:
            logger.info("No trades suggested by Grok")
            return

        top_trade = trades[0]
        symbol = top_trade["symbol"]
        market = market_data.get(symbol)
        if not market:
            logger.warning("No market data for top symbol %s", symbol)
            return

        chart_prompt = self.build_chart_prompt(symbol)
        append_jsonl(
            self.config.prompts_log,
            {"timestamp": dt.datetime.utcnow().isoformat(), "prompt": chart_prompt},
        )
        chart_response = call_grok(
            self.config, chart_prompt, images=[str(market["chart"])]
        )
        append_jsonl(
            self.config.prompts_log,
            {
                "timestamp": dt.datetime.utcnow().isoformat(),
                "response": chart_response,
            },
        )

        self.process_trade_signal(
            symbol=symbol,
            direction=top_trade["direction"],
            trade_levels=chart_response,
            account_info=info,
            positions=positions,
        )

    def run_forever(self) -> None:
        initialize_mt5(self.config.account)
        try:
            while True:
                with self.lock:
                    try:
                        self.run_once()
                    except Exception:
                        logger.exception("Error during trading loop")
                time.sleep(self.config.loop_interval_seconds)
        finally:
            shutdown_mt5()

    def run_backtest(
        self,
        symbol: str,
        start: dt.datetime,
        end: dt.datetime,
        initial_balance: float = 10000.0,
    ) -> pd.DataFrame:
        rates = mt5.copy_rates_range(symbol, self.config.timeframe, start, end)
        if rates is None:
            raise RuntimeError(f"Unable to fetch historical rates: {mt5.last_error()}")

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        balance = initial_balance
        equity_curve = []
        positions: List[Dict] = []

        for i in range(self.config.history_bars, len(df)):
            window = df.iloc[i - self.config.history_bars : i]
            chart_path = generate_chart(symbol, window, self.config)
            prompt = (
                "You are backtesting trading decisions. Based on the provided chart,"
                " return JSON with keys 'entry', 'stop', 'target'."
            )
            response = call_grok(self.config, prompt, images=[str(chart_path)])
            entry = response.get("entry")
            stop = response.get("stop")
            target = response.get("target")
            if not all(isinstance(v, (int, float)) for v in (entry, stop, target)):
                continue

            direction = "BUY" if target > entry else "SELL"
            lot = calculate_lot_size(balance, self.config.risk, symbol, entry, stop)

            bar = df.iloc[i]
            if direction == "BUY":
                stop_hit = bar.low <= stop
                target_hit = bar.high >= target
            else:
                stop_hit = bar.high >= stop
                target_hit = bar.low <= target

            if stop_hit and target_hit:
                target_hit = bar.close > entry

            risk_amount = balance * self.config.risk.risk_per_trade
            reward = risk_amount if target_hit else -risk_amount
            balance += reward
            equity_curve.append({"time": bar.name, "balance": balance})
            positions.append(
                {
                    "time": bar.name.isoformat(),
                    "entry": entry,
                    "stop": stop,
                    "target": target,
                    "direction": direction,
                    "result": reward,
                }
            )

        results_df = pd.DataFrame(equity_curve)
        results_df.to_csv(self.config.results_log, index=False)
        append_jsonl(
            self.config.trades_log,
            {
                "timestamp": dt.datetime.utcnow().isoformat(),
                "mode": "backtest",
                "symbol": symbol,
                "positions": positions,
            },
        )
        return results_df


def main() -> None:
    bot = GrokTradingBot()
    mode = os.getenv("BOT_MODE", "live")
    if mode == "backtest":
        start = dt.datetime.utcnow() - dt.timedelta(days=30)
        end = dt.datetime.utcnow()
        initialize_mt5(bot.config.account)
        try:
            bot.run_backtest(symbol=bot.config.watchlist[0], start=start, end=end)
        finally:
            shutdown_mt5()
    else:
        bot.run_forever()


if __name__ == "__main__":
    main()
