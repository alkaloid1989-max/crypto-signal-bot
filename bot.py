import os, json, time, logging, math, requests, signal
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from telegram import Bot

# ---------------------- ЛОГИ ----------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ---------------------- КОНФИГ ----------------------
TG_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN")            # ⚠️ Токен бота (@BotFather)
CHAT_ID    = os.getenv("LOG_CHAT_ID")                   # ⚠️ ID канала/чата (-100xxxx) или @username
PRODUCT    = os.getenv("PRODUCT_TYPE", "USDT-FUTURES")
TF         = os.getenv("TF", "15m")                     # 15m
SLEEP_SEC  = int(os.getenv("SCAN_INTERVAL_SEC", "60"))  # опрос раз в N сек
LEVERAGE   = float(os.getenv("LEVERAGE", "20"))         # для расчета ROI в сообщениях

# Индикаторы
KC_N       = int(os.getenv("KC_N", "20"))
KC_M       = float(os.getenv("KC_M", "2.0"))            # 1.8–2.0 мягче; 2.0–2.5 строже
EMA_FAST   = int(os.getenv("EMA_FAST", "50"))
EMA_SLOW   = int(os.getenv("EMA_SLOW", "200"))
RSI_N      = int(os.getenv("RSI_N", "14"))

# Фильтры
EMA_TOL_PCT = float(os.getenv("EMA_TOL_PCT", "0.002"))
RSI_LONG_MIN = float(os.getenv("RSI_LONG_MIN", "48.0"))
RSI_SHORT_MAX= float(os.getenv("RSI_SHORT_MAX","52.0"))

# Задержка публикации на 1 бар
DELAY_BARS = int(os.getenv("DELAY_BARS", "1"))

# Entry / SL / TP / Trailing
ENTRY_MODE     = os.getenv("ENTRY_MODE", "atr")         
ENTRY_ATR_MULT = float(os.getenv("ENTRY_ATR_MULT", "0.25"))
ENTRY_PCT      = float(os.getenv("ENTRY_PCT", "0.005")) 

SL_MODE        = os.getenv("SL_MODE", "atr")            
SL_ATR_MULT    = float(os.getenv("SL_ATR_MULT", "1.0"))
SL_PCT         = float(os.getenv("SL_PCT", "0.01"))     

TP_R_MULTS     = [float(x) for x in os.getenv("TP_R_MULTS", "0.5,1.0,1.5,2.0,2.5").split(",")]
TRAIL_AFTER_TP = int(os.getenv("TRAIL_AFTER_TP", "2"))  
TRAIL_MODE     = os.getenv("TRAIL_MODE", "atr")         
TRAIL_ATR_MULT = float(os.getenv("TRAIL_ATR_MULT", "1.0"))
TRAIL_PCT      = float(os.getenv("TRAIL_PCT", "0.01"))  

MAX_PER_MINUTE = int(os.getenv("MAX_MSGS_PER_MIN", "20"))  

STATE_FILE = os.getenv("STATE_FILE", "state.json")

DEFAULT_COINS = [
    "BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","BNBUSDT","ADAUSDT","DOGEUSDT","TRXUSDT","MATICUSDT","DOTUSDT",
    "LTCUSDT","BCHUSDT","AVAXUSDT","LINKUSDT","ATOMUSDT","XLMUSDT","APTUSDT","ARBUSDT","OPUSDT","SUIUSDT",
    "NEARUSDT","INJUSDT","SEIUSDT","FILUSDT","AAVEUSDT","RUNEUSDT","DYDXUSDT","SHIBUSDT","PEPEUSDT","FTMUSDT",
    "THETAUSDT","EGLDUSDT","HBARUSDT","ROSEUSDT","ALGOUSDT","ICPUSDT","IMXUSDT","LDOUSDT","STXUSDT","CRVUSDT",
    "SUSHIUSDT","1INCHUSDT","GALAUSDT","MANAUSDT","SANDUSDT","KAVAUSDT","COREUSDT","TONUSDT","WLDUSDT","PYTHUSDT",
    "BLURUSDT","GMXUSDT","QNTUSDT","RNDRUSDT","JUPUSDT","BONKUSDT","TIAUSDT"
]
COINS = [s.strip().upper() for s in os.getenv("COINS", ",".join(DEFAULT_COINS)).split(",") if s.strip()]

assert TG_TOKEN and CHAT_ID, "Нужны TELEGRAM_BOT_TOKEN и LOG_CHAT_ID"

bot = Bot(TG_TOKEN)

# ---------------------- УТИЛИТЫ ----------------------
def save_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def load_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def fmt_price(p: float) -> str:
    if p <= 0: return f"{p:.6f}"
    if p >= 100: return f"{p:.2f}"
    if p >= 1:   return f"{p:.4f}"
    return f"{p:.8f}"

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)
  def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

def keltner(df: pd.DataFrame, n=20, m=2.0):
    mid = ema(df["close"], n)
    rng = atr(df, n)
    upper = mid + m * rng
    lower = mid - m * rng
    return mid, upper, lower, rng

def fetch_bitget_contracts(product_type: str) -> List[str]:
    url = "https://api.bitget.com/api/v2/mix/market/contracts"
    r = requests.get(url, params={"productType": product_type}, timeout=15)
    r.raise_for_status()
    data = r.json().get("data") or []
    return [x["symbol"] for x in data if "symbol" in x]

def fetch_candles(symbol: str, product_type: str, granularity: str, limit: int = 200) -> pd.DataFrame:
    url = "https://api.bitget.com/api/v2/mix/market/history-mark-candles"
    params = {"symbol": symbol, "productType": product_type, "granularity": granularity, "limit": min(max(50, limit), 200)}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json().get("data") or []
    rows = []
    for it in data:
        ts, o, h, l, c, v = it[:6]
        rows.append({"ts": int(ts), "open": float(o), "high": float(h), "low": float(l), "close": float(c), "volume": float(v)})
    df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    return df

def send(text: str):
    try:
        bot.send_message(chat_id=CHAT_ID, text=text, parse_mode="HTML", disable_web_page_preview=True)
    except Exception as e:
        logging.error("Ошибка send(): %s", e)

# ---------------------- СОСТОЯНИЕ ----------------------
state: Dict[str, Any] = load_json(STATE_FILE, {"symbols_seen": {}, "active": {}, "allowed_symbols": []})

_last_minute = 0
_sent_in_minute = 0
def throttle():
    global _last_minute, _sent_in_minute
    now_min = int(time.time() // 60)
    if now_min != _last_minute:
        _last_minute = now_min
        _sent_in_minute = 0
    if _sent_in_minute >= MAX_PER_MINUTE:
        time.sleep(2)
    _sent_in_minute += 1

def build_levels(side: str, close_prev: float, atr_prev: float):
    halfw = atr_prev * ENTRY_ATR_MULT if ENTRY_MODE == "atr" else close_prev * ENTRY_PCT
    entry_low, entry_high = close_prev - halfw, close_prev + halfw
    sl_off = atr_prev * SL_ATR_MULT if SL_MODE == "atr" else close_prev * SL_PCT
    if side == "LONG":
        sl = entry_low - sl_off
        R = entry_high - sl
        tps = [entry_high + R * m for m in TP_R_MULTS]
    else:
        sl = entry_high + sl_off
        R = sl - entry_low
        tps = [entry_low - R * m for m in TP_R_MULTS]
    entry_mid = (entry_low + entry_high) / 2.0
    return entry_low, entry_high, entry_mid, sl, tps

def check_filters_for_bar(df: pd.DataFrame, idx: int) -> Dict[str, Any]:
    mid, upper, lower, rng = keltner(df, KC_N, KC_M)
    df_ind = df.copy()
    df_ind["upper"], df_ind["lower"] = upper, lower
    df_ind["ema_fast"], df_ind["ema_slow"] = ema(df["close"], EMA_FAST), ema(df["close"], EMA_SLOW)
    df_ind["rsi"], df_ind["atr"] = rsi(df["close"], RSI_N), atr(df, KC_N)
    row, prev = df_ind.iloc[idx], df_ind.iloc[idx-1]
    return {"ts": int(row["ts"]), "close": float(row["close"]), "upper": float(row["upper"]), "lower": float(row["lower"]),
            "ema_fast": float(row["ema_fast"]), "ema_slow": float(row["ema_slow"]), "rsi": float(row["rsi"]), "atr": float(row["atr"]),
            "close_prev": float(prev["close"]), "upper_prev": float(prev["upper"]), "lower_prev": float(prev["lower"]),
            "rsi_prev": float(prev["rsi"]), "atr_prev": float(prev["atr"]),
            "ema_fast_prev": float(prev["ema_fast"]), "ema_slow_prev": float(prev["ema_slow"])}

def signal_from_prev_bar(meta: Dict[str, Any]) -> str:
    ef, es = meta["ema_fast_prev"], meta["ema_slow_prev"]
    ema_long_ok  = ef >= es * (1 - EMA_TOL_PCT)
    ema_short_ok = ef <= es * (1 + EMA_TOL_PCT)
    long_break  = meta["close_prev"] > meta["upper_prev"]
    short_break = meta["close_prev"] < meta["lower_prev"]
    rsi_long_ok  = meta["rsi_prev"] >= RSI_LONG_MIN
    rsi_short_ok = meta["rsi_prev"] <= RSI_SHORT_MAX
    if long_break and ema_long_ok and rsi_long_ok: return "LONG"
    if short_break and ema_short_ok and rsi_short_ok: return "SHORT"
    return ""

def format_open_msg(symbol: str, side: str, entry_low: float, entry_high: float, tps: List[float], sl: float):
    base = symbol.replace("USDT", "")
    lines = [f"📌 #{base}/USDT", f"Leverage: Cross {int(LEVERAGE)}X", f"⚠️ Signal: {'long' if side=='LONG' else 'short'}", "",
             "☑️ Entry Zone:", f"{fmt_price(entry_low)}-{fmt_price(entry_high)}", "", "🎯 Take-Profit Targets:"]
    lines += [f"{i+1}) {fmt_price(tp)}" for i, tp in enumerate(tps)]
    lines += ["", f"⛔️ Stop-Loss: {fmt_price(sl)}", "", "Риск-менеджмент:", "Не более 2-3% депозита на одну сделку"]
    return "\n".join(lines)

def main():
    if not state.get("allowed_symbols"):
        try:
            available = set(fetch_bitget_contracts(PRODUCT))
            state["allowed_symbols"] = [s for s in COINS if s in available]
            save_json(STATE_FILE, state)
        except: state["allowed_symbols"] = COINS

    send(f"✅ Бот сигналов запущен. Таймфрейм {TF}, Keltner n={KC_N} m={KC_M}")
    while True:
        try:
            for symbol in state["allowed_symbols"]:
                df = fetch_candles(symbol, PRODUCT, TF, 200)
                if len(df) < max(EMA_SLOW, KC_N) + 5: continue
                last_ts = int(df["ts"].iloc[-1])
                if state["symbols_seen"].get(symbol) == last_ts: continue
                meta = check_filters_for_bar(df, -1)
                side = signal_from_prev_bar(meta)
                if side and symbol not in state["active"]:
                    entry_low, entry_high, entry_mid, sl, tps = build_levels(side, meta["close_prev"], meta["atr_prev"])
                    send(format_open_msg(symbol, side, entry_low, entry_high, tps, sl))
                    throttle()
                    state["active"][symbol] = {"side": side, "entry_mid": entry_mid, "sl": sl, "tps": tps, "reached_tp": -1}
                state["symbols_seen"][symbol] = last_ts
            save_json(STATE_FILE, state)
        except Exception as e:
            logging.exception("Ошибка: %s", e)
        time.sleep(SLEEP_SEC)

if __name__ == "__main__":
    main()

