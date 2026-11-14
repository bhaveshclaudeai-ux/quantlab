import copy, itertools, json, math, os, re, time
from datetime import datetime, timezone
import numpy as np, pandas as pd, requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="QuantLab")
HERE = os.path.dirname(os.path.abspath(__file__))
BASES = ["https://data-api.binance.vision", "https://api.binance.com", "https://api1.binance.com", "https://api.binance.us"]
TF = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240, "1d": 1440}
GROUPS = ("entry_long", "exit_long", "entry_short", "exit_short")
CACHE = {}

# ---------------- data ----------------
def _get(path, params):
    err = None
    for b in BASES:
        try:
            r = requests.get(b + path, params=params, timeout=20)
            if r.status_code == 200:
                return r.json()
            err = r.text[:200]
        except requests.RequestException as e:
            err = str(e)
    raise HTTPException(502, f"Data source error (shared server IP may be rate-limited, try again in a few minutes): {err}")

@app.get("/api/symbols")
def symbols():
    c = CACHE.get("sym")
    if not c or time.time() - c[0] > 3600:
        d = _get("/api/v3/exchangeInfo", {})
        s = sorted(x["symbol"] for x in d["symbols"] if x["status"] == "TRADING" and x["quoteAsset"] in ("USDT", "USDC", "FDUSD", "BTC"))
        CACHE["sym"] = c = (time.time(), s)
    return c[1]

def load(sym, tf, n):
    c = CACHE.get((sym, tf))
    if c and time.time() - c[0] < 300 and len(c[1]) >= n:
        return c[1].tail(n).reset_index(drop=True)
    rows, end = [], None
    while len(rows) < n:
        p = {"symbol": sym, "interval": tf, "limit": min(1000, n + 1 - len(rows))}
        if end:
            p["endTime"] = end
        d = _get("/api/v3/klines", p)
        if not d:
            break
        rows = d + rows
        end = d[0][0] - 1
        if len(d) < p["limit"]:
            break
    if len(rows) < 100:
        raise HTTPException(400, "Not enough data for this symbol/interval")
    df = pd.DataFrame(rows).iloc[:-1, :6]  # drop still-forming candle
    df.columns = ["t", "open", "high", "low", "close", "volume"]
    df = df.astype(float)
    df["t"] = (df["t"] // 1000).astype(int)
    CACHE[(sym, tf)] = (time.time(), df)
    return df.tail(n).reset_index(drop=True)

def load_range(sym, tf, a, b, cap=20000):
    """Candles between two dates (UTC, ms). Only the start -> end window, max 20,000 candles."""
    tfms = TF[tf] * 60000
    if a is None:
        a = b - cap * tfms
    key = (sym, tf, a, b)
    c = CACHE.get(key)
    if c and time.time() - c[0] < 300:
        return c[1]
    rows, cur = [], a
    while len(rows) < cap:
        p = {"symbol": sym, "interval": tf, "limit": 1000, "startTime": cur}
        if b:
            p["endTime"] = b
        d = _get("/api/v3/klines", p)
        if not d:
            break
        rows += d
        cur = d[-1][0] + 1
        if len(d) < 1000:
            break
    rows = [r for r in rows if r[0] + tfms <= time.time() * 1000]  # drop candles still forming
    if len(rows) < 100:
        raise HTTPException(400, "Not enough data in this date range (need at least 100 candles)")
    df = pd.DataFrame(rows).iloc[:, :6]
    df.columns = ["t", "open", "high", "low", "close", "volume"]
    df = df.astype(float)
    df["t"] = (df["t"] // 1000).astype(int)
    CACHE[key] = (time.time(), df)
    return df

# ---------------- futures / gold via Yahoo Finance (NQ, S&P 500, Gold) ----------------
YF = {"NQ": "NQ=F", "NQ1": "NQ=F", "NASDAQ": "NQ=F", "NASDAQ100": "NQ=F", "NDX": "NQ=F", "MNQ": "MNQ=F",
      "ES": "ES=F", "ES1": "ES=F", "SPX": "ES=F", "SPX500": "ES=F", "SP500": "ES=F", "SNP": "ES=F", "SNP500": "ES=F", "MES": "MES=F",
      "GOLD": "GC=F", "GC": "GC=F", "GC1": "GC=F", "XAU": "GC=F", "XAUUSD": "GC=F", "MGC": "MGC=F"}
YLIM = {"1m": ("1m", 6), "3m": ("1m", 6), "5m": ("5m", 59), "15m": ("15m", 59), "30m": ("30m", 59),
        "1h": ("60m", 729), "2h": ("60m", 729), "4h": ("60m", 729), "1d": ("1d", 36500)}
RESAMPLE = {"3m": "3min", "2h": "2h", "4h": "4h"}

def _yahoo_get(tick, interval, p1, p2):
    err = None
    for host in ("query1", "query2"):
        try:
            r = requests.get(f"https://{host}.finance.yahoo.com/v8/finance/chart/{tick}", timeout=25,
                             params={"interval": interval, "period1": int(p1), "period2": int(p2)},
                             headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"})
            ch = r.json().get("chart", {})
            if ch.get("error"):
                err = ch["error"].get("description") or str(ch["error"])
                continue
            res = (ch.get("result") or [None])[0]
            if res and res.get("timestamp"):
                return res
            err = "no data returned"
        except Exception as e:
            err = str(e)
    raise HTTPException(502, f"Yahoo data error for {tick}: {str(err)[:140]}")

def load_yahoo(tick, tf, n, a=None, b=None):
    """NQ / ES / Gold futures candles. dk = CME trading day (a new day starts 18:00 New York time)."""
    key = (tick, tf, n if (a is None and b is None) else 0, a, b)
    c = CACHE.get(key)
    if c and time.time() - c[0] < 300:
        return c[1]
    yi, lim = YLIM[tf]
    now = time.time()
    p2 = min(b / 1000, now) if b else now
    p1 = max((a / 1000) if a else p2 - lim * 86400, now - lim * 86400)   # Yahoo keeps only limited intraday history
    if p1 >= p2:
        raise HTTPException(400, f"That date range is outside Yahoo's history for {tf} candles (1m ~6 days, 5-30m ~59 days, 1h ~2 years)")
    res = _yahoo_get(tick, yi, p1, p2)
    qd = res["indicators"]["quote"][0]
    df = pd.DataFrame({"t": res["timestamp"], "open": qd["open"], "high": qd["high"], "low": qd["low"], "close": qd["close"], "volume": qd.get("volume") or 0})
    df = df.dropna(subset=["open", "high", "low", "close"])
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    df = df.astype(float)
    if tf in RESAMPLE:
        g = df.set_index(pd.to_datetime(df["t"], unit="s", utc=True)).drop(columns="t").resample(RESAMPLE[tf], origin="start_day").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna(subset=["open"])
        df = g.reset_index(drop=True)
        df.insert(0, "t", ((g.index - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)).astype(float))
    df["t"] = df["t"].astype(int)
    df = df[df["t"] + TF[tf] * 60 <= now].reset_index(drop=True)          # drop the candle still forming
    if len(df) < 100:
        raise HTTPException(400, f"Only {len(df)} candles available from Yahoo for {tf} (history: 1m ~6 days, 5-30m ~59 days, 1h ~2 years). Try a bigger timeframe or an earlier date.")
    loc = pd.to_datetime(df["t"], unit="s", utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None) + pd.Timedelta(hours=6)
    df["dk"] = (loc.dt.normalize() - pd.Timestamp("1970-01-01")).dt.days.astype(int)
    if a is None and b is None:
        df = df.tail(n).reset_index(drop=True)
    CACHE[key] = (time.time(), df)
    return df

def _dk(df):
    """Trading-day index: UTC day for crypto, CME session day for futures."""
    return df["dk"] if "dk" in df else df["t"] // 86400

# ---------------- CSV import (any market, any history) ----------------
CSVS = {}
CSV_CAP = 60000
CSV_TZ = ("UTC", "Asia/Kolkata", "America/New_York", "Europe/London", "Europe/Zurich")
BASES_MIN = [1, 3, 5, 15, 30, 60, 120, 240, 1440]

def _ptime(s, tz):
    s = s.astype(str).str.strip()
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.9:                                     # unix seconds / milliseconds
        return pd.to_datetime(num, unit="ms" if num.median() > 1e11 else "s", utc=True, errors="coerce")
    f = s.iloc[0]
    if re.match(r"^\d{4}\.\d{2}\.\d{2}", f):                         # MT4: 2024.01.02
        s = s.str.replace(r"^(\d{4})\.(\d{2})\.(\d{2})", r"\1-\2-\3", regex=True)
    t = pd.to_datetime(s, dayfirst=bool(re.match(r"^\d{2}[./]\d{2}[./]\d{4}", f)), errors="coerce")
    if t.dt.tz is None:
        return t.dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")
    return t.dt.tz_convert("UTC")

def parse_csv(text, tz="UTC"):
    import io
    try:
        raw = pd.read_csv(io.StringIO(text), sep=None, engine="python", header=None, dtype=str, skip_blank_lines=True)
    except Exception as e:
        raise HTTPException(400, f"Could not read this file as CSV ({str(e)[:80]})")
    if raw.shape[1] < 5 or len(raw) < 10:
        raise HTTPException(400, "CSV needs at least: time, open, high, low, close")
    head = [str(x).strip().lower() for x in raw.iloc[0]]
    vol = None
    if re.search(r"open|high|low|close", " ".join(head)):            # file has a header row
        raw = raw.iloc[1:]
        col = lambda *ks: next((i for i, h in enumerate(head) if any(k in h for k in ks) and "interest" not in h), None)
        io_, ih, il, ic, iv = col("open"), col("high"), col("low"), col("close"), col("vol")
        tcols = [i for i, h in enumerate(head) if re.search(r"time|date|gmt|stamp", h)]
        if None in (io_, ih, il, ic) or not tcols:
            raise HTTPException(400, f"Could not find time/open/high/low/close columns in header: {head[:8]}")
        ts = raw.iloc[:, tcols[0]] if len(tcols) == 1 else raw.iloc[:, tcols[0]].astype(str) + " " + raw.iloc[:, tcols[1]].astype(str)
        o, h, l, c = (raw.iloc[:, i] for i in (io_, ih, il, ic))
        vol = raw.iloc[:, iv] if iv is not None else None
    else:                                                            # no header: [dt,o,h,l,c,(v)] or MT4 [date,time,o,h,l,c,(v)]
        if raw.shape[1] >= 6 and re.match(r"^\d{1,2}:\d{2}", str(raw.iloc[0, 1]).strip()):
            ts = raw.iloc[:, 0].astype(str) + " " + raw.iloc[:, 1].astype(str)
            o, h, l, c = (raw.iloc[:, i] for i in (2, 3, 4, 5))
            vol = raw.iloc[:, 6] if raw.shape[1] > 6 else None
        else:
            ts = raw.iloc[:, 0]
            o, h, l, c = (raw.iloc[:, i] for i in (1, 2, 3, 4))
            vol = raw.iloc[:, 5] if raw.shape[1] > 5 else None
    t = _ptime(ts, tz)
    df = pd.DataFrame({"ts": t.values if hasattr(t, "values") else t, "open": pd.to_numeric(o, errors="coerce"), "high": pd.to_numeric(h, errors="coerce"),
                       "low": pd.to_numeric(l, errors="coerce"), "close": pd.to_numeric(c, errors="coerce"),
                       "volume": pd.to_numeric(vol, errors="coerce").fillna(0) if vol is not None else 1.0})
    df = df.dropna(subset=["ts", "open", "high", "low", "close"])
    df["t"] = ((pd.to_datetime(df["ts"], utc=True) - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)).astype("int64")
    df = df.drop(columns="ts").sort_values("t").drop_duplicates("t").reset_index(drop=True)
    df = df[["t", "open", "high", "low", "close", "volume"]]
    if len(df) < 100:
        raise HTTPException(400, f"Only {len(df)} valid rows found - check the date format / columns")
    step = float(np.median(np.diff(df["t"].to_numpy()))) / 60
    if step < 0.99:
        raise HTTPException(400, "This looks like tick data. Please export 1-minute (or larger) candles.")
    base = min(BASES_MIN, key=lambda x: abs(x - step))
    return df, base

def _sess_day(df):
    loc = pd.to_datetime(df["t"], unit="s", utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None) + pd.Timedelta(hours=6)
    return (loc.dt.normalize() - pd.Timestamp("1970-01-01")).dt.days.astype(int)

class CsvReq(BaseModel):
    name: str
    text: str
    tz: str = "UTC"
    day: str = "utc"

@app.post("/api/csv")
def api_csv(q: CsvReq):
    if len(q.text) > 80_000_000:
        raise HTTPException(400, "File too large (max ~80 MB). Use a shorter period or a larger timeframe.")
    tz = q.tz if q.tz in CSV_TZ else "UTC"
    df, base = parse_csv(q.text, tz)
    key = re.sub(r"[^a-z0-9_]", "", q.name.lower()) or "file"
    CSVS.pop(key, None)
    CSVS[key] = dict(df=df, base=base, day="ny18" if q.day == "ny18" else "utc")
    while len(CSVS) > 5:
        CSVS.pop(next(iter(CSVS)))
    iso = lambda s: pd.Timestamp(int(s), unit="s", tz="UTC").strftime("%Y-%m-%d %H:%M")
    return dict(key="CSV:" + key, rows=len(df), base=base, first=iso(df["t"].iloc[0]), last=iso(df["t"].iloc[-1]))

def load_csv(key, tf, n, a, b):
    e = CSVS.get(key)
    if not e:
        raise HTTPException(404, f"CSV not loaded: '{key}' is not on the server (the server forgets files when it sleeps). Import the file again.")
    base, want, df = e["base"], TF[tf], e["df"]
    if want < base:
        raise HTTPException(400, f"This file has {base}-minute candles, so you cannot use a smaller timeframe ({tf}). Pick {base}m or larger.")
    if want % base:
        raise HTTPException(400, f"Timeframe {tf} must be a multiple of the file's {base}-minute candles.")
    if want > base:
        g = df.set_index(pd.to_datetime(df["t"], unit="s", utc=True)).drop(columns="t").resample(f"{want}min", origin="start_day").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna(subset=["open"])
        df = g.reset_index(drop=True)
        df.insert(0, "t", ((g.index - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)).astype("int64"))
    if a is not None or b is not None:
        df = df[(df["t"] >= (a / 1000 if a else 0)) & (df["t"] <= (b / 1000 if b else 4e9))]
        df = df.tail(CSV_CAP)
    else:
        df = df.tail(n)
    df = df.reset_index(drop=True)
    if len(df) < 100:
        raise HTTPException(400, f"Only {len(df)} candles in this selection - widen the dates or lower the timeframe.")
    if e["day"] == "ny18":
        df["dk"] = _sess_day(df)
    return df

# ---------------- indicators / signals ----------------
CATALOG = {
    "Price": {"close": 0, "open": 0, "high": 0, "low": 0, "volume": 0},
    "Trend": {"ema": 1, "sma": 1, "supertrend": 1, "vwap": 0, "adx": 1, "plus_di": 1, "minus_di": 1, "hh": 1, "ll": 1},
    "Momentum": {"rsi": 1, "macd": 0, "macd_sig": 0, "stoch": 1, "cci": 1, "roc": 1},
    "Volatility": {"atr": 1, "bb_up": 1, "bb_lo": 1, "body_pct": 0, "range_pct": 0},
    "Volume": {"vol_sma": 1},
    "Opening range": {"or_high": 3, "or_low": 3},
    "Volume profile": {"vp_poc": 1, "vp_vah": 1, "vp_val": 1},
    "Price action": {"day_open": 0, "day_high": 0, "day_low": 0, "pw_high": 0, "pw_low": 0, "prev_high": 0, "prev_low": 0, "prev_close": 0, "pd_high": 0, "pd_low": 0, "pd_close": 0,
                     "bull_candle": 0, "bear_candle": 0, "bull_engulf": 0, "bear_engulf": 0, "hammer": 0,
                     "shooting_star": 0, "doji": 0, "inside_bar": 0},
    "Time": {"hour": 0, "dow": 0},
}
KIND_FLAG = {k: v for g in CATALOG.values() for k, v in g.items()}
KIND_FLAG["num"] = 2
PATTERNS = ("bull_candle", "bear_candle", "bull_engulf", "bear_engulf", "hammer", "shooting_star", "doji", "inside_bar")

def _tr(df):
    h, l, c = df["high"], df["low"], df["close"]
    return pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)

_VP = {}

def _vp(df, p, bins=100, va=0.7):
    """Volume profile of the previous p days, per day: (POC, VAH, VAL). Volume is spread evenly over each candle's range."""
    key = (len(df), int(df["t"].iat[0]), int(df["t"].iat[-1]), p)
    if key in _VP:
        return _VP[key]
    t, h, l, v = df["t"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy(), df["volume"].to_numpy()
    ds = np.r_[0, np.flatnonzero(np.diff(_dk(df).to_numpy())) + 1, len(t)]
    poc, vah, val = (np.full(len(t), np.nan) for _ in range(3))
    for j in range(1, len(ds) - 1):
        a, b = ds[max(0, j - p)], ds[j]
        lo, hi = l[a:b].min(), h[a:b].max()
        if hi <= lo:
            continue
        w = (hi - lo) / bins
        i0 = np.clip(((l[a:b] - lo) / w).astype(int), 0, bins - 1)
        i1 = np.clip(((h[a:b] - lo) / w).astype(int), 0, bins - 1)
        share = v[a:b] / (i1 - i0 + 1)
        dd = np.zeros(bins + 1)
        np.add.at(dd, i0, share)
        np.add.at(dd, i1 + 1, -share)
        prof = np.cumsum(dd)[:bins]
        k = int(prof.argmax()); acc, lo_i, hi_i, tot = prof[k], k, k, prof.sum()
        while acc < va * tot and (lo_i > 0 or hi_i < bins - 1):
            up = prof[hi_i + 1] if hi_i < bins - 1 else -1.0
            dn = prof[lo_i - 1] if lo_i > 0 else -1.0
            if up >= dn: hi_i += 1; acc += up
            else: lo_i -= 1; acc += dn
        s, e = ds[j], ds[j + 1]
        poc[s:e], vah[s:e], val[s:e] = lo + (k + .5) * w, lo + (hi_i + 1) * w, lo + lo_i * w
    if len(_VP) > 8:
        _VP.clear()
    _VP[key] = (poc, vah, val)
    return _VP[key]

def _pattern(df, k):
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body, rng = (c - o).abs(), h - l
    lo, up = np.minimum(o, c) - l, h - np.maximum(o, c)
    po, pc = o.shift(), c.shift()
    r = {"bull_candle": c > o, "bear_candle": c < o,
         "bull_engulf": (c > o) & (pc < po) & (c >= po) & (o <= pc),
         "bear_engulf": (c < o) & (pc > po) & (c <= po) & (o >= pc),
         "hammer": (lo >= 2 * body) & (up <= 0.25 * rng) & (rng > 0),
         "shooting_star": (up >= 2 * body) & (lo <= 0.25 * rng) & (rng > 0),
         "doji": (rng > 0) & (body <= 0.1 * rng),
         "inside_bar": (h < h.shift()) & (l > l.shift())}[k]
    return r.astype(float)

def ind(df, k, p, q=None):
    if k in ("open", "high", "low", "close", "volume"):
        return df[k].to_numpy()
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    p0 = p
    p = max(1, int(p or 14))
    if k == "ema": s = c.ewm(span=p, adjust=False).mean()
    elif k == "sma": s = c.rolling(p).mean()
    elif k == "vol_sma": s = v.rolling(p).mean()
    elif k == "rsi":
        d = c.diff()
        up = d.clip(lower=0).ewm(alpha=1 / p, adjust=False).mean()
        dn = (-d.clip(upper=0)).ewm(alpha=1 / p, adjust=False).mean()
        s = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    elif k == "atr":
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        s = tr.ewm(alpha=1 / p, adjust=False).mean()
    elif k == "bb_up": s = c.rolling(p).mean() + 2 * c.rolling(p).std()
    elif k == "bb_lo": s = c.rolling(p).mean() - 2 * c.rolling(p).std()
    elif k == "macd": s = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    elif k == "macd_sig":
        m = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
        s = m.ewm(span=9, adjust=False).mean()
    elif k == "hh": s = h.rolling(p).max().shift(1)   # highest high of previous p bars
    elif k == "ll": s = l.rolling(p).min().shift(1)
    elif k == "vwap":  # resets every UTC day
        day = _dk(df)
        tp = (h + l + c) / 3
        s = (tp * v).groupby(day).cumsum() / v.groupby(day).cumsum()
    elif k == "stoch":
        s = 100 * (c - l.rolling(p).min()) / (h.rolling(p).max() - l.rolling(p).min()).replace(0, np.nan)
    elif k == "cci":
        tp = (h + l + c) / 3
        dv = tp - tp.rolling(p).mean()
        s = dv / (0.015 * dv.abs().rolling(p).mean().replace(0, np.nan))
    elif k == "roc": s = c.pct_change(p) * 100
    elif k in ("adx", "plus_di", "minus_di"):
        up, dn = h.diff(), -l.diff()
        a = _tr(df).ewm(alpha=1 / p, adjust=False).mean()
        pdi = 100 * pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index).ewm(alpha=1 / p, adjust=False).mean() / a
        mdi = 100 * pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index).ewm(alpha=1 / p, adjust=False).mean() / a
        s = pdi if k == "plus_di" else mdi if k == "minus_di" else (100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)).ewm(alpha=1 / p, adjust=False).mean()
    elif k == "supertrend":  # ATR multiplier 3; bullish when close > line
        a = _tr(df).ewm(alpha=1 / p, adjust=False).mean()
        mid = (h + l) / 2
        ub, lb, cl = (mid + 3 * a).to_numpy(), (mid - 3 * a).to_numpy(), c.to_numpy()
        fu, fl, st, up = ub.copy(), lb.copy(), np.full(len(cl), np.nan), True
        for i in range(1, len(cl)):
            fu[i] = ub[i] if (ub[i] < fu[i - 1] or cl[i - 1] > fu[i - 1]) else fu[i - 1]
            fl[i] = lb[i] if (lb[i] > fl[i - 1] or cl[i - 1] < fl[i - 1]) else fl[i - 1]
            if up and cl[i] < fl[i]: up = False
            elif not up and cl[i] > fu[i]: up = True
            st[i] = fl[i] if up else fu[i]
        s = pd.Series(st)
    elif k in ("prev_high", "prev_low", "prev_close"): s = df[k[5:]].shift(1)
    elif k in ("pd_high", "pd_low", "pd_close"):
        day = _dk(df)
        dv = df.groupby(day)[k[3:]].agg({"pd_high": "max", "pd_low": "min", "pd_close": "last"}[k])
        s = day.map(dv.shift(1))
    elif k == "body_pct": s = (c - df["open"]).abs() / df["open"] * 100
    elif k == "range_pct": s = (h - l) / df["open"] * 100
    elif k in PATTERNS: s = _pattern(df, k)
    elif k in ("or_high", "or_low"):  # opening range: p = minutes, q = start hour (UTC, decimal)
        day = df["t"] // 86400
        ws = day * 86400 + float(q or 0) * 3600
        we = ws + (p0 or 30) * 60
        ob = (h if k == "or_high" else l).where((df["t"] >= ws) & (df["t"] < we))
        s = ob.groupby(day).transform("max" if k == "or_high" else "min").where(df["t"] >= we)
    elif k in ("vp_poc", "vp_vah", "vp_val"):  # previous p days' volume profile
        s = pd.Series(_vp(df, min(p, 30))[("vp_poc", "vp_vah", "vp_val").index(k)], index=df.index)
    elif k == "day_open": s = df["open"].groupby(_dk(df)).transform("first")
    elif k in ("day_high", "day_low"):  # today's high/low so far, excluding the current candle
        g = (h if k == "day_high" else l).groupby(_dk(df))
        s = (g.cummax() if k == "day_high" else g.cummin()).groupby(_dk(df)).shift(1)
    elif k in ("pw_high", "pw_low"):
        wk = (_dk(df) + 3) // 7
        dv = df.groupby(wk)[k[3:]].agg("max" if k == "pw_high" else "min")
        s = wk.map(dv.shift(1))
    elif k == "hour": s = ((df["t"] % 86400) // 3600).astype(float)
    elif k == "dow": s = ((df["t"] // 86400 + 3) % 7).astype(float)
    else:
        raise ValueError(f"unknown indicator {k}")
    return s.to_numpy()

def val(df, o, cache):
    if o["k"] == "num":
        return np.full(len(df), float(o["v"]))
    key = (o["k"], o.get("p"), o.get("q"))
    if key not in cache:
        cache[key] = ind(df, o["k"], o.get("p"), o.get("q"))
    return cache[key]

def rule(df, r, cache):
    a, b, op = val(df, r["l"], cache), val(df, r["r"], cache), r["op"]
    if op == ">": return a > b
    if op == "<": return a < b
    s = (a > b) & (np.roll(a, 1) <= np.roll(b, 1)) if op == "x_up" else (a < b) & (np.roll(a, 1) >= np.roll(b, 1))
    s[0] = False
    return s

def group(df, g, cache):
    rules = g.get("rules") or []
    if not rules:
        return None
    sig = [rule(df, r, cache) for r in rules]
    return np.logical_and.reduce(sig) if g.get("logic", "AND") == "AND" else np.logical_or.reduce(sig)

PANE = {"rsi": "RSI", "stoch": "Stochastic", "cci": "CCI", "roc": "ROC", "macd": "MACD", "macd_sig": "MACD", "adx": "ADX",
        "plus_di": "ADX", "minus_di": "ADX", "atr": "ATR", "body_pct": "Candle %", "range_pct": "Candle %"}
NOPLOT = {"close", "open", "high", "low", "volume", "num", "hour", "dow", "vol_sma"} | set(PATTERNS)

def overlays(df, items, limit=12):
    """Indicator series for the chart: price-scale ones go on the candle chart, oscillators get their own pane."""
    out, seen, t = [], set(), df["t"].tolist()
    for o in items:
        k = o.get("k")
        key = (k, o.get("p"), o.get("q"))
        if k in NOPLOT or k not in KIND_FLAG or key in seen or len(out) >= limit:
            continue
        seen.add(key)
        a = ind(df, k, o.get("p"), o.get("q"))
        label = k.upper() + (f" {o['p']}" if o.get("p") else "") + (f"@{o['q']:g}h" if o.get("q") is not None else "")
        out.append(dict(label=label, k=k, p=o.get("p"), q=o.get("q"), pane=PANE.get(k, "price"),
                        data=[{"time": t[i], "value": round(float(a[i]), 4)} if not np.isnan(a[i]) else {"time": t[i]} for i in range(len(t))]))
    return out

def _ov(df, q):
    try:
        ex = [e for e in (q.extra or []) if isinstance(e, dict) and e.get("k") in KIND_FLAG]
        items = [r[sd] for g in GROUPS for r in ((q.strategy.get(g) or {}).get("rules") or []) for sd in ("l", "r")] + ex
        return overlays(df, items)
    except Exception:
        return []

# ---------------- backtest ----------------
def backtest(df, s):
    cache = {}
    sig = {g: group(df, s.get(g) or {}, cache) for g in GROUPS}
    mode = s.get("direction", "long")
    o, h, l, c, t = (df[k].to_numpy() for k in ("open", "high", "low", "close", "t"))
    n = len(df)
    sl = float(s.get("sl_pct") or 0) / 100
    tp = float(s.get("tp_pct") or 0) / 100
    fee = float(s.get("fee_pct") or 0) / 100
    cap0 = float(s.get("capital") or 10000)
    size = float(s.get("size_pct") or 100) / 100 * float(s.get("leverage") or 1)
    eq, pos, ent, qty, ei = cap0, 0, 0.0, 0.0, 0
    eq0 = cap0
    entry_break, sl_candle = s.get("entry_mode") == "break", s.get("sl_mode") == "candle"
    rr = float(s.get("rr") or 1)
    stop_tp, max_td = bool(s.get("stop_after_tp")), int(s.get("max_trades_day") or 0)
    blocked, tcount, sl_px, tp_px = -1, {}, None, None
    dkv = _dk(df).to_numpy()
    equity, trades = np.full(n, cap0), []

    def close(i, px, why):
        nonlocal eq, pos, blocked
        gross = pos * qty * (px - ent)
        eq += gross - fee * qty * px
        pnl = gross - fee * qty * (ent + px)
        trades.append(dict(side="long" if pos > 0 else "short", t0=int(t[ei]), t1=int(t[i]),
                           entry=float(ent), exit=float(px), pnl=float(pnl), pct=float(pnl / (qty * ent) * 100), ret=float(pnl / eq0), why=why))
        pos = 0
        if why == "tp" and stop_tp:
            blocked = int(dkv[i])

    for i in range(1, n):
        if eq <= 0:
            equity[i:] = 0
            break
        el, es, xl, xs = sig["entry_long"], sig["entry_short"], sig["exit_long"], sig["exit_short"]
        wl = el is not None and mode in ("long", "both") and bool(el[i - 1])
        ws = es is not None and mode in ("short", "both") and bool(es[i - 1])
        rev = not entry_break
        if pos > 0 and ((xl is not None and xl[i - 1]) or (rev and ws and not wl)):
            close(i, o[i], "signal")
        elif pos < 0 and ((xs is not None and xs[i - 1]) or (rev and wl and not ws)):
            close(i, o[i], "signal")
        day = int(dkv[i])
        if pos == 0 and (wl or ws) and day != blocked and not (max_td and tcount.get(day, 0) >= max_td):
            side, px = 0, o[i]
            if not entry_break:
                side = 1 if wl else -1
            elif wl and h[i] > h[i - 1]:      # long: next candle breaks the signal candle high
                side, px = 1, max(o[i], h[i - 1])
            elif ws and l[i] < l[i - 1]:      # short: next candle breaks the signal candle low
                side, px = -1, min(o[i], l[i - 1])
            if side:
                if sl_candle:
                    sl_px = l[i - 1] if side > 0 else h[i - 1]
                    if abs(px - sl_px) <= 0:
                        side = 0
                    else:
                        tp_px = px + side * rr * abs(px - sl_px)
                else:
                    sl_px = px * (1 - side * sl) if sl else None
                    tp_px = px * (1 + side * tp) if tp else None
            if side:
                pos, ent, ei = side, px, i
                tcount[day] = tcount.get(day, 0) + 1
                eq0 = eq
                qty = eq * size / ent
                eq -= fee * qty * ent
        if pos:
            if pos > 0:
                if sl_px is not None and l[i] <= sl_px: close(i, min(sl_px, o[i]), "sl")
                elif tp_px is not None and h[i] >= tp_px: close(i, max(tp_px, o[i]), "tp")
            else:
                if sl_px is not None and h[i] >= sl_px: close(i, max(sl_px, o[i]), "sl")
                elif tp_px is not None and l[i] <= tp_px: close(i, min(tp_px, o[i]), "tp")
        equity[i] = eq + (pos * qty * (c[i] - ent) if pos else 0)
    if pos:
        close(n - 1, c[n - 1], "end")
        equity[-1] = eq
    return equity, trades, cap0

def _streak(trades, win):
    """Longest run of consecutive winning (or losing) trades and the P&L of that run."""
    best, cur, tot = (0, 0.0), 0, 0.0
    for x in trades:
        if (x["pnl"] > 0) == win:
            cur, tot = cur + 1, tot + x["pnl"]
            if cur > best[0]:
                best = (cur, tot)
        else:
            cur, tot = 0, 0.0
    return best

def metrics(df, equity, trades, tf, cap0):
    (mws, mws_pnl), (mls, mls_pnl) = _streak(trades, True), _streak(trades, False)
    r = np.diff(equity) / np.where(equity[:-1] == 0, np.nan, equity[:-1])
    sd = np.nanstd(r)
    sharpe = float(np.nanmean(r) / sd * math.sqrt(525600 / TF[tf])) if sd > 0 else 0.0
    w = [x["pnl"] for x in trades if x["pnl"] > 0]
    ls = [x["pnl"] for x in trades if x["pnl"] <= 0]
    pf = sum(w) / abs(sum(ls)) if ls and sum(ls) != 0 else (999.0 if w else 0.0)
    dd = float((equity / np.maximum.accumulate(equity) - 1).min()) * 100
    wp = [x["pct"] for x in trades if x["pnl"] > 0]
    lp = [abs(x["pct"]) for x in trades if x["pnl"] <= 0]
    aw = float(np.mean(wp)) if wp else 0.0
    al = float(np.mean(lp)) if lp else 0.0
    rr = aw / al if al > 0 else 0.0
    cl = df["close"].to_numpy()
    return dict(ret=round(float((equity[-1] / cap0 - 1) * 100), 2), bh=round(float((cl[-1] / cl[0] - 1) * 100), 2),
                dd=round(dd, 2), win=round(len(w) / len(trades) * 100, 1) if trades else 0.0,
                pf=round(min(pf, 999.0), 2), sharpe=round(sharpe, 2), trades=len(trades),
                avg=round(float(np.mean([x["pct"] for x in trades])), 3) if trades else 0.0,
                final=round(float(equity[-1]), 2), rr=round(rr, 2), aw=round(aw, 2), al=round(al, 2), nw=len(w), mws=mws, mws_pnl=round(mws_pnl, 2), mls=mls, mls_pnl=round(mls_pnl, 2))

# ---------------- api ----------------
class Req(BaseModel):
    symbol: str
    interval: str = "1h"
    candles: int = 3000
    strategy: dict
    start: str = ""
    end: str = ""
    extra: list = []

class OptReq(Req):
    params: list[dict]
    metric: str = "return_dd"
    min_trades: int = 10

def _dates(q):
    if not (q.start or q.end):
        return None, None
    try:
        a = int(pd.Timestamp(q.start, tz="UTC").timestamp() * 1000) if q.start else None
        b = int((pd.Timestamp(q.end, tz="UTC").timestamp() + 86400) * 1000) - 1 if q.end else None
    except Exception:
        raise HTTPException(400, "Bad date - use the date picker")
    return a, b

def prep(q):
    q.interval = {"60m": "1h", "240m": "4h", "1440m": "1d", "h": "1h", "hour": "1h", "d": "1d", "day": "1d", "daily": "1d"}.get(
        q.interval.strip().lower(), q.interval.strip().lower())
    raw = re.sub(r"[^A-Z0-9]", "", q.symbol.upper())
    a, b = _dates(q)
    if q.symbol.strip().upper().startswith("CSV:"):           # imported file
        if q.interval not in TF:
            raise HTTPException(400, f"Bad timeframe '{q.interval}'")
        return load_csv(re.sub(r"[^a-z0-9_]", "", q.symbol.strip()[4:].lower()), q.interval, max(200, min(q.candles, CSV_CAP)), a, b)
    if raw in YF:                                             # NQ, S&P 500 (ES), Gold
        if q.interval not in TF:
            raise HTTPException(400, f"Bad timeframe '{q.interval}'")
        return load_yahoo(YF[raw], q.interval, max(200, min(q.candles, 20000)), a, b)
    sym = raw
    if sym and len(sym) <= 5 and not sym.endswith(("USDT", "USDC", "BTC")):
        sym += "USDT"                                         # "BTC" -> BTCUSDT
    if q.interval not in TF or not re.fullmatch(r"[A-Z0-9]{5,20}", sym):
        raise HTTPException(400, f"Bad symbol or interval (got symbol='{q.symbol}', interval='{q.interval}'). Use a pair like BTCUSDT (or NQ, ES, GOLD) and a timeframe like 1h")
    if a is not None or b is not None:
        return load_range(sym, q.interval, a, b)
    return load(sym, q.interval, max(200, min(q.candles, 20000)))

@app.post("/api/backtest")
def api_backtest(q: Req):
    df = prep(q)
    try:
        equity, trades, cap0 = backtest(df, q.strategy)
    except Exception as e:
        raise HTTPException(400, f"Strategy error: {e!r}")
    t = df["t"].tolist()
    return dict(
        candles=[dict(time=t[i], open=df.open[i], high=df.high[i], low=df.low[i], close=df.close[i]) for i in range(len(df))],
        equity=[dict(time=t[i], value=round(float(equity[i]), 2)) for i in range(len(df))],
        trades=trades, metrics=metrics(df, equity, trades, q.interval, cap0), overlays=_ov(df, q))

def set_path(d, path, v):
    a = path.split(".")
    for k in a[:-1]:
        d = d[int(k)] if isinstance(d, list) else d[k]
    if isinstance(d, list): d[int(a[-1])] = v
    else: d[a[-1]] = v

@app.post("/api/optimize")
def api_optimize(q: OptReq):
    df = prep(q)
    if not q.params:
        raise HTTPException(400, "Add at least one parameter")
    try:
        ranges = [np.arange(p["start"], p["end"] + p["step"] / 2, p["step"]) for p in q.params]
    except Exception:
        raise HTTPException(400, "Bad parameter range")
    total = math.prod(len(r) for r in ranges)
    lim = max(20, min(1500, 6_000_000 // len(df)))      # fewer combos when there is a lot of data
    if total > lim:
        raise HTTPException(400, f"{total} combinations - the limit for {len(df)} candles is {lim}. Reduce the range, increase the step, or use fewer candles.")
    L = [len(r) for r in ranges]
    fmt = lambda p, v: int(round(v)) if p["path"].endswith(".p") else round(float(v), 4)
    two = len(q.params) == 2
    Z = np.full((L[0], L[1]), np.nan) if two else None
    out = []
    for idx in itertools.product(*[range(k) for k in L]):
        s, vals = copy.deepcopy(q.strategy), []
        for p, r, i in zip(q.params, ranges, idx):
            v = fmt(p, r[i])
            set_path(s, p["path"], v)
            vals.append(v)
        try:
            eq, tr, cap0 = backtest(df, s)
        except Exception as e:
            raise HTTPException(400, f"Strategy error: {e!r}")
        m = metrics(df, eq, tr, q.interval, cap0)
        if m["trades"] < q.min_trades:
            continue
        score = {"return": m["ret"], "sharpe": m["sharpe"], "pf": m["pf"]}.get(q.metric, m["ret"] / max(abs(m["dd"]), 1))
        if two:
            Z[idx] = score
        out.append(dict(params=vals, m=m, score=score))
    out.sort(key=lambda x: -x["score"])
    res = dict(total=total, kept=len(out), results=out[:30])
    if two and out:  # plateau analysis: average score of each 3x3 neighbourhood
        SM = np.full(Z.shape, np.nan)
        for i in range(L[0]):
            for j in range(L[1]):
                if np.isnan(Z[i, j]):
                    continue
                w = Z[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2]
                if np.sum(~np.isnan(w)) >= 5:
                    SM[i, j] = np.nanmean(w)
        cl = lambda M: [[None if np.isnan(x) else round(float(x), 3) for x in row] for row in M]
        b = np.unravel_index(np.nanargmax(Z), Z.shape)
        pl = None if np.all(np.isnan(SM)) else np.unravel_index(np.nanargmax(SM), SM.shape)
        res["grid"] = dict(xs=[fmt(q.params[0], v) for v in ranges[0]], ys=[fmt(q.params[1], v) for v in ranges[1]],
                           z=cl(Z), s=cl(SM), best=[int(b[0]), int(b[1])],
                           plat=None if pl is None else [int(pl[0]), int(pl[1])])
    return res

class MCReq(Req):
    sims: int = 1000
    method: str = "bootstrap"
    ruin_pct: float = 50.0

@app.post("/api/montecarlo")
def api_mc(q: MCReq):
    df = prep(q)
    try:
        _, trades, cap0 = backtest(df, q.strategy)
    except Exception as e:
        raise HTTPException(400, f"Strategy error: {e!r}")
    if len(trades) < 10:
        raise HTTPException(400, f"Only {len(trades)} trades — Monte Carlo needs at least 10")
    r = np.array([t["ret"] for t in trades])
    n, S = len(r), max(100, min(q.sims, 5000))
    rng = np.random.default_rng(42)
    idx = np.argsort(rng.random((S, n)), axis=1) if q.method == "shuffle" else rng.integers(0, n, (S, n))
    paths = np.concatenate([np.ones((S, 1)), np.cumprod(1 + r[idx], axis=1)], axis=1) * cap0
    dd = (paths / np.maximum.accumulate(paths, axis=1) - 1).min(axis=1) * 100
    fin = (paths[:, -1] / cap0 - 1) * 100
    pc = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
    act = cap0 * np.concatenate([[1.0], np.cumprod(1 + r)])
    rd = lambda a: [round(float(x), 2) for x in a]
    adn = float((act / np.maximum.accumulate(act) - 1).min() * 100)
    return dict(n=n, sims=S, method=q.method, ruin_pct=q.ruin_pct, actual=rd(act),
                bands={k: rd(v) for k, v in zip(("p5", "p25", "p50", "p75", "p95"), pc)},
                stats=dict(med_ret=round(float(np.median(fin)), 2), p5_ret=round(float(np.percentile(fin, 5)), 2),
                           p95_ret=round(float(np.percentile(fin, 95)), 2), prob_profit=round(float((fin > 0).mean() * 100), 1),
                           med_dd=round(float(np.median(dd)), 2), worst_dd=round(float(np.percentile(dd, 5)), 2),
                           ruin=round(float((dd <= -q.ruin_pct).mean() * 100), 1),
                           actual_ret=round(float((act[-1] / cap0 - 1) * 100), 2), actual_dd=round(adn, 2)))

# ---------------- chat -> strategy ----------------
DEFAULT_P = {"ema": 20, "sma": 20, "rsi": 14, "atr": 14, "adx": 14, "plus_di": 14, "minus_di": 14, "stoch": 14, "cci": 20,
             "roc": 10, "supertrend": 10, "bb_up": 20, "bb_lo": 20, "hh": 20, "ll": 20, "vol_sma": 20}
ALIAS = [("previous day high", "pd_high"), ("previous day low", "pd_low"), ("previous day close", "pd_close"),
         ("prev day high", "pd_high"), ("prev day low", "pd_low"), ("prev day close", "pd_close"),
         ("yesterday high", "pd_high"), ("yesterday low", "pd_low"), ("yesterday close", "pd_close"),
         ("previous high", "prev_high"), ("previous low", "prev_low"), ("previous close", "prev_close"),
         ("prev high", "prev_high"), ("prev low", "prev_low"), ("prev close", "prev_close"),
         ("upper bollinger", "bb_up"), ("bollinger upper", "bb_up"), ("upper band", "bb_up"),
         ("lower bollinger", "bb_lo"), ("bollinger lower", "bb_lo"), ("lower band", "bb_lo"),
         ("macd signal", "macd_sig"), ("signal line", "macd_sig"), ("moving average", "sma"), ("volume sma", "vol_sma"),
         ("stochastic", "stoch"), ("stoch", "stoch"), ("supertrend", "supertrend"), ("vwap", "vwap"), ("macd", "macd"),
         ("ema", "ema"), ("sma", "sma"), ("rsi", "rsi"), ("atr", "atr"), ("adx", "adx"), ("cci", "cci"), ("roc", "roc"),
         ("price", "close"), ("close", "close"), ("open", "open"), ("high", "high"), ("low", "low"),
         ("volume", "volume"), ("hour", "hour"), ("hh", "hh"), ("ll", "ll")]
DEFAULT_P.update({"vp_poc": 1, "vp_vah": 1, "vp_val": 1, "or_high": 30, "or_low": 30})
ALIAS += [("opening range high", "or_high"), ("opening range low", "or_low"), ("orb high", "or_high"), ("orb low", "or_low"),
          ("or high", "or_high"), ("or low", "or_low"), ("point of control", "vp_poc"), ("poc", "vp_poc"),
          ("value area high", "vp_vah"), ("vah", "vp_vah"), ("value area low", "vp_val"), ("val", "vp_val"),
          ("day open", "day_open"), ("today open", "day_open"), ("day high", "day_high"), ("today high", "day_high"),
          ("day low", "day_low"), ("today low", "day_low"), ("previous week high", "pw_high"), ("previous week low", "pw_low"),
          ("last week high", "pw_high"), ("last week low", "pw_low"), ("pw high", "pw_high"), ("pw low", "pw_low")]
ALIASD = dict(ALIAS)
PHR = {"bullish engulfing": "bull_engulf", "bearish engulfing": "bear_engulf", "hammer": "hammer",
       "shooting star": "shooting_star", "doji": "doji", "inside bar": "inside_bar", "bullish candle": "bull_candle",
       "green candle": "bull_candle", "bearish candle": "bear_candle", "red candle": "bear_candle"}
_ALT = "|".join(re.escape(a) for a, _ in sorted(ALIAS, key=lambda x: -len(x[0])))
_OPD = r"(?:(?<![a-z_])(?:%s)(?![a-z_])|-?\d+(?:\.\d+)?)" % _ALT
_OPR = (r"(?:(?:cross(?:es|ed|ing)?|break(?:s|ing)?)\s+(?:above|over|up(?:\s+through)?|upwards?|out above)"
        r"|(?:cross(?:es|ed|ing)?|break(?:s|ing)?)\s+(?:below|under|down(?:\s+through)?|downwards?)"
        r"|(?:is\s+|goes\s+|closes\s+|close\s+|stays\s+)?(?:above|greater than|more than|higher than|over|>)"
        r"|(?:is\s+|goes\s+|closes\s+|close\s+|stays\s+)?(?:below|less than|lower than|under|<))")
_CL = re.compile(r"(?P<a>%s)(?:\s*\(?\s*(?P<ap>\d+)\s*\)?)?\s+(?P<op>%s)\s+(?P<b>%s)(?:\s*\(?\s*(?P<bp>\d+)\s*\)?)?" % (_OPD, _OPR, _OPD))
_EL = re.compile(r"(?:\band\b|\bor\b|,)\s*(?P<op>%s)\s+(?P<b>%s)(?:\s*\(?\s*(?P<bp>\d+)\s*\)?)?" % (_OPR, _OPD))
_MARK = re.compile(r"\b(exit|square off|close (?:the )?(?:long|short|position|trade)|entry|enter|buy|sell|short|long)\b")
OPTXT = {">": "is above", "<": "is below", "x_up": "crosses above", "x_dn": "crosses below"}
HELP = ("Mujhe koi rule samajh nahi aaya. Is tarah likho:\n"
        "“long entry when ema 9 crosses above ema 21 and rsi 14 below 70; exit when rsi above 80; sl 1% tp 2%”\n"
        "“short when close crosses below supertrend; 15m ethusdt”\n"
        "“buy on bullish engulfing, trade only between 8 and 20 utc”\n"
        "“close above opening range high; 15 min opening range from 13:30 utc” (5m timeframe ke saath)\n"
        "“long when close crosses above poc and above previous day high” (volume profile: poc, vah, val)\n"
        "“long when green candle and close below vwap, enter on break of signal candle, sl at signal candle, rr 1:1, stop trading for the day after first target” (candle-based SL/RR)\n"
        "“new strategy” likhne se sab reset hota hai, “add …” likhne se rule jud jata hai.")

def ensure(st):
    out = {g: {"logic": "AND", "rules": []} for g in GROUPS}
    out.update({"direction": "long", "sl_pct": 1, "tp_pct": 2, "fee_pct": 0.05, "capital": 10000, "size_pct": 100, "leverage": 1})
    for k, v in (st or {}).items():
        if k in GROUPS:
            if isinstance(v, dict):
                out[k] = {"logic": "OR" if v.get("logic") == "OR" else "AND", "rules": list(v.get("rules") or [])}
        else:
            out[k] = v
    return out

def _co(o):
    k = o["k"]
    if k not in KIND_FLAG:
        raise ValueError(f"unknown indicator {k}")
    if k == "num":
        return {"k": "num", "v": float(o["v"])}
    if KIND_FLAG[k] == 3:
        return {"k": k, "p": max(1, min(int(o.get("p") or 30), 1440)), "q": max(0.0, min(float(o.get("q") or 0), 23.99))}
    return {"k": k, "p": max(1, min(int(o.get("p") or DEFAULT_P.get(k, 14)), 500))} if KIND_FLAG[k] == 1 else {"k": k}

def clean_strategy(st, cur):
    out, st = ensure(cur), st or {}
    for g in GROUPS:
        gg = st.get(g)
        if isinstance(gg, dict):
            rules = []
            for r in gg.get("rules") or []:
                if r["op"] not in OPTXT:
                    raise ValueError(f"bad operator {r['op']}")
                rules.append({"l": _co(r["l"]), "op": r["op"], "r": _co(r["r"])})
            out[g] = {"logic": "OR" if gg.get("logic") == "OR" else "AND", "rules": rules}
    if st.get("direction") in ("long", "short", "both"):
        out["direction"] = st["direction"]
    if st.get("sl_mode") in ("pct", "candle"): out["sl_mode"] = st["sl_mode"]
    if st.get("entry_mode") in ("next_open", "break"): out["entry_mode"] = st["entry_mode"]
    if "rr" in st: out["rr"] = max(0.1, min(float(st["rr"]), 20.0))
    if "stop_after_tp" in st: out["stop_after_tp"] = bool(st["stop_after_tp"])
    if "max_trades_day" in st: out["max_trades_day"] = max(0, min(int(st["max_trades_day"]), 100))
    for k, hi in (("sl_pct", 100), ("tp_pct", 1000), ("fee_pct", 5), ("size_pct", 100), ("leverage", 100), ("capital", 1e9)):
        if k in st:
            out[k] = max(0.0, min(float(st[k]), hi))
    return out

def _dsc(o):
    return f"{o['v']:g}" if o["k"] == "num" else o["k"].upper() + (f" {o['p']}" if "p" in o else "") + (f" @{o['q']:g}h UTC" if "q" in o else "")

PNAME = {"bull_candle": "Green candle", "bear_candle": "Red candle", "bull_engulf": "Bullish engulfing", "bear_engulf": "Bearish engulfing",
         "hammer": "Hammer", "shooting_star": "Shooting star", "doji": "Doji", "inside_bar": "Inside bar"}

def _rtxt(r):
    if r["l"]["k"] in PNAME and r["op"] == ">" and r["r"]["k"] == "num" and r["r"]["v"] == 0.5:
        return f"{PNAME[r['l']['k']]} formed"
    return f"{_dsc(r['l'])} {OPTXT[r['op']]} {_dsc(r['r'])}"

def _dg(g):
    return (" OR " if g["logic"] == "OR" else " AND ").join(_rtxt(r) for r in g["rules"])

def _opd(tok, per, ctx):
    tok = tok.strip()
    if re.fullmatch(r"-?\d+(?:\.\d+)?", tok):
        return {"k": "num", "v": float(tok)}
    k = ALIASD[tok]
    if KIND_FLAG[k] == 3:
        return {"k": k, "p": ctx["or"][0], "q": ctx["or"][1]}
    if k.startswith("vp_"):
        return {"k": k, "p": ctx["vp"]}
    return {"k": k, "p": int(per) if per else DEFAULT_P.get(k, 14)} if KIND_FLAG[k] == 1 else {"k": k}

def _op(s):
    if re.search(r"cross|break", s):
        return "x_dn" if re.search(r"below|under|down", s) else "x_up"
    return "<" if re.search(r"below|less|lower|under|<", s) else ">"

def _prep(t):
    t = t.replace("'s ", " ").replace("’s ", " ")
    t = re.sub(r"(\d+)[- ]?(?:bar|candle|period|day)s?\s+high(?:est)?", r"hh \1", t)
    t = re.sub(r"(\d+)[- ]?(?:bar|candle|period|day)s?\s+low(?:est)?", r"ll \1", t)
    t = re.sub(r"highest high(?: of| over)?\s+(\d+)", r"hh \1", t)
    t = re.sub(r"lowest low(?: of| over)?\s+(\d+)", r"ll \1", t)
    t = re.sub(r"\b(ema|sma|rsi|atr|adx|cci|roc|stoch|supertrend)(\d+)", r"\1 \2", t)
    t = re.sub(r"\b(\d+)[- ]?(ema|sma|rsi)\b", r"\2 \1", t)
    return t.replace("%", " ").replace("≥", ">").replace("≤", "<")

def basic_parse(msg, cur):
    t = _prep(msg.lower())
    st, changed, notes = ensure(cur), False, []
    if re.search(r"\b(new strategy|reset|start over|from scratch|clear all)\b", t):
        for g in GROUPS:
            st[g] = {"logic": "AND", "rules": []}
        for k in ("sl_mode", "entry_mode", "rr", "stop_after_tp", "max_trades_day"):
            st.pop(k, None)
        changed = True
    ctx = {"or": (30, 0.0), "vp": 1}
    if re.search(r"opening range|\borb\b", t):
        md = re.search(r"(\d+)\s*(?:min(?:ute)?s?)\b", t)
        ms = re.search(r"(?:from|at|starting|start)\s+(\d{1,2})(?::(\d\d))?\s*(utc|gmt|ist|et|est)?\b", t)
        hr = 0.0
        if ms:
            hr = int(ms.group(1)) + int(ms.group(2) or 0) / 60 - (5.5 if ms.group(3) == "ist" else -5 if ms.group(3) in ("et", "est") else 0)
        ctx["or"] = (int(md.group(1)) if md else 30, round(hr % 24, 2))
        notes.append(f"Opening range: {ctx['or'][0]} min from {int(ctx['or'][1])}:{round(ctx['or'][1] % 1 * 60):02d} UTC (use timeframe ≤ range length)")
    mv = re.search(r"(?:last|past|previous|prev)\s+(\d+)\s+days?", t)
    if mv:
        ctx["vp"] = max(1, min(int(mv.group(1)), 30))
    ev, spans = [], []
    for m in _CL.finditer(t):
        ev.append((m.start(), "c", m)); spans.append((m.start(), m.end()))
    for ph, key in PHR.items():
        for m in re.finditer(re.escape(ph), t):
            ev.append((m.start(), "p", key)); spans.append((m.start(), m.end()))
    for m in _EL.finditer(t):  # "... and below X": reuse the previous left operand
        if not any(a <= m.start() < b or m.start() <= a < m.end() for a, b in spans):
            ev.append((m.start(), "e", m)); spans.append((m.start(), m.end()))
    for m in _MARK.finditer(t):
        if not any(a <= m.start() < b for a, b in spans):
            ev.append((m.start(), "m", m.group(1)))
    for m in re.finditer(r"[.;\n](?!\d)", t):
        ev.append((m.start(), "b", None))
    ev.sort(key=lambda e: e[0])
    new, side, ex, last = {g: [] for g in GROUPS}, "long", False, None
    for pos, kd, v in ev:
        if kd == "m":
            sd = "short" if v in ("short", "sell") or v.endswith("short") else "long" if v in ("long", "buy") or v.endswith("long") else None
            if t[max(0, pos - 4):pos] == "for " and last and sd:  # "... for short" -> move previous rule
                g0, r0 = last
                new[g0].remove(r0)
                last = (f"{g0.split('_')[0]}_{sd}", r0)
                new[last[0]].append(r0)
            elif v.startswith(("exit", "square", "close")):
                ex, side = True, sd or side
            elif v in ("entry", "enter"):
                ex = False
            else:
                side = sd or side
                ex = False if v in ("buy", "sell") else ex
        elif kd == "b":
            ex = False
        elif kd == "e":
            if last:
                r0 = {"l": dict(last[1]["l"]), "op": _op(v["op"]), "r": _opd(v["b"], v["bp"], ctx)}
                new[last[0]].append(r0)
                last = (last[0], r0)
        else:
            g = f"{'exit' if ex else 'entry'}_{side}"
            r0 = ({"l": _opd(v["a"], v["ap"], ctx), "op": _op(v["op"]), "r": _opd(v["b"], v["bp"], ctx)} if kd == "c"
                  else {"l": {"k": v}, "op": ">", "r": {"k": "num", "v": 0.5}})
            new[g].append(r0)
            last = (g, r0)
    use_or = bool(re.search(r"\bor\b", t)) and not re.search(r"\band\b", t)
    add = bool(re.search(r"\b(add|also|plus|additionally)\b", t))
    label = {"entry_long": "Long entry", "exit_long": "Long exit", "entry_short": "Short entry", "exit_short": "Short exit"}
    for g in GROUPS:
        if new[g]:
            st[g] = {"logic": "OR" if use_or else "AND", "rules": (st[g]["rules"] if add else []) + new[g]}
            notes.append(f"{label[g]}: {_dg(st[g])}"); changed = True
    tm = re.search(r"(?:between|from)\s+(\d{1,2})(?::\d\d)?\s*(?:and|to|-)\s*(\d{1,2})(?::\d\d)?\s*(?:utc|gmt|hours?|hrs?|session)", t)
    if tm:
        a, b = int(tm.group(1)), int(tm.group(2))
        for g in ("entry_long", "entry_short"):
            if st[g]["rules"] and st[g]["logic"] == "AND" and not any(r["l"]["k"] == "hour" for r in st[g]["rules"]):
                st[g]["rules"] += [{"l": {"k": "hour"}, "op": ">", "r": {"k": "num", "v": a - 0.5}},
                                   {"l": {"k": "hour"}, "op": "<", "r": {"k": "num", "v": b - 0.5}}]
        notes.append(f"Trade only {a}:00–{b}:00 UTC"); changed = True
    d = None
    if re.search(r"(long only|only long|buy only|only buy)", t): d = "long"
    elif re.search(r"(short only|only short|sell only|only sell)", t): d = "short"
    elif re.search(r"\b(both|long and short|long/short|long \+ short|both sides)\b", t): d = "both"
    elif new["entry_long"] or new["entry_short"]:
        hl, hs = bool(st["entry_long"]["rules"]), bool(st["entry_short"]["rules"])
        d = "both" if hl and hs else "short" if hs else "long"
    if d and d != st["direction"]:
        st["direction"] = d; notes.append(f"Direction: {d}"); changed = True
    spec = (("sl_mode", "candle", r"(?:sl|stop ?loss)\s+(?:at|below|above|on)\s+(?:the\s+)?(?:signal|setup|entry|trigger)\s+candle", "Stop-loss: signal candle low (long) / high (short)"),
            ("entry_mode", "break", r"(?:enter|entry)?\s*(?:on\s+)?break(?:out)?\s+of\s+(?:the\s+)?(?:signal|setup)\s+candle", "Entry: when the next candle breaks the signal candle high (long) / low (short)"),
            ("stop_after_tp", True, r"stop\w*\s+(?:trading\s+)?(?:for\s+the\s+day\s+)?after\s+(?:the\s+)?(?:first\s+|1\s+|one\s+)?(?:tp|target)", "Stop trading for the day after a take-profit"))
    for key, val, pat, note in spec:
        if re.search(pat, t):
            st[key] = val; notes.append(note); changed = True
            t = re.sub(pat, " ", t)
    mt = re.search(r"max(?:imum)?\s+(\d+)\s+trades?\s+(?:per|a|each)\s+day", t)
    if mt:
        st["max_trades_day"] = int(mt.group(1)); notes.append(f"Max trades per day: {mt.group(1)}"); changed = True
        t = t.replace(mt.group(0), " ")
    num = lambda p: (lambda m: float(m.group(1)) if m else None)(re.search(p, t))
    for key, off, pat, nm in (("sl_pct", r"(?:no|without|remove|disable)\s+(?:the\s+)?(?:sl|stop ?loss)", r"(?:\bsl\b|stop ?loss)\D{0,15}?(\d+(?:\.\d+)?)", "SL"),
                              ("tp_pct", r"(?:no|without|remove|disable)\s+(?:the\s+)?(?:tp|take ?profit|target)", r"(?:\btp\b|take ?profit|target)\D{0,15}?(\d+(?:\.\d+)?)", "TP")):
        if re.search(off, t): st[key] = 0; notes.append(f"{nm}: off"); changed = True
        elif num(pat) is not None: st[key] = num(pat); notes.append(f"{nm}: {st[key]:g}%"); changed = True
    rr = re.search(r"\b(?:rr|risk[ -]?reward)\D{0,6}(\d+(?:\.\d+)?)(?:\s*[:/]\s*(\d+(?:\.\d+)?))?", t)
    if rr and st.get("sl_mode") == "candle":
        st["rr"] = float(rr.group(2)) / float(rr.group(1)) if rr.group(2) else float(rr.group(1))
        notes.append(f"Risk-reward: 1:{st['rr']:g}"); changed = True
    elif rr and st["sl_pct"]:
        st["tp_pct"] = round(st["sl_pct"] * (float(rr.group(2)) / float(rr.group(1)) if rr.group(2) else float(rr.group(1))), 3)
        notes.append(f"TP: {st['tp_pct']:g}% (risk-reward)"); changed = True
    for key, pat, nm in (("fee_pct", r"(?:fees?|commission)\D{0,15}?(\d+(?:\.\d+)?)", "Fee"), ("capital", r"(?:capital|balance)\D{0,15}?(\d+(?:\.\d+)?)", "Capital"),
                         ("size_pct", r"(?:position size|size)\D{0,15}?(\d+(?:\.\d+)?)", "Size")):
        if num(pat) is not None: st[key] = num(pat); notes.append(f"{nm}: {st[key]:g}"); changed = True
    lv = re.search(r"leverage\D{0,10}?(\d+)|\b(\d+)\s*x\b", t)
    if lv: st["leverage"] = float(lv.group(1) or lv.group(2)); notes.append(f"Leverage: {st['leverage']:g}x"); changed = True
    sym = iv = None
    ms = re.search(r"\b([a-z0-9]{2,10}(?:usdt|usdc|btc))\b", t)
    coins = {"btc": "BTCUSDT", "bitcoin": "BTCUSDT", "eth": "ETHUSDT", "ethereum": "ETHUSDT", "sol": "SOLUSDT", "solana": "SOLUSDT",
             "bnb": "BNBUSDT", "xrp": "XRPUSDT", "doge": "DOGEUSDT",
             "nq": "NQ", "nasdaq": "NQ", "snp": "ES", "sp500": "ES", "spx": "ES", "s&p": "ES", "gold": "GOLD", "xau": "GOLD", "xauusd": "GOLD"}
    mc = re.search(r"\b(btc|bitcoin|eth|ethereum|sol|solana|bnb|xrp|doge|nq|nasdaq|snp|sp500|spx|s&p|gold|xau|xauusd)\b", t)
    sym = ms.group(1).upper() if ms else coins[mc.group(1)] if mc else None
    mi = re.search(r"\b(1m|3m|5m|15m|30m|1h|2h|4h|1d)\b", t)
    iv = mi.group(1) if mi else "1d" if "daily" in t else "1h" if "hourly" in t else None
    if sym: notes.append(f"Symbol: {sym}"); changed = True
    if iv: notes.append(f"Timeframe: {iv}"); changed = True
    reply = ("Done ✅\n" + "\n".join(notes) + "\nRules left side mein dekh lo, backtest chal raha hai.") if changed else HELP
    return dict(reply=reply, strategy=st, symbol=sym, interval=iv, changed=changed, mode="basic")

_AI_CALLS = []

def _ai_ok():
    now = time.time()
    _AI_CALLS[:] = [x for x in _AI_CALLS if now - x < 3600]
    if len(_AI_CALLS) >= 40:
        return False
    _AI_CALLS.append(now)
    return True

def _ai_sys():
    cat = "; ".join(f"{g}: " + ", ".join(k + ("(p)" if v == 1 else "(p,q)" if v == 3 else "") for k, v in d.items()) for g, d in CATALOG.items())
    return ("You convert a trader's plain-language request into a strategy JSON for a backtesting app. Return ONLY one JSON object, no markdown: "
            '{"reply": short friendly explanation in the user\'s language (Hinglish is fine), "symbol": Binance pair like BTCUSDT or null, '
            '"interval": one of 1m,3m,5m,15m,30m,1h,2h,4h,1d or null, "strategy": the FULL updated strategy}. '
            'Strategy keys: direction (long|short|both); entry_long, exit_long, entry_short, exit_short, each {"logic":"AND"|"OR","rules":[{"l":operand,"op":op,"r":operand}]}; '
            'sl_pct and tp_pct (percent, 0=off); fee_pct; size_pct; leverage; capital; optional sl_mode (pct or candle: stop at the signal candle low for longs and high for shorts), rr (take-profit = rr x risk when sl_mode is candle), entry_mode (next_open or break: enter when the candle after the signal candle breaks its high for longs and low for shorts), stop_after_tp (true: no more trades that UTC day after a take-profit), max_trades_day (0=off). Operand: {"k":kind,"p":period} (p only for kinds marked (p)) or {"k":"num","v":number}. '
            'op is one of ">", "<", "x_up" (crosses above), "x_dn" (crosses below). Available kinds - ' + cat + ". "
            'Candle patterns (bull_candle, bear_candle, bull_engulf, bear_engulf, hammer, shooting_star, doji, inside_bar) are 1/0 series: use {"l":{"k":"hammer"},"op":">","r":{"k":"num","v":0.5}}. '
            "hour is the UTC hour 0-23 (hours 8 to 19 means hour > 7.5 AND hour < 19.5); dow 0=Monday. pd_* are previous UTC day values, prev_* the previous candle; "
            "hh/ll are the highest high / lowest low of the previous p candles (breakout: close > hh). supertrend is bullish when close > supertrend. "
            "or_high/or_low = opening range high/low: p = range length in minutes, q = start hour in UTC (decimal, e.g. 13.5 = 13:30; 9:15 IST = 3.75); valid only after the range completes each UTC day; the timeframe should not be larger than p. "
            "vp_poc/vp_vah/vp_val = point of control / value-area high / value-area low (70%) of the volume profile built from the previous p days. day_open/day_high/day_low are today's values so far; pw_* previous week. "
            "Entries fill at the next candle open. When the user asks to change or add something, keep the rest of the current strategy; if they ask for a new strategy, replace everything. "
            "Exits are optional (SL/TP can be the only exit). If something cannot be expressed with these kinds, use the closest and say so in reply.")

def ai_chat(msg, cur, sym, tf, key):
    r = requests.post("https://api.anthropic.com/v1/messages", timeout=45,
                      headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                      json={"model": os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5"), "max_tokens": 1500, "system": _ai_sys(),
                            "messages": [{"role": "user", "content": f"Current strategy JSON:\n{json.dumps(cur)}\nCurrent market: {sym} {tf}\n\nUser request: {msg}"}]})
    r.raise_for_status()
    txt = "".join(b.get("text", "") for b in r.json()["content"])
    j = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
    s = str(j.get("symbol") or "").upper()
    return dict(reply=str(j.get("reply", ""))[:700], strategy=clean_strategy(j["strategy"], cur),
                symbol=s if re.fullmatch(r"[A-Z0-9]{5,20}", s) else None,
                interval=j.get("interval") if j.get("interval") in TF else None, changed=True, mode="ai")

class ChatReq(BaseModel):
    message: str
    strategy: dict = {}
    symbol: str = ""
    interval: str = ""

@app.post("/api/chat")
def api_chat(q: ChatReq):
    msg = q.message.strip()[:1000]
    if not msg:
        raise HTTPException(400, "Empty message")
    cur, key, note = ensure(q.strategy), os.environ.get("ANTHROPIC_API_KEY"), ""
    if key:
        if _ai_ok():
            try:
                return ai_chat(msg, cur, q.symbol, q.interval, key)
            except Exception as e:
                note = f"(AI error: {str(e)[:80]} — basic parser use hua)\n"
        else:
            note = "(AI hourly limit reached — basic parser use hua)\n"
    try:
        out = basic_parse(msg, cur)
    except Exception as e:
        raise HTTPException(400, f"Could not understand: {e!r}")
    out["reply"] = note + out["reply"]
    return out

# ---------------- GEX (gamma exposure) from Deribit options ----------------
GEXC = {}

def _gamma(S, K, sig, T):
    d1 = (np.log(S / K) + 0.5 * sig * sig * T) / (sig * np.sqrt(T))
    return np.exp(-0.5 * d1 * d1) / np.sqrt(2 * np.pi) / (S * sig * np.sqrt(T))

@app.get("/api/gex")
def api_gex(currency: str = "BTC", days: int = 30):
    """Dealer gamma exposure per strike from Deribit open interest. Calls count +, puts count - (standard assumption)."""
    cur, days = currency.upper(), max(1, min(days, 365))
    if cur not in ("BTC", "ETH"):
        raise HTTPException(400, "GEX is available for BTC and ETH only")
    c = GEXC.get((cur, days))
    if c and time.time() - c[0] < 300:
        return c[1]
    try:
        r = requests.get("https://www.deribit.com/api/v2/public/get_book_summary_by_currency",
                         params={"currency": cur, "kind": "option"}, timeout=25)
        r.raise_for_status()
        rows = r.json()["result"]
    except Exception as e:
        raise HTTPException(502, f"Deribit data error: {str(e)[:120]}")
    now = datetime.now(timezone.utc)
    K, T, IV, OI, SG, SP = [], [], [], [], [], []
    for x in rows:
        try:
            _, ex, k, ty = x["instrument_name"].split("-")
            exp = datetime.strptime(ex, "%d%b%y").replace(hour=8, tzinfo=timezone.utc)
            t = (exp - now).total_seconds() / 31536000
            iv, oi, sp = (x.get("mark_iv") or 0) / 100, x.get("open_interest") or 0, x.get("underlying_price")
            if t <= 0 or t * 365 > days or iv <= 0 or oi <= 0 or not sp:
                continue
            K.append(float(k)); T.append(t); IV.append(iv); OI.append(float(oi)); SG.append(1.0 if ty == "C" else -1.0); SP.append(float(sp))
        except Exception:
            continue
    if len(K) < 5:
        raise HTTPException(502, "Not enough option data from Deribit for this range")
    K, T, IV, OI, SG, SP = (np.array(a) for a in (K, T, IV, OI, SG, SP))
    spot = float(np.median(SP[T <= T.min() * 1.5 + 1e-9]))

    def gex(S):  # per-option dollar gamma per 1% move if the underlying were at S
        f = SP * (S / spot)
        return SG * _gamma(f, K, IV, T) * OI * S * S * 0.01

    g0 = gex(spot)
    strikes = sorted(set(K.tolist()))
    call = {k: float(g0[(K == k) & (SG > 0)].sum()) for k in strikes}
    put = {k: float(-g0[(K == k) & (SG < 0)].sum()) for k in strikes}  # magnitude
    up = [k for k in strikes if k >= spot] or strikes
    dn = [k for k in strikes if k <= spot] or strikes
    call_wall, put_wall = max(up, key=lambda k: call[k]), max(dn, key=lambda k: put[k])
    grid = spot * np.linspace(0.7, 1.3, 121)
    tot = np.array([gex(s).sum() for s in grid])
    flip = None
    cross = [i for i in range(len(grid) - 1) if tot[i] == 0 or tot[i] * tot[i + 1] < 0]
    if cross:
        i = min(cross, key=lambda j: abs(grid[j] - spot))
        flip = float(grid[i] - tot[i] * (grid[i + 1] - grid[i]) / (tot[i + 1] - tot[i])) if tot[i + 1] != tot[i] else float(grid[i])
    total = float(g0.sum())
    near = [k for k in strikes if 0.75 * spot <= k <= 1.25 * spot]
    out = dict(currency=cur, spot=round(spot, 2), days=days, asof=now.strftime("%H:%M UTC"), call_wall=call_wall, put_wall=put_wall,
               flip=None if flip is None else round(flip, 1), total_gex_m=round(total / 1e6, 2),
               regime="positive" if total > 0 else "negative",
               strikes=[dict(k=k, call=round(call[k] / 1e6, 3), put=round(put[k] / 1e6, 3), net=round((call[k] - put[k]) / 1e6, 3)) for k in near])
    GEXC[(cur, days)] = (time.time(), out)
    return out

@app.get("/")
def index():
    for p in (os.path.join(HERE, "index.html"), os.path.join(HERE, "static", "index.html")):
        if os.path.exists(p):
            return FileResponse(p)
    raise HTTPException(404, "index.html missing")
