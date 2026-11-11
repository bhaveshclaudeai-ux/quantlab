import copy, itertools, math, os, re, time
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

# ---------------- indicators / signals ----------------
def ind(df, k, p):
    if k in ("open", "high", "low", "close", "volume"):
        return df[k].to_numpy()
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
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
        day = df["t"] // 86400
        tp = (h + l + c) / 3
        s = (tp * v).groupby(day).cumsum() / v.groupby(day).cumsum()
    else:
        raise ValueError(f"unknown indicator {k}")
    return s.to_numpy()

def val(df, o, cache):
    if o["k"] == "num":
        return np.full(len(df), float(o["v"]))
    key = (o["k"], o.get("p"))
    if key not in cache:
        cache[key] = ind(df, o["k"], o.get("p"))
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
    equity, trades = np.full(n, cap0), []

    def close(i, px, why):
        nonlocal eq, pos
        gross = pos * qty * (px - ent)
        eq += gross - fee * qty * px
        pnl = gross - fee * qty * (ent + px)
        trades.append(dict(side="long" if pos > 0 else "short", t0=int(t[ei]), t1=int(t[i]),
                           entry=float(ent), exit=float(px), pnl=float(pnl), pct=float(pnl / (qty * ent) * 100), ret=float(pnl / eq0), why=why))
        pos = 0

    for i in range(1, n):
        if eq <= 0:
            equity[i:] = 0
            break
        el, es, xl, xs = sig["entry_long"], sig["entry_short"], sig["exit_long"], sig["exit_short"]
        wl = el is not None and mode in ("long", "both") and bool(el[i - 1])
        ws = es is not None and mode in ("short", "both") and bool(es[i - 1])
        if pos > 0 and ((xl is not None and xl[i - 1]) or (ws and not wl)):
            close(i, o[i], "signal")
        elif pos < 0 and ((xs is not None and xs[i - 1]) or (wl and not ws)):
            close(i, o[i], "signal")
        if pos == 0 and (wl or ws):
            pos, ent, ei = (1 if wl else -1), o[i], i
            eq0 = eq
            qty = eq * size / ent
            eq -= fee * qty * ent
        if pos:
            slp, tpp = ent * (1 - pos * sl), ent * (1 + pos * tp)
            if pos > 0:
                if sl and l[i] <= slp: close(i, min(slp, o[i]), "sl")
                elif tp and h[i] >= tpp: close(i, max(tpp, o[i]), "tp")
            else:
                if sl and h[i] >= slp: close(i, max(slp, o[i]), "sl")
                elif tp and l[i] <= tpp: close(i, min(tpp, o[i]), "tp")
        equity[i] = eq + (pos * qty * (c[i] - ent) if pos else 0)
    if pos:
        close(n - 1, c[n - 1], "end")
        equity[-1] = eq
    return equity, trades, cap0

def metrics(df, equity, trades, tf, cap0):
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
                final=round(float(equity[-1]), 2), rr=round(rr, 2), aw=round(aw, 2), al=round(al, 2), nw=len(w))

# ---------------- api ----------------
class Req(BaseModel):
    symbol: str
    interval: str = "1h"
    candles: int = 3000
    strategy: dict

class OptReq(Req):
    params: list[dict]
    metric: str = "return_dd"
    min_trades: int = 10

def prep(q):
    sym = q.symbol.upper().strip()
    if q.interval not in TF or not re.fullmatch(r"[A-Z0-9]{5,20}", sym):
        raise HTTPException(400, "Bad symbol or interval")
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
        trades=trades, metrics=metrics(df, equity, trades, q.interval, cap0))

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
    if total > 1500:
        raise HTTPException(400, f"{total} combinations — limit is 1500. Reduce range or increase step.")
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

@app.get("/")
def index():
    for p in (os.path.join(HERE, "index.html"), os.path.join(HERE, "static", "index.html")):
        if os.path.exists(p):
            return FileResponse(p)
    raise HTTPException(404, "index.html missing")
