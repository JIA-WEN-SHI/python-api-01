# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import os
import time
import hashlib
import pandas as pd
import traceback

import tushare as ts

app = FastAPI(title="Stock Data Proxy", version="1.2.0")

# -----------------------------
# Env / Settings
# -----------------------------
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")
JQ_USER = os.getenv("JQ_USER", "")
JQ_PASSWORD = os.getenv("JQ_PASSWORD", "")

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "120"))

_CACHE: Dict[str, Dict[str, Any]] = {}
_RATE_BUCKET: Dict[str, List[float]] = {}

# -----------------------------
# Helpers
# -----------------------------
def _cache_key(path: str, payload: Dict[str, Any]) -> str:
    raw = path + "::" + str(sorted(payload.items()))
    return hashlib.md5(raw.encode("utf-8")).hexdigest()

def cache_get(key: str):
    v = _CACHE.get(key)
    if not v:
        return None
    if time.time() - v["ts"] > CACHE_TTL_SECONDS:
        _CACHE.pop(key, None)
        return None
    return v["data"]

def cache_set(key: str, data: Any):
    _CACHE[key] = {"ts": time.time(), "data": data}

def rate_limit(key: str):
    now = time.time()
    bucket = _RATE_BUCKET.get(key, [])
    bucket = [t for t in bucket if now - t < 60]
    if len(bucket) >= RATE_LIMIT_PER_MIN:
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded for {key}")
    bucket.append(now)
    _RATE_BUCKET[key] = bucket

# -----------------------------
# Ticker helpers
# -----------------------------
def to_ts_code(ticker: str) -> str:
    t = ticker.strip().upper()
    if t.endswith(".SH") or t.endswith(".SZ"):
        return t
    if len(t) == 6 and t.isdigit():
        return f"{t}.SH" if t.startswith("6") else f"{t}.SZ"
    return ticker

def to_6digit_a_share(ticker: str) -> str:
    t = ticker.strip().upper()
    if "." in t:
        t = t.split(".")[0]
    return t.replace("SH", "").replace("SZ", "")

# -----------------------------
# Request Models
# -----------------------------
class FundamentalsReq(BaseModel):
    ticker: str
    years: int = 5
    provider: str = "tushare"

class ValuationReq(BaseModel):
    ticker: str
    years: int = 10
    provider: str = "tushare"

class NewsReq(BaseModel):
    ticker: Optional[str] = None
    limit: int = 10
    provider: str = "em"

# -----------------------------
# TuShare
# -----------------------------
def tushare_client():
    if not TUSHARE_TOKEN:
        raise Exception("Missing TUSHARE_TOKEN")
    return ts.pro_api(TUSHARE_TOKEN)

def fundamentals_tushare(ticker: str, years: int):
    try:
        pro = tushare_client()
        ts_code = to_ts_code(ticker)
    except Exception as e:
        return {
            "items": [],
            "error": f"tushare init failed: {e}",
            "source": {"source_id": "src_tushare_fin"}
        }

    now_year = pd.Timestamp.utcnow().year
    periods = [f"{y}1231" for y in range(now_year - years, now_year)]
    items = []

    for p in periods:
        try:
            inc = pro.income(
                ts_code=ts_code,
                period=p,
                fields="end_date,total_revenue,n_income_attr_p"
            )
            if inc is None or inc.empty:
                continue

            row = inc.iloc[0]
            items.append({
                "period": str(row.get("end_date", ""))[:4] + "A",
                "revenue": float(row.get("total_revenue") or 0),
                "net_profit": float(row.get("n_income_attr_p") or 0),
                "cfo": None,
                "roe": None,
                "gross_margin": None,
                "net_margin": None,
            })
        except Exception as e:
            print(f"[fundamentals skip {p}] {e}")
            continue

    return {
        "items": items,
        "source": {
            "source_id": "src_tushare_fin",
            "title": "TuShare Pro"
        }
    }

def valuation_tushare(ticker: str, years: int):
    try:
        pro = tushare_client()
        ts_code = to_ts_code(ticker)

        end = pd.Timestamp.utcnow().strftime("%Y%m%d")
        start = (pd.Timestamp.utcnow() - pd.DateOffset(years=years)).strftime("%Y%m%d")

        df = pro.daily_basic(
            ts_code=ts_code,
            start_date=start,
            end_date=end,
            fields="trade_date,pe,pb,ps"
        )
        if df is None or df.empty:
            return {"current": {}, "percentile": {}, "source": {"source_id": "src_tushare_mkt"}}

        df = df.sort_values("trade_date")
        cur = df.iloc[-1]

        def pct(series, v):
            s = series.dropna().astype(float)
            if s.empty or v is None:
                return None
            return float((s < v).mean())

        pe = float(cur["pe"]) if pd.notna(cur["pe"]) else None
        pb = float(cur["pb"]) if pd.notna(cur["pb"]) else None
        ps = float(cur["ps"]) if pd.notna(cur["ps"]) else None

        return {
            "current": {"pe": pe, "pb": pb, "ps": ps},
            "percentile": {
                "pe_10y": pct(df["pe"], pe),
                "pb_10y": pct(df["pb"], pb),
                "ps_10y": pct(df["ps"], ps),
            },
            "source": {"source_id": "src_tushare_mkt"}
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "current": {},
            "percentile": {},
            "error": str(e),
            "source": {"source_id": "src_tushare_mkt"}
        }

# -----------------------------
# News (AKShare)
# -----------------------------
def news_em(ticker: str, limit: int):
    import akshare as ak
    code6 = to_6digit_a_share(ticker)
    items = []

    try:
        df = ak.stock_news_em(symbol=code6)
        df = df.head(limit)
        for _, r in df.iterrows():
            items.append({
                "title": str(r.get("新闻标题", "")),
                "url": str(r.get("新闻链接", "")),
                "published_at": str(r.get("发布时间", "")),
                "snippet": str(r.get("新闻内容", ""))[:180],
            })
    except Exception as e:
        print(f"[news_em] {e}")

    return {
        "items": items,
        "source": {"source_id": "src_em_news"}
    }

# -----------------------------
# API Endpoints
# -----------------------------
@app.post("/finance/fundamentals")
def fundamentals(req: FundamentalsReq):
    payload = req.dict()
    ck = _cache_key("/finance/fundamentals", payload)
    cached = cache_get(ck)
    if cached:
        return cached

    if req.provider.lower() != "tushare":
        raise HTTPException(400, "Only tushare supported (free tier)")

    rate_limit("tushare:fundamentals")
    out = fundamentals_tushare(req.ticker, req.years)
    cache_set(ck, out)
    return out

@app.post("/market/valuation")
def valuation(req: ValuationReq):
    payload = req.dict()
    ck = _cache_key("/market/valuation", payload)
    cached = cache_get(ck)
    if cached:
        return cached

    if req.provider.lower() != "tushare":
        raise HTTPException(400, "Only tushare supported (free tier)")

    rate_limit("tushare:valuation")
    out = valuation_tushare(req.ticker, req.years)
    cache_set(ck, out)
    return out

@app.post("/news/search")
def news(req: NewsReq):
    payload = req.dict()
    ck = _cache_key("/news/search", payload)
    cached = cache_get(ck)
    if cached:
        return cached

    if req.provider.lower() != "em":
        raise HTTPException(400, "Only em supported")

    rate_limit("em:news")
    out = news_em(req.ticker or "", req.limit)
    cache_set(ck, out)
    return out

@app.get("/health")
def health():
    return {"ok": True, "cache_size": len(_CACHE)}
