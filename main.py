from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import os, time, hashlib, traceback

import pandas as pd
import tushare as ts
import akshare as ak

app = FastAPI(title="Stock Data Proxy", version="1.4.0")

# =========================================================
# ENV
# =========================================================
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "120"))

_CACHE: Dict[str, Dict[str, Any]] = {}
_RATE_BUCKET: Dict[str, List[float]] = {}

# =========================================================
# Utils
# =========================================================
def _cache_key(path: str, payload: Dict[str, Any]) -> str:
    raw = path + "::" + str(sorted(payload.items()))
    return hashlib.md5(raw.encode()).hexdigest()

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
        raise HTTPException(429, f"Rate limit exceeded: {key}")
    bucket.append(now)
    _RATE_BUCKET[key] = bucket

# =========================================================
# Ticker helpers
# =========================================================
def to_ts_code(ticker: str) -> str:
    t = ticker.strip().upper()
    if t.endswith(".SH") or t.endswith(".SZ"):
        return t
    if len(t) == 6 and t.isdigit():
        return f"{t}.SH" if t.startswith("6") else f"{t}.SZ"
    return t

def to_6digit(ticker: str) -> str:
    t = ticker.strip().upper()
    if "." in t:
        t = t.split(".")[0]
    return t.replace("SH", "").replace("SZ", "")

# =========================================================
# Request Models
# =========================================================
class FundamentalsReq(BaseModel):
    ticker: str
    years: int = 5

class ValuationReq(BaseModel):
    ticker: str
    years: int = 10

class NewsReq(BaseModel):
    ticker: str
    limit: int = 10

# =========================================================
# TuShare
# =========================================================
def tushare_client():
    if not TUSHARE_TOKEN:
        raise Exception("Missing TUSHARE_TOKEN")
    return ts.pro_api(TUSHARE_TOKEN)

def fundamentals_tushare(ticker: str, years: int):
    items = []
    try:
        pro = tushare_client()
        ts_code = to_ts_code(ticker)
        now_year = pd.Timestamp.utcnow().year

        for y in range(now_year - years, now_year):
            period = f"{y}1231"
            try:
                df = pro.income(
                    ts_code=ts_code,
                    period=period,
                    fields="end_date,total_revenue,n_income_attr_p"
                )
                if df is None or df.empty:
                    continue
                r = df.iloc[0]
                items.append({
                    "period": f"{y}A",
                    "revenue": float(r.get("total_revenue") or 0),
                    "net_profit": float(r.get("n_income_attr_p") or 0),
                })
            except Exception:
                continue
    except Exception as e:
        return {"items": [], "error": str(e)}

    return {
        "items": items,
        "source": {"source_id": "src_tushare_fin"}
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
            return {"current": {}, "percentile": {}}

        df = df.sort_values("trade_date")
        cur = df.iloc[-1]

        def pct(series, v):
            s = series.dropna().astype(float)
            return float((s < v).mean()) if not s.empty and v is not None else None

        return {
            "current": {
                "pe": float(cur["pe"]) if pd.notna(cur["pe"]) else None,
                "pb": float(cur["pb"]) if pd.notna(cur["pb"]) else None,
                "ps": float(cur["ps"]) if pd.notna(cur["ps"]) else None,
            },
            "percentile": {
                "pe_10y": pct(df["pe"], cur["pe"]),
                "pb_10y": pct(df["pb"], cur["pb"]),
                "ps_10y": pct(df["ps"], cur["ps"]),
            },
            "source": {"source_id": "src_tushare_mkt"}
        }
    except Exception as e:
        traceback.print_exc()
        return {"current": {}, "percentile": {}, "error": str(e)}

# =========================================================
# News – 东方财富（AKShare）
# =========================================================
def news_em(ticker: str, limit: int):
    items = []
    try:
        df = ak.stock_news_em(symbol=to_6digit(ticker)).head(limit)
        for _, r in df.iterrows():
            items.append({
                "title": str(r.get("新闻标题", "")),
                "url": str(r.get("新闻链接", "")),
                "published_at": str(r.get("发布时间", "")),
            })
    except Exception as e:
        return {"items": [], "error": str(e)}

    return {"items": items, "source": {"source_id": "src_em_news"}}

# =========================================================
# API
# =========================================================
@app.post("/finance/fundamentals")
def fundamentals(req: FundamentalsReq):
    ck = _cache_key("/finance/fundamentals", req.dict())
    if (cached := cache_get(ck)):
        return cached

    rate_limit("tushare:fundamentals")
    out = fundamentals_tushare(req.ticker, req.years)
    cache_set(ck, out)
    return out

@app.post("/market/valuation")
def valuation(req: ValuationReq):
    ck = _cache_key("/market/valuation", req.dict())
    if (cached := cache_get(ck)):
        return cached

    rate_limit("tushare:valuation")
    out = valuation_tushare(req.ticker, req.years)
    cache_set(ck, out)
    return out

@app.post("/news/search")
def news(req: NewsReq):
    ck = _cache_key("/news/search", req.dict())
    if (cached := cache_get(ck)):
        return cached

    rate_limit("em:news")
    out = news_em(req.ticker, req.limit)
    cache_set(ck, out)
    return out

@app.get("/health")
def health():
    return {"ok": True, "cache": len(_CACHE)}
