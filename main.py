# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import os, time, hashlib

import pandas as pd
import tushare as ts

app = FastAPI(title="Stock Data Proxy", version="1.5.0")

# =========================================================
# Env / Settings
# =========================================================
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")
LIXINGER_TOKEN = os.getenv("LIXINGER_TOKEN", "")
LIXINGER_BASE_URL = os.getenv("LIXINGER_BASE_URL", "https://www.lixinger.com/open/api")
LIXINGER_TIMEOUT = int(os.getenv("LIXINGER_TIMEOUT", "30"))

JQ_USER = os.getenv("JQ_USER", "")
JQ_PASSWORD = os.getenv("JQ_PASSWORD", "")

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "120"))

_CACHE: Dict[str, Dict[str, Any]] = {}
_RATE_BUCKET: Dict[str, List[float]] = {}

# =========================================================
# Cache / Rate limit
# =========================================================
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
    bucket = [t for t in _RATE_BUCKET.get(key, []) if now - t < 60]
    if len(bucket) >= RATE_LIMIT_PER_MIN:
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded: {key}")
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

def to_6digit_a_share(ticker: str) -> str:
    t = ticker.strip().upper().split(".")[0]
    return t.replace("SH", "").replace("SZ", "")

def to_jq_code(ticker: str) -> str:
    t = ticker.strip().upper()
    if t.endswith(".SZ"):
        return t.replace(".SZ", ".XSHE")
    if t.endswith(".SH"):
        return t.replace(".SH", ".XSHG")
    if len(t) == 6 and t.isdigit():
        return f"{t}.XSHG" if t.startswith("6") else f"{t}.XSHE"
    return t

# =========================================================
# Request Models
# =========================================================
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

# =========================================================
# TuShare
# =========================================================
def tushare_client():
    if not TUSHARE_TOKEN:
        return None
    return ts.pro_api(TUSHARE_TOKEN)

def fundamentals_tushare(ticker: str, years: int):
    pro = tushare_client()
    if not pro:
        return {"items": [], "error": "TUSHARE_TOKEN missing", "source": {"source_id": "src_tushare_fin"}}

    ts_code = to_ts_code(ticker)
    now_year = pd.Timestamp.utcnow().year
    items = []

    for y in range(now_year - years, now_year):
        try:
            df = pro.income(
                ts_code=ts_code,
                period=f"{y}1231",
                fields="end_date,total_revenue,n_income_attr_p"
            )
            if df is None or df.empty:
                continue
            r = df.iloc[0]
            items.append({
                "period": f"{y}A",
                "revenue": float(r.get("total_revenue") or 0),
                "net_profit": float(r.get("n_income_attr_p") or 0),
                "cfo": None,
                "roe": None,
                "gross_margin": None,
                "net_margin": None,
            })
        except Exception:
            continue

    return {"items": items, "source": {"source_id": "src_tushare_fin"}}

def valuation_tushare(ticker: str, years: int):
    pro = tushare_client()
    if not pro:
        return {"current": {}, "percentile": {}, "error": "TUSHARE_TOKEN missing",
                "source": {"source_id": "src_tushare_mkt"}}

    ts_code = to_ts_code(ticker)
    end = pd.Timestamp.utcnow().strftime("%Y%m%d")
    start = (pd.Timestamp.utcnow() - pd.DateOffset(years=years)).strftime("%Y%m%d")

    try:
        df = pro.daily_basic(
            ts_code=ts_code,
            start_date=start,
            end_date=end,
            fields="trade_date,pe,pb,ps"
        )
    except Exception as e:
        return {"current": {}, "percentile": {}, "error": str(e),
                "source": {"source_id": "src_tushare_mkt"}}

    if df is None or df.empty:
        return {"current": {}, "percentile": {}, "source": {"source_id": "src_tushare_mkt"}}

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

# =========================================================
# Lixinger / JoinQuant 保留（Skeleton，不影响构建）
# =========================================================
def fundamentals_lixinger(ticker: str, years: int):
    return {"items": [], "error": "Lixinger not enabled", "source": {"source_id": "src_lixinger_fin"}}

def valuation_joinquant(ticker: str, years: int):
    return {"current": {}, "percentile": {}, "error": "JoinQuant not enabled",
            "source": {"source_id": "src_joinquant_mkt"}}

# =========================================================
# News – AKShare（runtime import，避免 build 失败）
# =========================================================
def news_em(ticker: str, limit: int):
    try:
        import akshare as ak
    except Exception as e:
        return {"items": [], "error": f"akshare import failed: {e}",
                "source": {"source_id": "src_em_news"}}

    code = to_6digit_a_share(ticker or "")
    items = []

    try:
        df = ak.stock_news_em(symbol=code).head(limit)
        for _, r in df.iterrows():
            items.append({
                "title": str(r.get("新闻标题", "")),
                "url": str(r.get("新闻链接", "")),
                "published_at": str(r.get("发布时间", "")),
                "snippet": str(r.get("新闻内容", ""))[:180],
            })
    except Exception as e:
        return {"items": [], "error": str(e), "source": {"source_id": "src_em_news"}}

    return {"items": items, "source": {"source_id": "src_em_news"}}

# =========================================================
# API
# =========================================================
@app.post("/finance/fundamentals")
def fundamentals(req: FundamentalsReq):
    ck = _cache_key("/finance/fundamentals", req.dict())
    if (v := cache_get(ck)) is not None:
        return v

    rate_limit("tushare:fundamentals")
    out = fundamentals_tushare(req.ticker, req.years)
    cache_set(ck, out)
    return out

@app.post("/market/valuation")
def valuation(req: ValuationReq):
    ck = _cache_key("/market/valuation", req.dict())
    if (v := cache_get(ck)) is not None:
        return v

    rate_limit("tushare:valuation")
    out = valuation_tushare(req.ticker, req.years)
    cache_set(ck, out)
    return out

@app.post("/news/search")
def news(req: NewsReq):
    ck = _cache_key("/news/search", req.dict())
    if (v := cache_get(ck)) is not None:
        return v

    rate_limit("em:news")
    out = news_em(req.ticker, req.limit)
    cache_set(ck, out)
    return out

@app.get("/health")
def health():
    return {"ok": True, "cache_size": len(_CACHE)}
