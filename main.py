# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import os
import time
import hashlib
import pandas as pd

import tushare as ts

app = FastAPI(title="Stock Data Proxy", version="1.1.0")

# -----------------------------
# Env / Settings
# -----------------------------
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")
LIXINGER_TOKEN = os.getenv("LIXINGER_TOKEN", "")  # 你的理杏仁 OpenAPI token
LIXINGER_BASE_URL = os.getenv("LIXINGER_BASE_URL", "https://www.lixinger.com/open/api")
LIXINGER_TIMEOUT = int(os.getenv("LIXINGER_TIMEOUT", "30"))

JQ_USER = os.getenv("JQ_USER", "")
JQ_PASSWORD = os.getenv("JQ_PASSWORD", "")

# 简单缓存（同一参数短时间重复请求直接返回）
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))  # 1小时；你也可以设成 86400
# 简单限流（每分钟最多请求次数）
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "120"))  # 120/min 默认偏宽

# 内存缓存与限流计数器（单实例有效；多实例需换 Redis）
_CACHE: Dict[str, Dict[str, Any]] = {}
_RATE_BUCKET: Dict[str, List[float]] = {}  # key -> timestamps


# -----------------------------
# Helpers: cache / ratelimit
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
    """非常简单的滑动窗口限流：每 key 每分钟最多 RATE_LIMIT_PER_MIN 次"""
    now = time.time()
    bucket = _RATE_BUCKET.get(key, [])
    # 只保留最近 60 秒
    bucket = [t for t in bucket if now - t < 60]
    if len(bucket) >= RATE_LIMIT_PER_MIN:
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded for {key}")
    bucket.append(now)
    _RATE_BUCKET[key] = bucket


# -----------------------------
# Ticker converters
# -----------------------------
def to_ts_code(ticker: str) -> str:
    t = ticker.strip().upper()
    if t.endswith(".SH") or t.endswith(".SZ"):
        return t
    if len(t) == 6 and t.isdigit():
        return f"{t}.SH" if t.startswith("6") else f"{t}.SZ"
    return ticker

def to_6digit_a_share(ticker: str) -> str:
    """用于 AKShare/部分数据源：取 6 位"""
    t = ticker.strip().upper()
    if "." in t:
        t = t.split(".")[0]
    t = t.replace("SH", "").replace("SZ", "")
    return t

def to_jq_code(ticker: str) -> str:
    """JoinQuant 代码规则：000001.XSHE / 600000.XSHG"""
    t = ticker.strip().upper()
    if t.endswith(".SZ"):
        return t.replace(".SZ", ".XSHE")
    if t.endswith(".SH"):
        return t.replace(".SH", ".XSHG")
    if len(t) == 6 and t.isdigit():
        return f"{t}.XSHG" if t.startswith("6") else f"{t}.XSHE"
    return ticker


# -----------------------------
# Request models
# -----------------------------
class FundamentalsReq(BaseModel):
    ticker: str
    years: int = 5
    provider: str = "tushare"  # tushare | lixinger | joinquant

class ValuationReq(BaseModel):
    ticker: str
    years: int = 10
    provider: str = "tushare"  # tushare | lixinger | joinquant

class NewsReq(BaseModel):
    ticker: Optional[str] = None
    name: Optional[str] = None
    limit: int = 10
    provider: str = "em"  # em | joinquant | ...（你后续可扩）


# -----------------------------
# TuShare
# -----------------------------
def tushare_client():
    if not TUSHARE_TOKEN:
        raise HTTPException(status_code=500, detail="Missing env TUSHARE_TOKEN")
    return ts.pro_api(TUSHARE_TOKEN)

def fundamentals_tushare(ticker: str, years: int):
    pro = tushare_client()
    ts_code = to_ts_code(ticker)

    now_year = pd.Timestamp.utcnow().year
    periods = [f"{y}1231" for y in range(now_year - years, now_year)]
    items = []

    for p in periods:
        inc = pro.income(ts_code=ts_code, period=p, fields="end_date,total_revenue,n_income_attr_p")
        cf = pro.cashflow(ts_code=ts_code, period=p, fields="end_date,n_cashflow_act")
        fi = pro.fina_indicator(ts_code=ts_code, period=p, fields="end_date,roe,grossprofit_margin,netprofit_margin")

        if inc is None or inc.empty:
            continue

        end_date = str(inc.iloc[0]["end_date"])
        revenue = float(inc.iloc[0].get("total_revenue") or 0)
        net_profit = float(inc.iloc[0].get("n_income_attr_p") or 0)

        cfo = float(cf.iloc[0].get("n_cashflow_act") or 0) if cf is not None and not cf.empty else None
        roe = float(fi.iloc[0].get("roe") or 0) if fi is not None and not fi.empty else None
        gross_margin = float(fi.iloc[0].get("grossprofit_margin") or 0) if fi is not None and not fi.empty else None
        net_margin = float(fi.iloc[0].get("netprofit_margin") or 0) if fi is not None and not fi.empty else None

        items.append({
            "period": end_date[:4] + "A",
            "revenue": revenue,
            "net_profit": net_profit,
            "cfo": cfo,
            "roe": roe,
            "gross_margin": gross_margin,
            "net_margin": net_margin,
        })

    return {
        "items": items,
        "source": {"source_id": "src_tushare_fin", "title": "TuShare Pro 财务接口", "url": "https://tushare.pro/"}
    }

def valuation_tushare(ticker: str, years: int):
    pro = tushare_client()
    ts_code = to_ts_code(ticker)

    end = pd.Timestamp.utcnow().strftime("%Y%m%d")
    start = (pd.Timestamp.utcnow() - pd.DateOffset(years=years)).strftime("%Y%m%d")

    df = pro.daily_basic(ts_code=ts_code, start_date=start, end_date=end, fields="trade_date,pe,pb,ps")
    if df is None or df.empty:
        return {
            "current": {},
            "percentile": {},
            "source": {"source_id": "src_tushare_mkt", "title": "TuShare Pro daily_basic", "url": "https://tushare.pro/"}
        }

    df = df.sort_values("trade_date")
    cur = df.iloc[-1]

    def pct(series: pd.Series, v: float) -> Optional[float]:
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
        "source": {"source_id": "src_tushare_mkt", "title": "TuShare Pro daily_basic", "url": "https://tushare.pro/"}
    }


# -----------------------------
# Lixinger (skeleton)
# -----------------------------
def lixinger_headers() -> Dict[str, str]:
    if not LIXINGER_TOKEN:
        raise HTTPException(status_code=500, detail="Missing env LIXINGER_TOKEN")
    # 注意：理杏仁真实鉴权方式以你后台文档为准（可能不是 Bearer）
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LIXINGER_TOKEN}",
    }

def fundamentals_lixinger(ticker: str, years: int):
    """
    你要做的两件事：
    1) 确认 endpoint（文档中 api-key=cn/company/fundamental/non_financial 指向的真实路径）
    2) 确认请求 payload 字段名 + 返回 JSON 结构
    """
    import requests

    # 你给的 doc 参数：cn/company/fundamental/non_financial
    # 通常 endpoint 会长这样（但你必须按后台文档确认）
    endpoint = "/cn/company/fundamental/non_financial"
    url = LIXINGER_BASE_URL.rstrip("/") + endpoint

    # ====== 关键：payload 字段请按理杏仁文档替换 ======
    payload = {
        # 下面字段名大概率不对，你用后台“请求示例”替换
        "stockCode": ticker,          # 可能需要 market + code 或 lixinger 的内部 code
        "years": years,               # 可能需要 startYear/endYear 或 dateRange
    }
    # =============================================

    rate_limit("lixinger:fundamentals")
    r = requests.post(url, headers=lixinger_headers(), json=payload, timeout=LIXINGER_TIMEOUT)
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Lixinger error {r.status_code}: {r.text[:300]}")
    data = r.json()

    # ====== 关键：解析返回 JSON 映射到统一 items ======
    # 你需要把 data 中的字段映射成：
    #   period / revenue / net_profit / cfo / roe / gross_margin / net_margin
    items: List[Dict[str, Any]] = []

    # 示例（伪代码）：假设返回里有 data["data"] 是按年列表
    # for row in data.get("data", []):
    #     year = row["year"]
    #     items.append({
    #       "period": f"{year}A",
    #       "revenue": row.get("revenue"),
    #       "net_profit": row.get("netProfit"),
    #       "cfo": row.get("cfo"),
    #       "roe": row.get("roe"),
    #       "gross_margin": row.get("grossMargin"),
    #       "net_margin": row.get("netMargin"),
    #     })

    # 如果你暂时只想“先跑通”，也可以先返回空 items，避免 n8n 报错
    # =============================================

    return {
        "items": items,
        "source": {"source_id": "src_lixinger_fin", "title": "理杏仁 OpenAPI 财务", "url": url}
    }


# -----------------------------
# JoinQuant (valuation skeleton)
# -----------------------------
_JQ_READY = False

def joinquant_init():
    global _JQ_READY
    if _JQ_READY:
        return
    if not JQ_USER or not JQ_PASSWORD:
        raise HTTPException(status_code=500, detail="Missing env JQ_USER/JQ_PASSWORD")
    import jqdatasdk
    jqdatasdk.auth(JQ_USER, JQ_PASSWORD)
    _JQ_READY = True

def valuation_joinquant(ticker: str, years: int):
    """
    这里给你一个可落地的框架：
    - 你需要在 JoinQuant 里取到历史 pe/pb/ps 序列（字段名取决于你权限与接口）
    - 拿到 df 后，分位算法完全复用你原来的 pct()
    """
    joinquant_init()
    import jqdatasdk
    jq_code = to_jq_code(ticker)

    end = pd.Timestamp.utcnow().date()
    start = (pd.Timestamp.utcnow() - pd.DateOffset(years=years)).date()

    # ====== 关键：这里的数据接口/字段按你的 JoinQuant 权限调整 ======
    # 常见思路1：用 get_valuation 拉估值（若你权限支持）
    # df = jqdatasdk.get_valuation(jq_code, start_date=str(start), end_date=str(end),
    #                             fields=["day", "pe_ratio", "pb_ratio", "ps_ratio"])
    #
    # 常见思路2：用 finance 表或指标表（更复杂）
    #
    # 我这里先做一个“保底”：如果拿不到就返回空，n8n 不炸
    df = None
    try:
        df = jqdatasdk.get_valuation(
            jq_code,
            start_date=str(start),
            end_date=str(end),
            fields=["day", "pe_ratio", "pb_ratio", "ps_ratio"]
        )
    except Exception:
        df = None
    # =============================================

    if df is None or len(df) == 0:
        return {
            "current": {},
            "percentile": {},
            "source": {"source_id": "src_joinquant_mkt", "title": "JoinQuant 估值数据", "url": "joinquant://get_valuation"}
        }

    df = df.sort_values("day")
    cur = df.iloc[-1]

    def pct(series: pd.Series, v: float) -> Optional[float]:
        s = series.dropna().astype(float)
        if s.empty or v is None:
            return None
        return float((s < v).mean())

    pe = float(cur["pe_ratio"]) if pd.notna(cur["pe_ratio"]) else None
    pb = float(cur["pb_ratio"]) if pd.notna(cur["pb_ratio"]) else None
    ps = float(cur["ps_ratio"]) if pd.notna(cur["ps_ratio"]) else None

    return {
        "current": {"pe": pe, "pb": pb, "ps": ps},
        "percentile": {
            "pe_10y": pct(df["pe_ratio"], pe),
            "pb_10y": pct(df["pb_ratio"], pb),
            "ps_10y": pct(df["ps_ratio"], ps),
        },
        "source": {"source_id": "src_joinquant_mkt", "title": "JoinQuant 估值数据", "url": "joinquant://get_valuation"}
    }


# -----------------------------
# News (AKShare 东方财富，保留)
# -----------------------------
def news_em(ticker: str, limit: int):
    import akshare as ak
    code6 = to_6digit_a_share(ticker)

    items = []
    if code6.isdigit() and len(code6) == 6:
        df = ak.stock_news_em(symbol=code6)
        df = df.head(limit)
        for _, r in df.iterrows():
            items.append({
                "title": str(r.get("新闻标题") or r.get("title") or ""),
                "url": str(r.get("新闻链接") or r.get("url") or ""),
                "published_at": str(r.get("发布时间") or r.get("date") or ""),
                "snippet": str(r.get("新闻内容") or r.get("content") or "")[:180]
            })

    return {
        "items": items[:limit],
        "source": {"source_id": "src_em_news", "title": "东方财富个股新闻(AKShare stock_news_em)", "url": "https://akshare.akfamily.xyz/"}
    }


# -----------------------------
# API endpoints (with provider switch + cache)
# -----------------------------
@app.post("/finance/fundamentals")
def fundamentals(req: FundamentalsReq):
    payload = req.dict()
    ck = _cache_key("/finance/fundamentals", payload)
    cached = cache_get(ck)
    if cached is not None:
        return cached

    provider = (req.provider or "tushare").lower().strip()

    if provider == "tushare":
        rate_limit("tushare:fundamentals")
        out = fundamentals_tushare(req.ticker, req.years)
    elif provider == "lixinger":
        out = fundamentals_lixinger(req.ticker, req.years)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported fundamentals provider: {provider}")

    cache_set(ck, out)
    return out


@app.post("/market/valuation")
def valuation(req: ValuationReq):
    payload = req.dict()
    ck = _cache_key("/market/valuation", payload)
    cached = cache_get(ck)
    if cached is not None:
        return cached

    provider = (req.provider or "tushare").lower().strip()

    if provider == "tushare":
        rate_limit("tushare:valuation")
        out = valuation_tushare(req.ticker, req.years)
    elif provider == "joinquant":
        out = valuation_joinquant(req.ticker, req.years)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported valuation provider: {provider}")

    cache_set(ck, out)
    return out


@app.post("/news/search")
def news(req: NewsReq):
    payload = req.dict()
    ck = _cache_key("/news/search", payload)
    cached = cache_get(ck)
    if cached is not None:
        return cached

    provider = (req.provider or "em").lower().strip()

    ticker = req.ticker or ""
    limit = int(req.limit or 10)

    if provider == "em":
        rate_limit("em:news")
        out = news_em(ticker, limit)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported news provider: {provider}")

    cache_set(ck, out)
    return out


@app.get("/health")
def health():
    return {"ok": True, "cache_size": len(_CACHE)}
