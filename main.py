# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import pandas as pd
import tushare as ts

app = FastAPI()

def to_ts_code(ticker: str) -> str:
    t = ticker.strip().upper()
    # 已带后缀直接返回
    if t.endswith(".SH") or t.endswith(".SZ"):
        return t
    # 常见 6 位 A 股规则
    if len(t) == 6 and t.isdigit():
        return f"{t}.SH" if t.startswith("6") else f"{t}.SZ"
    return ticker  # 兜底：别瞎改

class FundamentalsReq(BaseModel):
    ticker: str
    years: int = 5

class ValuationReq(BaseModel):
    ticker: str
    years: int = 10

class NewsReq(BaseModel):
    ticker: Optional[str] = None
    name: Optional[str] = None
    limit: int = 10

def tushare_client():
    token = os.getenv("TUSHARE_TOKEN", "")
    if not token:
        raise RuntimeError("Missing env TUSHARE_TOKEN")
    return ts.pro_api(token)

@app.post("/finance/fundamentals")
def fundamentals(req: FundamentalsReq):
    pro = tushare_client()
    ts_code = to_ts_code(req.ticker)

    # 取最近 N 个年报：用 period=YYYY1231
    now_year = pd.Timestamp.utcnow().year
    periods = [f"{y}1231" for y in range(now_year - req.years, now_year)]
    items = []

    for p in periods:
        # income: total_revenue, n_income_attr_p
        inc = pro.income(ts_code=ts_code, period=p,
                         fields="end_date,total_revenue,n_income_attr_p")
        # cashflow: n_cashflow_act
        cf = pro.cashflow(ts_code=ts_code, period=p,
                          fields="end_date,n_cashflow_act")
        # fina_indicator: roe, grossprofit_margin, netprofit_margin
        fi = pro.fina_indicator(ts_code=ts_code, period=p,
                                fields="end_date,roe,grossprofit_margin,netprofit_margin")

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

@app.post("/market/valuation")
def valuation(req: ValuationReq):
    pro = tushare_client()
    ts_code = to_ts_code(req.ticker)

    end = pd.Timestamp.utcnow().strftime("%Y%m%d")
    start = (pd.Timestamp.utcnow() - pd.DateOffset(years=req.years)).strftime("%Y%m%d")

    df = pro.daily_basic(ts_code=ts_code, start_date=start, end_date=end,
                         fields="trade_date,pe,pb,ps")
    if df is None or df.empty:
        return {"current": {}, "percentile": {}, "source": {"source_id": "src_tushare_mkt", "title": "TuShare Pro daily_basic", "url": "https://tushare.pro/"}}

    df = df.sort_values("trade_date")
    cur = df.iloc[-1]

    def pct(series: pd.Series, v: float) -> float:
        s = series.dropna().astype(float)
        if s.empty:
            return None
        return float((s < v).mean())

    pe = float(cur["pe"]) if pd.notna(cur["pe"]) else None
    pb = float(cur["pb"]) if pd.notna(cur["pb"]) else None
    ps = float(cur["ps"]) if pd.notna(cur["ps"]) else None

    return {
        "current": {"pe": pe, "pb": pb, "ps": ps},
        "percentile": {
            "pe_10y": pct(df["pe"], pe) if pe is not None else None,
            "pb_10y": pct(df["pb"], pb) if pb is not None else None,
            "ps_10y": pct(df["ps"], ps) if ps is not None else None,
        },
        "source": {"source_id": "src_tushare_mkt", "title": "TuShare Pro daily_basic", "url": "https://tushare.pro/"}
    }

@app.post("/news/search")
def news(req: NewsReq):
    # MVP：优先 AKShare 东方财富个股新闻
    import akshare as ak

    ticker = (req.ticker or "").strip()
    # AKShare 这里一般用 6 位代码
    if "." in ticker:
        ticker = ticker.split(".")[0]
    ticker = ticker.replace("SH", "").replace("SZ", "")

    items = []
    if ticker.isdigit() and len(ticker) == 6:
        df = ak.stock_news_em(symbol=ticker)
        # 常见列：标题/内容/发布时间/链接（不同版本列名可能略有差异）
        df = df.head(req.limit)
        for _, r in df.iterrows():
            items.append({
                "title": str(r.get("新闻标题") or r.get("title") or ""),
                "url": str(r.get("新闻链接") or r.get("url") or ""),
                "published_at": str(r.get("发布时间") or r.get("date") or ""),
                "snippet": str(r.get("新闻内容") or r.get("content") or "")[:180]
            })

    return {
        "items": items[:req.limit],
        "source": {"source_id": "src_em_news", "title": "东方财富个股新闻(AKShare stock_news_em)", "url": "https://akshare.akfamily.xyz/"}
    }
