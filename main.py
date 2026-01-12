import os
import uuid
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from supabase import create_client, Client
import tushare as ts
import akshare as ak

app = FastAPI(title="Stock Intelligence Agent API")

# --- 初始化配置 ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") # 建议用 Service Role Key 避开 RLS
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_pro():
    token = os.getenv("TUSHARE_TOKEN")
    if not token: raise RuntimeError("TUSHARE_TOKEN missing")
    return ts.pro_api(token)

def to_ts_code(ticker: str) -> str:
    t = ticker.strip().upper()
    if "." in t: return t
    if len(t) == 6 and t.isdigit():
        return f"{t}.SH" if t.startswith("6") else f"{t}.SZ"
    return t

# --- 数据模型 ---
class RunInitReq(BaseModel):
    ticker: str
    method: str = "价值投资"
    target: str = "15%年化"
    horizon: str = "3年以上"

class TaskReq(BaseModel):
    run_id: str
    ticker: str

# --- 数据库辅助函数 ---
def save_fact(run_id: str, metric: str, value: Any, period: str, entity_type: str = "subject"):
    """存入结构化事实表"""
    data = {
        "run_id": run_id,
        "metric": metric,
        "value": float(value) if value is not None else None,
        "period": period,
        "entity_type": entity_type
    }
    supabase.table("normalized_facts").insert(data).execute()

# --- API 接口 ---

@app.post("/analysis/init")
async def init_run(req: RunInitReq):
    """Phase 0: 初始化任务并获取基本面概况 (Phase 1 基础)"""
    pro = get_pro()
    ts_code = to_ts_code(req.ticker)
    
    # 1. 创建 Run 记录
    basic = pro.stock_basic(ts_code=ts_code, fields="name,industry,market").iloc[0]
    run_data = {
        "ticker": ts_code,
        "name": basic['name'],
        "industry": basic['industry'],
        "method": req.method,
        "target": req.target,
        "horizon": req.horizon,
        "status": "running"
    }
    res = supabase.table("runs").insert(run_data).execute()
    run_id = res.data[0]['run_id']

    # 2. 获取主营业务构成 (Phase 1: 商业模式)
    try:
        biz = pro.fina_mainbz(ts_code=ts_code, type='P').head(5) # 按产品分类
        for _, row in biz.iterrows():
            save_fact(run_id, f"biz_segment_{row['bz_item']}", row['bz_sales'], row['end_date'])
    except: pass

    return {"run_id": run_id, "ticker": ts_code, "industry": basic['industry']}

@app.post("/analysis/financial-audit")
async def financial_audit(req: TaskReq):
    """Phase 3: 盈利能力与质量 (杜邦分析+现金流)"""
    pro = get_pro()
    
    # 获取 5 年财务指标
    df = pro.fina_indicator(ts_code=req.ticker, fields="end_date,roe,grossprofit_margin,netprofit_margin,debt_to_assets,asset_turnover")
    df_inc = pro.income(ts_code=req.ticker, fields="end_date,total_revenue,n_income")
    df_cf = pro.cashflow(ts_code=req.ticker, fields="end_date,n_cashflow_act")

    # 写入数据库
    for i in range(min(5, len(df))):
        p = df.iloc[i]['end_date'][:4] + "A"
        save_fact(req.run_id, "roe", df.iloc[i]['roe'], p)
        save_fact(req.run_id, "gross_margin", df.iloc[i]['grossprofit_margin'], p)
        save_fact(req.run_id, "net_profit", df_inc.iloc[i]['n_income'], p)
        save_fact(req.run_id, "ocf", df_cf.iloc[i]['n_cashflow_act'], p)

    return {"status": "success", "msg": "Phase 3 data loaded"}

@app.post("/analysis/governance")
async def governance_audit(req: TaskReq):
    """Phase 4: 股权与治理 (质押比例+前十大股东)"""
    pro = get_pro()
    
    # 1. 股东质押 (风险指标)
    pledge = pro.stk_pledge(ts_code=req.ticker)
    if not pledge.empty:
        latest_pledge = pledge.iloc[0]['pledge_ratio']
        save_fact(req.run_id, "pledge_ratio", latest_pledge, "LATEST")

    # 2. 前十大股东 (股权结构)
    holders = pro.top10_holders(ts_code=req.ticker).head(10)
    # 存入 sources 作为背景信息
    source_data = {
        "run_id": req.run_id,
        "type": "announcement",
        "title": f"{req.ticker} 前十大股东名单",
        "quote": str(holders[['holder_name', 'hold_ratio']].to_dict())
    }
    supabase.table("sources").insert(source_data).execute()
    
    return {"status": "success"}

@app.post("/analysis/market-valuation")
async def market_valuation(req: TaskReq):
    """Phase 6: 估值逻辑 (10年历史分位)"""
    pro = get_pro()
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now().replace(year=datetime.now().year-10)).strftime("%Y%m%d")
    
    df = pro.daily_basic(ts_code=req.ticker, start_date=start_date, end_date=end_date, fields="trade_date,pe,pb,ps")
    if df.empty: return {"status": "error"}

    latest = df.iloc[0]
    pe_pct = (df['pe'] < latest['pe']).mean()
    
    save_fact(req.run_id, "pe_current", latest['pe'], "LATEST")
    save_fact(req.run_id, "pe_percentile_10y", pe_pct * 100, "LATEST")

    return {"pe": latest['pe'], "pe_percentile": pe_pct}

@app.post("/analysis/news-rag")
async def news_to_sources(req: TaskReq):
    """Phase 5: 抓取新闻并存入 Sources 表供 Agent 提取多空分歧"""
    ticker_short = req.ticker.split('.')[0]
    df = ak.stock_news_em(symbol=ticker_short).head(15)
    
    for _, row in df.iterrows():
        source_id = f"news_{uuid.uuid4().hex[:8]}"
        data = {
            "source_id": source_id,
            "run_id": req.run_id,
            "type": "news",
            "title": row['新闻标题'],
            "url": row['新闻链接'],
            "published_at": row['发布时间'][:10],
            "quote": row['新闻内容'][:500]
        }
        supabase.table("sources").upsert(data).execute()
        
    return {"status": "success", "count": len(df)}