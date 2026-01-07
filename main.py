from fastapi import FastAPI, HTTPException
import akshare as ak
import pandas as pd

app = FastAPI(title="Stock_Agent_Backend")

# --- 子 Agent 1: 商业模式数据 ---
@app.get("/agent/business")
def get_business(ticker: str):
    # 逻辑：获取主营构成 + 公司概况
    try:
        df_zygc = ak.stock_zygc_em(symbol=ticker[2:])
        df_info = ak.stock_individual_info_em(symbol=ticker[2:])
        return {
            "business": df_zygc.head(10).to_dict('records'),
            "profile": df_info.to_dict('records')
        }
    except:
        return {"error": "Data fetch failed"}

# --- 子 Agent 2: 行业周期数据 ---
@app.get("/agent/industry")
def get_industry(ticker: str):
    # 逻辑：获取行业板块信息 + 行业市盈率
    df_valuation = ak.stock_a_ttm_exotic(symbol=ticker)
    return {"industry_valuation": df_valuation.to_dict('records')}

# --- 子 Agent 3: 财务审计数据 ---
@app.get("/agent/financial")
def get_financial(ticker: str):
    # 逻辑：获取核心财务指标
    df_indicator = ak.stock_financial_analysis_indicator(symbol=ticker[2:])
    return {"metrics": df_indicator.head(5).to_dict('records')}

# --- 子 Agent 4: 治理监察数据 ---
@app.get("/agent/governance")
def get_governance(ticker: str):
    # 逻辑：获取股权质押 + 前十大股东
    df_pledge = ak.stock_gpzy_pledge_ratio_em(symbol=ticker[2:])
    return {"pledge_ratio": df_pledge.head(1).to_dict('records')}

# --- 子 Agent 5: 估值精算数据 (DCF) ---
@app.post("/agent/calculate_dcf")
def do_dcf(fcf: float, g: float, r: float):
    # 逻辑：纯数学计算
    intrinsic_value = fcf * (1 + g) / (r - g) # 简化版永续增长模型
    return {"intrinsic_value": round(intrinsic_value, 2)}