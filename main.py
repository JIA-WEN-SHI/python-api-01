import akshare as ak
import pandas as pd
from fastapi import FastAPI
import os

app = FastAPI()

@app.get("/fetch_all")
def fetch_all(symbol: str):
    data = {"status": "success", "symbol": symbol}
    
    try:
        # 1. 获取核心财务指标 (ROE, 净利率等)
        df_indicator = ak.stock_financial_analysis_indicator(symbol=symbol)
        data["finance"] = df_indicator.head(5).to_dict(orient='records')
        
        # 2. 获取主营构成 (Phase 1 核心)
        df_zygc = ak.stock_zygc_em(symbol=symbol)
        data["business"] = df_zygc.to_dict(orient='records')
        
        # 3. 获取估值指标 (PE, PB, 股息率)
        df_valuation = ak.stock_a_indicator_lg(symbol=symbol)
        # 只要最新的一行，包含当前 PE 和历史分位参考
        data["valuation"] = df_valuation.tail(1).to_dict(orient='records')

    except Exception as e:
        data["status"] = "error"
        data["message"] = str(e)
        
    return data

# Zeabur 运行代码保持不变...