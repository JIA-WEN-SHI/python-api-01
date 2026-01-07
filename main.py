from fastapi import FastAPI
import akshare as ak
import pandas as pd
import os
import uvicorn

app = FastAPI()

@app.get("/analyze")
def get_stock_data(symbol: str):
    try:
        # 1. 自动处理代码格式：如果是 600519 -> sh600519
        # AkShare 的财务指标接口通常需要带前缀或特定格式
        
        # 获取主要财务指标 (由东方财富提供数据源，更稳定)
        # 替代旧的 baidu 接口
        df_finance = ak.stock_financial_analysis_indicator(symbol=symbol)
        
        # 转换日期并取最近 5 年
        df_finance = df_finance.head(5)
        finance_data = df_finance.to_dict(orient='records')
        
        # 2. 获取主营构成 (Phase 1 商业模式核心)
        df_zygc = ak.stock_zygc_em(symbol=symbol)
        zygc_data = df_zygc.to_dict(orient='records')
        
        return {
            "status": "success",
            "symbol": symbol,
            "finance": finance_data,
            "business_structure": zygc_data
        }
    except Exception as e:
        return {
            "status": "error", 
            "message": f"AkShare Error: {str(e)}. 请检查代码 {symbol} 是否正确。"
        }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)