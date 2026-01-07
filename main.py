from fastapi import FastAPI
import akshare as ak
import pandas as pd

app = FastAPI()

@app.get("/analyze")
def get_stock_data(symbol: str):
    try:
        # 1. 获取财务指标 (过去5年)
        df_finance = ak.stock_financial_analysis_indicator_finance_baidu(symbol=symbol, range="年度")
        finance_data = df_finance.head(5).to_dict(orient='records')
        
        # 2. 获取主营构成 (Phase 1 商业模式核心)
        df_zygc = ak.stock_zygc_em(symbol=symbol)
        zygc_data = df_zygc.to_dict(orient='records')
        
        # 3. 获取个股指标 (估值、PE/PB)
        # 注意：这里简化处理，实际可根据需要增加更多接口
        return {
            "status": "success",
            "symbol": symbol,
            "finance": finance_data,
            "business_structure": zygc_data
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    import os
    # Zeabur 会自动分配 PORT 环境变量
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)