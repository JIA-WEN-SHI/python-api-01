from fastapi import FastAPI
import akshare as ak
import pandas as pd
import os
import uvicorn

app = FastAPI()

@app.get("/analyze")
def get_stock_data(symbol: str):
    response = {"status": "success", "symbol": symbol, "finance": [], "business_structure": []}
    
    # 1. 尝试抓取财务指标 (最核心数据)
    try:
        # 使用财务分析指标接口
        df_finance = ak.stock_financial_analysis_indicator(symbol=symbol)
        if not df_finance.empty:
            # 只取最近 4 条，减少内存占用和 JSON 体积
            response["finance"] = df_finance.head(4).to_dict(orient='records')
    except Exception as e:
        print(f"Finance Error: {e}")
        response["finance_error"] = str(e)

    # 2. 尝试抓取主营构成 (Phase 1 核心)
    # 如果接口 zygc_em 报错，尝试另一种替代方案
    try:
        df_zygc = ak.stock_zygc_em(symbol=symbol)
        if not df_zygc.empty:
            response["business_structure"] = df_zygc.to_dict(orient='records')
    except Exception:
        # 如果主营构成接口崩了，不抛出异常，只返回空数组，让 n8n 继续运行
        response["business_structure"] = []
        response["business_error"] = "主营构成接口暂时不可用，建议从新闻中提取"

    return response

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)