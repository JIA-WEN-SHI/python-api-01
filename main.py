# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List, Literal, Optional, Dict, Any, Tuple
from datetime import datetime, date, timedelta
import hashlib, json, os
import httpx

# =========================================================
# Config (Zeabur Variables)
# =========================================================
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "").strip()
LIXINGER_TOKEN = os.getenv("LIXINGER_TOKEN", "").strip()  # 预留
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "30"))

CNINFO_USER_AGENT = os.getenv(
    "CNINFO_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari/537.36",
).strip()
CNINFO_COOKIE = os.getenv("CNINFO_COOKIE", "").strip()  # 可选：遇到风控再配

RSS_FEEDS = os.getenv("RSS_FEEDS", "").strip()  # 可选：逗号分隔
GDELT_MAXRECORDS = int(os.getenv("GDELT_MAXRECORDS", "50"))

CollectorName = Literal["financials", "valuation", "filings", "news", "peers"]

# =========================================================
# Helpers
# =========================================================
def md5_text(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)

def normalize_ts_code(ticker: str) -> str:
    t = (ticker or "").strip().upper()
    if not t:
        return t
    if "." in t:
        return t
    if t.startswith("6"):
        return f"{t}.SH"
    if t.startswith(("0", "3")):
        return f"{t}.SZ"
    if t.startswith(("8", "4")):
        return f"{t}.BJ"
    return t

def make_source_id(run_id: str, provider: str, stype: str, url_hash: str) -> str:
    return f"src_{provider}_{stype}__{run_id}__{url_hash[:16]}"

async def http_get_json(url: str, params: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        r = await client.get(url, params=params, headers=headers)
        r.raise_for_status()
        return r.json()

async def http_post_form(url: str, data: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        r = await client.post(url, data=data, headers=headers)
        r.raise_for_status()
        return r.json()

async def tushare_post(api_name: str, params: Dict[str, Any], fields: str = "") -> Dict[str, Any]:
    if not TUSHARE_TOKEN:
        return {"code": -1, "msg": "TUSHARE_TOKEN missing", "data": None}
    payload: Dict[str, Any] = {"api_name": api_name, "token": TUSHARE_TOKEN, "params": params}
    if fields:
        payload["fields"] = fields
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        r = await client.post("https://api.tushare.pro", json=payload)
        r.raise_for_status()
        return r.json()

def tushare_first_row(resp: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], List[str], str]:
    """
    返回：(row_dict or None, fields, err_msg)
    """
    if not resp or resp.get("code") != 0:
        return None, [], resp.get("msg", "tushare error")
    data = resp.get("data") or {}
    fields = data.get("fields") or []
    items = data.get("items") or []
    if not fields or not items:
        return None, fields, "empty fields/items"
    return dict(zip(fields, items[0])), fields, ""

# =========================================================
# Models (align DB)
# =========================================================
class RunContext(BaseModel):
    run_id: str
    ticker: str
    name: Optional[str] = None
    asof: Optional[date] = None

    @field_validator("asof", mode="before")
    @classmethod
    def empty_str_to_none(cls, v):
        if v == "" or v is None:
            return None
        return v

class CollectRequest(BaseModel):
    run_context: RunContext
    collectors: List[CollectorName] = Field(default_factory=lambda: ["financials", "valuation", "filings", "news", "peers"])

class SourceRow(BaseModel):
    source_id: str
    run_id: str
    type: str
    title: Optional[str] = None
    url: str
    url_hash: str
    published_at: Optional[date] = None
    quote: Optional[str] = None
    retrieved_at: datetime
    content_hash: str
    raw: Optional[Dict[str, Any]] = None

class FactRow(BaseModel):
    run_id: str
    entity_kind: str = "company"
    entity_ticker: Optional[str] = None
    entity_name: Optional[str] = None
    metric: str
    value: float
    period: str
    unit: Optional[str] = None
    currency: Optional[str] = None
    basis: Optional[str] = None
    asof_date: Optional[date] = None
    source_id: str

class PeerRow(BaseModel):
    peer_ticker: str
    peer_name: Optional[str] = None

class EvidencePack(BaseModel):
    run_id: str
    ticker: str
    asof: Optional[date] = None
    sources: List[SourceRow] = Field(default_factory=list)
    facts: List[FactRow] = Field(default_factory=list)
    peers: List[PeerRow] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

# =========================================================
# App
# =========================================================
app = FastAPI(title="Collector Service", version="1.0.0")

@app.get("/health")
async def health():
    return {"ok": True, "ts": datetime.utcnow().isoformat()}

def build_source(
    run_id: str,
    source_type: str,
    provider: str,
    url: str,
    title: str,
    raw: Dict[str, Any],
    published_at: Optional[date] = None,
    quote: Optional[str] = None,
) -> SourceRow:
    url_hash = md5_text(url)
    source_id = make_source_id(run_id, provider, source_type, url_hash)
    retrieved_at = datetime.utcnow()
    content_hash = sha256_text(safe_json(raw))
    return SourceRow(
        source_id=source_id,
        run_id=run_id,
        type=source_type,
        title=title,
        url=url,
        url_hash=url_hash,
        published_at=published_at,
        quote=quote,
        retrieved_at=retrieved_at,
        content_hash=content_hash,
        raw=raw,
    )

# =========================================================
# Collectors
# =========================================================
async def collect_financials(ctx: RunContext) -> EvidencePack:
    """
    目标：三表 + 财务指标 + 当前估值快照（daily_basic）
    全部可审计：raw 存 sources.raw，facts 从 raw 映射（拿不到就不写）
    """
    pack = EvidencePack(run_id=ctx.run_id, ticker=ctx.ticker, asof=ctx.asof)
    ts_code = normalize_ts_code(ctx.ticker)

    if not TUSHARE_TOKEN:
        pack.warnings.append("TUSHARE_TOKEN 未配置：financials 未抓取。")
        return pack

    years = [2021, 2022, 2023]

    # ---- income ----
    income_raw = []
    for y in years:
        resp = await tushare_post("income", {"ts_code": ts_code, "period": f"{y}1231"})
        income_raw.append({"year": y, "resp": resp})
    src_income = build_source(
        run_id=ctx.run_id,
        source_type="market_data",
        provider="tushare",
        url=f"https://api.tushare.pro?api_name=income&ts_code={ts_code}",
        title=f"{ts_code} 利润表（Tushare income）",
        raw={"api": "income", "ts_code": ts_code, "years": years, "data": income_raw},
        quote="结构化接口返回（raw 可审计）。",
    )
    pack.sources.append(src_income)

    for item in income_raw:
        y = item["year"]
        row, _, err = tushare_first_row(item["resp"])
        if not row:
            pack.warnings.append(f"income FY{y} 无数据/无权限：{err}")
            continue
        mapping = {
            "revenue": row.get("revenue"),
            "net_profit": row.get("n_income"),
            "net_profit_parent": row.get("n_income_attr_p"),
            "cogs": row.get("oper_cost"),
        }
        for metric, val in mapping.items():
            if val is None:
                continue
            try:
                v = float(val)
            except Exception:
                continue
            pack.facts.append(FactRow(
                run_id=ctx.run_id, entity_kind="company", entity_ticker=ctx.ticker, entity_name=ctx.name,
                metric=metric, value=v, period=f"FY{y}",
                unit="cny", currency="CNY", basis="consolidated", asof_date=date(y,12,31),
                source_id=src_income.source_id
            ))

    # ---- balancesheet ----
    bs_raw = []
    for y in years:
        resp = await tushare_post("balancesheet", {"ts_code": ts_code, "period": f"{y}1231"})
        bs_raw.append({"year": y, "resp": resp})
    src_bs = build_source(
        run_id=ctx.run_id,
        source_type="market_data",
        provider="tushare",
        url=f"https://api.tushare.pro?api_name=balancesheet&ts_code={ts_code}",
        title=f"{ts_code} 资产负债表（Tushare balancesheet）",
        raw={"api": "balancesheet", "ts_code": ts_code, "years": years, "data": bs_raw},
        quote="结构化接口返回（raw 可审计）。",
    )
    pack.sources.append(src_bs)

    for item in bs_raw:
        y = item["year"]
        row, _, err = tushare_first_row(item["resp"])
        if not row:
            pack.warnings.append(f"balancesheet FY{y} 无数据/无权限：{err}")
            continue
        mapping = {
            "total_assets": row.get("total_assets"),
            "total_liab": row.get("total_liab"),
            "equity": row.get("total_hldr_eqy_exc_min_int"),
            "cash": row.get("money_cap"),
            "inventory": row.get("inventories"),
            "ar": row.get("accounts_receiv"),
            "debt_short": row.get("st_borr"),
            "debt_long": row.get("lt_borr"),
        }
        for metric, val in mapping.items():
            if val is None:
                continue
            try:
                v = float(val)
            except Exception:
                continue
            pack.facts.append(FactRow(
                run_id=ctx.run_id, entity_kind="company", entity_ticker=ctx.ticker, entity_name=ctx.name,
                metric=metric, value=v, period=f"FY{y}",
                unit="cny", currency="CNY", basis="consolidated", asof_date=date(y,12,31),
                source_id=src_bs.source_id
            ))

    # ---- cashflow ----
    cf_raw = []
    for y in years:
        resp = await tushare_post("cashflow", {"ts_code": ts_code, "period": f"{y}1231"})
        cf_raw.append({"year": y, "resp": resp})
    src_cf = build_source(
        run_id=ctx.run_id,
        source_type="market_data",
        provider="tushare",
        url=f"https://api.tushare.pro?api_name=cashflow&ts_code={ts_code}",
        title=f"{ts_code} 现金流量表（Tushare cashflow）",
        raw={"api": "cashflow", "ts_code": ts_code, "years": years, "data": cf_raw},
        quote="结构化接口返回（raw 可审计）。",
    )
    pack.sources.append(src_cf)

    for item in cf_raw:
        y = item["year"]
        row, _, err = tushare_first_row(item["resp"])
        if not row:
            pack.warnings.append(f"cashflow FY{y} 无数据/无权限：{err}")
            continue
        mapping = {
            "cfo": row.get("n_cashflow_act"),
            "cfi": row.get("n_cashflow_inv_act"),
            "cff": row.get("n_cashflow_fin_act"),
            "capex": row.get("c_pay_acq_const_fiolta"),  # 常用 capex 近似
        }
        for metric, val in mapping.items():
            if val is None:
                continue
            try:
                v = float(val)
            except Exception:
                continue
            pack.facts.append(FactRow(
                run_id=ctx.run_id, entity_kind="company", entity_ticker=ctx.ticker, entity_name=ctx.name,
                metric=metric, value=v, period=f"FY{y}",
                unit="cny", currency="CNY", basis="consolidated", asof_date=date(y,12,31),
                source_id=src_cf.source_id
            ))

    # ---- fina_indicator（比率类）----
    fi_raw = []
    for y in years:
        resp = await tushare_post("fina_indicator", {"ts_code": ts_code, "period": f"{y}1231"})
        fi_raw.append({"year": y, "resp": resp})
    src_fi = build_source(
        run_id=ctx.run_id,
        source_type="market_data",
        provider="tushare",
        url=f"https://api.tushare.pro?api_name=fina_indicator&ts_code={ts_code}",
        title=f"{ts_code} 财务指标（Tushare fina_indicator）",
        raw={"api": "fina_indicator", "ts_code": ts_code, "years": years, "data": fi_raw},
        quote="结构化接口返回（raw 可审计）。",
    )
    pack.sources.append(src_fi)

    for item in fi_raw:
        y = item["year"]
        row, _, err = tushare_first_row(item["resp"])
        if not row:
            pack.warnings.append(f"fina_indicator FY{y} 无数据/无权限：{err}")
            continue
        mapping = {
            "roe": row.get("roe"),
            "roa": row.get("roa"),
            "gross_margin": row.get("grossprofit_margin"),
            "net_margin": row.get("netprofit_margin"),
            "debt_to_asset": row.get("debt_to_assets"),
            "current_ratio": row.get("current_ratio"),
            "quick_ratio": row.get("quick_ratio"),
            "asset_turnover": row.get("assets_turn"),
        }
        for metric, val in mapping.items():
            if val is None:
                continue
            try:
                v = float(val)
            except Exception:
                continue
            pack.facts.append(FactRow(
                run_id=ctx.run_id, entity_kind="company", entity_ticker=ctx.ticker, entity_name=ctx.name,
                metric=metric, value=v, period=f"FY{y}",
                unit="pct" if metric in ["roe","roa","gross_margin","net_margin"] else "ratio",
                currency=None, basis="consolidated", asof_date=date(y,12,31),
                source_id=src_fi.source_id
            ))

    return pack

async def collect_valuation(ctx: RunContext) -> EvidencePack:
    """
    MVP：先做当前估值（PE/PB/PS）——用 Tushare daily_basic（可能也有权限门槛）
    历史分位后续再接：理杏仁 / 自算
    """
    pack = EvidencePack(run_id=ctx.run_id, ticker=ctx.ticker, asof=ctx.asof)
    if not TUSHARE_TOKEN:
        pack.warnings.append("TUSHARE_TOKEN 未配置：valuation 未抓取。")
        return pack

    ts_code = normalize_ts_code(ctx.ticker)
    today = (ctx.asof or date.today()).strftime("%Y%m%d")

    resp = await tushare_post("daily_basic", {"ts_code": ts_code, "trade_date": today})
    row, fields, err = tushare_first_row(resp)
    src = build_source(
        run_id=ctx.run_id,
        source_type="valuation_data",
        provider="tushare",
        url=f"https://api.tushare.pro?api_name=daily_basic&ts_code={ts_code}&trade_date={today}",
        title=f"{ts_code} 当前估值快照（Tushare daily_basic）",
        raw={"api":"daily_basic","ts_code":ts_code,"trade_date":today,"resp":resp,"fields_sample":fields[:50]},
        quote="结构化接口返回（raw 可审计）。"
    )
    pack.sources.append(src)

    if not row:
        pack.warnings.append(f"daily_basic 无数据/无权限：{err}")
        return pack

    mapping = {
        "pe_current": row.get("pe"),
        "pb_current": row.get("pb"),
        "ps_current": row.get("ps"),
        "total_mv": row.get("total_mv"),
    }
    for metric, val in mapping.items():
        if val is None:
            continue
        try:
            v = float(val)
        except Exception:
            continue
        pack.facts.append(FactRow(
            run_id=ctx.run_id, entity_kind="company", entity_ticker=ctx.ticker, entity_name=ctx.name,
            metric=metric, value=v, period="current",
            unit="times" if metric.endswith("_current") else "cny",
            currency="CNY" if metric == "total_mv" else None,
            basis="snapshot", asof_date=ctx.asof, source_id=src.source_id
        ))
    return pack

async def collect_news(ctx: RunContext) -> EvidencePack:
    """
    用 GDELT DOC API：要求输出必须有 url + published_at
    官方文档/示例见 GDELT 博客（DOC API）: api.gdeltproject.org/api/v2/doc/doc ... :contentReference[oaicite:4]{index=4}
    """
    pack = EvidencePack(run_id=ctx.run_id, ticker=ctx.ticker, asof=ctx.asof)

    # 查询关键词：公司名优先，其次 ticker
    q = (ctx.name or "").strip() or ctx.ticker
    if not q:
        pack.warnings.append("news：缺少 name/ticker 作为关键词")
        return pack

    # 最近 30 天
    params = {
        "query": f"\"{q}\"",
        "mode": "artlist",
        "format": "json",
        "sort": "datedesc",
        "maxrecords": str(GDELT_MAXRECORDS),
        "timespan": "30d",
    }

    try:
        data = await http_get_json("https://api.gdeltproject.org/api/v2/doc/doc", params=params)
    except Exception as e:
        pack.warnings.append(f"GDELT 请求失败：{e}")
        return pack

    articles = data.get("articles") or []
    # 用一个 source 汇总 raw（审计用）
    src_meta = build_source(
        run_id=ctx.run_id,
        source_type="news",
        provider="gdelt",
        url=f"https://api.gdeltproject.org/api/v2/doc/doc?query={q}",
        title=f"{q} 新闻候选池（GDELT DOC）",
        raw={"query": params, "count": len(articles), "resp_sample": articles[:3]},
        quote="新闻元数据来自 GDELT DOC API（URL 可追溯）。",
    )
    pack.sources.append(src_meta)

    # 每篇文章也单独落 sources（推荐，便于去重与引用）
    for a in articles:
        url = a.get("url")
        title = a.get("title") or "news"
        seendate = a.get("seendate")  # e.g. 20240101123000
        pub = None
        if seendate and len(seendate) >= 8:
            try:
                pub = date(int(seendate[0:4]), int(seendate[4:6]), int(seendate[6:8]))
            except Exception:
                pub = None
        if not url:
            continue
        s = build_source(
            run_id=ctx.run_id,
            source_type="news",
            provider="gdelt",
            url=url,
            title=title,
            raw={"gdelt": a},
            published_at=pub,
            quote=None,
        )
        pack.sources.append(s)

    return pack

async def collect_filings(ctx: RunContext) -> EvidencePack:
    """
    CNINFO 公告列表元数据（hisAnnouncement/query）
    常见入口： https://www.cninfo.com.cn/new/hisAnnouncement/query  :contentReference[oaicite:5]{index=5}
    注意：可能有风控/限流/验证码，失败就 warnings，不编造。
    """
    pack = EvidencePack(run_id=ctx.run_id, ticker=ctx.ticker, asof=ctx.asof)

    # CNINFO 常用的是 “secCode”/“stock” 等参数体系，实际可能需要你微调
    # MVP：先按代码模糊搜（跑通后再加 orgId 等精确字段）
    code = (ctx.ticker or "").strip()
    if not code:
        pack.warnings.append("filings：缺少 ticker")
        return pack

    url = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
    headers = {
        "User-Agent": CNINFO_USER_AGENT,
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.cninfo.com.cn/",
    }
    if CNINFO_COOKIE:
        headers["Cookie"] = CNINFO_COOKIE

    # 时间范围：最近 365 天
    end = (ctx.asof or date.today())
    start = end - timedelta(days=365)
    seDate = f"{start.strftime('%Y-%m-%d')}~{end.strftime('%Y-%m-%d')}"

    form = {
        "pageNum": "1",
        "pageSize": "30",
        "tabName": "fulltext",
        "seDate": seDate,
        "stock": code,
    }

    try:
        resp = await http_post_form(url, data=form, headers=headers)
    except Exception as e:
        pack.warnings.append(f"CNINFO 请求失败（可能风控/限流）：{e}")
        return pack

    ann = resp.get("announcements") or resp.get("data") or []
    src_meta = build_source(
        run_id=ctx.run_id,
        source_type="announcement",
        provider="cninfo",
        url=url,
        title=f"{code} 公告列表（CNINFO 元数据）",
        raw={"form": form, "resp_sample": ann[:3]},
        quote="公告元数据来自 CNINFO 列表接口（URL 可追溯）。",
    )
    pack.sources.append(src_meta)

    # 每条公告也落一条 sources（建议）
    for x in ann:
        title = x.get("announcementTitle") or x.get("title") or "announcement"
        adjunct = x.get("adjunctUrl") or x.get("url")
        pub_ms = x.get("announcementTime")  # ms timestamp 常见
        pub = None
        if isinstance(pub_ms, (int, float)):
            try:
                pub = datetime.utcfromtimestamp(pub_ms / 1000).date()
            except Exception:
                pub = None
        if adjunct:
            if adjunct.startswith("http"):
                pdf_url = adjunct
            else:
                pdf_url = f"https://static.cninfo.com.cn/{adjunct.lstrip('/')}"
        else:
            continue

        s = build_source(
            run_id=ctx.run_id,
            source_type="announcement",
            provider="cninfo",
            url=pdf_url,
            title=title,
            raw={"cninfo": x},
            published_at=pub,
            quote=None,
        )
        pack.sources.append(s)

    return pack

async def collect_peers(ctx: RunContext) -> EvidencePack:
    """
    同行池：先用 stock_basic 的 industry 字段做一个 MVP 同行列表
    注意：industry 口径各家不同，你后续可以换成 申万/中信分类
    """
    pack = EvidencePack(run_id=ctx.run_id, ticker=ctx.ticker, asof=ctx.asof)
    if not TUSHARE_TOKEN:
        pack.warnings.append("TUSHARE_TOKEN 未配置：peers 未抓取。")
        return pack

    # 先拿全量 stock_basic（MVP 简单做，后续可缓存）
    resp = await tushare_post("stock_basic", {"exchange": "", "list_status": "L"}, fields="ts_code,name,industry")
    if resp.get("code") != 0:
        pack.warnings.append(f"stock_basic 失败/无权限：{resp.get('msg')}")
        return pack

    data = resp.get("data") or {}
    fields = data.get("fields") or []
    items = data.get("items") or []
    if not fields or not items:
        pack.warnings.append("stock_basic 返回空")
        return pack

    # 找到当前公司行业
    my_ts = normalize_ts_code(ctx.ticker)
    my_ind = None
    for it in items:
        row = dict(zip(fields, it))
        if row.get("ts_code") == my_ts:
            my_ind = row.get("industry")
            break
    if not my_ind:
        pack.warnings.append("peers：无法从 stock_basic 找到该股或行业字段为空")
        return pack

    # 同行业取前 20 家
    peers = []
    for it in items:
        row = dict(zip(fields, it))
        if row.get("industry") == my_ind and row.get("ts_code") != my_ts:
            peers.append(row)
        if len(peers) >= 20:
            break

    # 用一个 source 记录“同行池构建规则”
    src = build_source(
        run_id=ctx.run_id,
        source_type="industry_report",
        provider="tushare",
        url="https://api.tushare.pro?api_name=stock_basic",
        title=f"{ctx.ticker} 同行池（同 industry={my_ind}）",
        raw={"industry": my_ind, "peers_sample": peers[:5]},
        quote="同行池由 Tushare stock_basic industry 字段构建（MVP 规则）。"
    )
    pack.sources.append(src)

    for p in peers:
        peers_ticker = (p.get("ts_code") or "").split(".")[0]
        pack.peers.append(PeerRow(peer_ticker=peers_ticker, peer_name=p.get("name")))

    return pack

COLLECTOR_MAP = {
    "financials": collect_financials,
    "valuation": collect_valuation,
    "filings": collect_filings,
    "news": collect_news,
    "peers": collect_peers,
}

def merge_packs(packs: List[EvidencePack]) -> EvidencePack:
    out = packs[0]

    seen_sid = set()
    merged_sources = []
    for p in packs:
        for s in p.sources:
            if s.source_id in seen_sid:
                continue
            seen_sid.add(s.source_id)
            merged_sources.append(s)
    out.sources = merged_sources

    out.facts = [f for p in packs for f in p.facts]

    seen_peer = set()
    merged_peers = []
    for p in packs:
        for pe in p.peers:
            if pe.peer_ticker in seen_peer:
                continue
            seen_peer.add(pe.peer_ticker)
            merged_peers.append(pe)
    out.peers = merged_peers

    out.warnings = [w for p in packs for w in p.warnings]
    return out

@app.post("/collect/evidence-pack", response_model=EvidencePack)
async def collect_evidence(req: CollectRequest):
    ctx = req.run_context
    if not ctx.run_id or not ctx.ticker:
        raise HTTPException(status_code=400, detail="run_context.run_id 和 run_context.ticker 必填")

    packs: List[EvidencePack] = []
    for c in req.collectors:
        fn = COLLECTOR_MAP.get(c)
        if not fn:
            raise HTTPException(status_code=400, detail=f"Unknown collector: {c}")
        packs.append(await fn(ctx))

    out = merge_packs(packs)

    # 真实性护栏：facts.source_id 必须能在 sources 中找到
    source_ids = {s.source_id for s in out.sources}
    bad_facts = [f for f in out.facts if f.source_id not in source_ids]
    if bad_facts:
        out.warnings.append(f"{len(bad_facts)} 条 facts 缺失 source_id 对应 sources：建议 n8n 拒绝写入 normalized_facts。")

    # sources hash 完整性
    bad_src = [s for s in out.sources if not s.url_hash or not s.content_hash]
    if bad_src:
        out.warnings.append(f"{len(bad_src)} 条 sources 缺少 hash：建议 n8n 拒绝写入 sources。")

    return out
