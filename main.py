# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List, Literal, Optional, Dict, Any, Tuple
from datetime import datetime, date, timedelta
import hashlib, json, os, re
import httpx
from xml.etree import ElementTree as ET

# =========================================================
# Config (Zeabur Variables)
# =========================================================
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "").strip()

# 理杏仁（Lixinger OpenAPI）
# 你现在用的是：https://open.lixinger.com/api/...
LIXINGER_TOKEN = os.getenv("LIXINGER_TOKEN", "").strip()
LIXINGER_BASE_URL = os.getenv("LIXINGER_BASE_URL", "https://open.lixinger.com/api").strip()
LIXINGER_TIMEOUT = float(os.getenv("LIXINGER_TIMEOUT", "30"))

# 理杏仁：fundamental/non_financial（可覆盖）
LIXINGER_ENDPOINT_FUNDAMENTAL_NON_FINANCIAL = os.getenv(
    "LIXINGER_ENDPOINT_FUNDAMENTAL_NON_FINANCIAL",
    "/cn/company/fundamental/non_financial",
).strip()

# 通用 HTTP
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "30"))
HTTP_UA = os.getenv(
    "HTTP_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari/537.36",
).strip()

# CNINFO（巨潮）
CNINFO_USER_AGENT = os.getenv("CNINFO_USER_AGENT", HTTP_UA).strip()
CNINFO_COOKIE = os.getenv("CNINFO_COOKIE", "").strip()  # 可选：遇到风控再配
CNINFO_PAGE_SIZE = int(os.getenv("CNINFO_PAGE_SIZE", "30"))
CNINFO_MAX_PAGES = int(os.getenv("CNINFO_MAX_PAGES", "2"))  # 防止翻太多

# RSS
RSS_FEEDS = os.getenv("RSS_FEEDS", "").strip()
RSS_MAX_ITEMS = int(os.getenv("RSS_MAX_ITEMS", "30"))

# GDELT
GDELT_MAXRECORDS = int(os.getenv("GDELT_MAXRECORDS", "50"))
GDELT_TIMESPAN_DAYS = int(os.getenv("GDELT_TIMESPAN_DAYS", "30"))
GDELT_LANG = os.getenv("GDELT_LANG", "").strip()  # 可选

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
    """
    支持：
    - 600519 / 600519.SH
    - 000001 / 300xxx
    - 北交所 8xxxx / 4xxxx -> .BJ
    """
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

def strip_html(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def parse_date_yyyymmdd(s: str) -> Optional[date]:
    if not s:
        return None
    s = s.strip()
    try:
        if len(s) == 8 and s.isdigit():
            return date(int(s[0:4]), int(s[4:6]), int(s[6:8]))
    except Exception:
        return None
    return None

async def http_get_text(url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> str:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers=headers) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        return r.text

async def http_get_json(url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers=headers) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        return r.json()

async def http_post_form(url: str, data: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers=headers) as client:
        r = await client.post(url, data=data)
        r.raise_for_status()
        return r.json()

async def http_post_json(url: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None, timeout: Optional[float] = None) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=(timeout or HTTP_TIMEOUT), headers=headers) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        return r.json()

# =========================================================
# Tushare
# =========================================================
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
    if not resp:
        return None, [], "empty resp"
    if resp.get("code") != 0:
        return None, [], resp.get("msg", "tushare error")
    data = resp.get("data") or {}
    fields = data.get("fields") or []
    items = data.get("items") or []
    if not fields or not items:
        return None, fields, "empty fields/items"
    return dict(zip(fields, items[0])), fields, ""

# =========================================================
# Lixinger (理杏仁) OpenAPI
# =========================================================
async def lixinger_post(endpoint_path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    理杏仁文档：token 在 body 里（不是 Authorization header）
    例如 POST https://open.lixinger.com/api/cn/company/fundamental/non_financial
    """
    if not endpoint_path:
        return {"code": -1, "message": "LIXINGER endpoint missing", "data": None}
    url = LIXINGER_BASE_URL.rstrip("/") + "/" + endpoint_path.lstrip("/")
    headers = {
        "User-Agent": HTTP_UA,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    try:
        return await http_post_json(url, payload=payload, headers=headers, timeout=LIXINGER_TIMEOUT)
    except Exception as e:
        return {"code": -1, "message": f"lixinger request failed: {e}", "data": None}

# =========================================================
# RSS Parser (RSS/Atom 简易解析)
# =========================================================
def parse_rss_or_atom(xml_text: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not xml_text:
        return out
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return out

    channel = root.find("channel")
    if channel is not None:
        for item in channel.findall("item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub = (item.findtext("pubDate") or "").strip()
            desc = (item.findtext("description") or "").strip()
            out.append({"title": strip_html(title), "link": link, "pub_raw": pub, "summary": strip_html(desc)})
        return out

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    if root.tag.endswith("feed"):
        for entry in root.findall("atom:entry", ns) + root.findall("entry"):
            title = (entry.findtext("atom:title", default="", namespaces=ns) or entry.findtext("title") or "").strip()
            link = ""
            link_el = entry.find("atom:link", ns) or entry.find("link")
            if link_el is not None:
                link = (link_el.attrib.get("href") or "").strip()
            updated = (entry.findtext("atom:updated", default="", namespaces=ns) or entry.findtext("updated") or "").strip()
            summary = (entry.findtext("atom:summary", default="", namespaces=ns) or entry.findtext("summary") or "").strip()
            out.append({"title": strip_html(title), "link": link, "pub_raw": updated, "summary": strip_html(summary)})
        return out

    return out

def try_parse_pub_date(pub_raw: str) -> Optional[date]:
    if not pub_raw:
        return None
    pub_raw = pub_raw.strip()
    m = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", pub_raw)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except Exception:
            return None
    m2 = re.search(r"(\d{8})", pub_raw)
    if m2:
        return parse_date_yyyymmdd(m2.group(1))
    return None

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

class CollectOptions(BaseModel):
    years: Optional[List[int]] = None
    gdelt_days: Optional[int] = None
    gdelt_maxrecords: Optional[int] = None
    rss_feeds: Optional[List[str]] = None
    rss_max_items: Optional[int] = None
    cninfo_days: Optional[int] = None
    cninfo_page_size: Optional[int] = None
    cninfo_max_pages: Optional[int] = None

class CollectRequest(BaseModel):
    run_context: RunContext
    collectors: List[CollectorName] = Field(default_factory=lambda: ["financials", "valuation", "filings", "news", "peers"])
    options: Optional[CollectOptions] = None

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
async def collect_financials(ctx: RunContext, opt: Optional[CollectOptions]) -> EvidencePack:
    pack = EvidencePack(run_id=ctx.run_id, ticker=ctx.ticker, asof=ctx.asof)
    if not TUSHARE_TOKEN:
        pack.warnings.append("TUSHARE_TOKEN 未配置：financials 未抓取。")
        return pack

    ts_code = normalize_ts_code(ctx.ticker)
    years = (opt.years if opt and opt.years else [2021, 2022, 2023])

    # income
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

    # balancesheet
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

    # cashflow
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
            "capex": row.get("c_pay_acq_const_fiolta"),
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

    # fina_indicator
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

    pct_metrics = {"roe", "roa", "gross_margin", "net_margin"}
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
                unit="pct" if metric in pct_metrics else "ratio",
                currency=None, basis="consolidated", asof_date=date(y,12,31),
                source_id=src_fi.source_id
            ))

    return pack

async def collect_valuation(ctx: RunContext, opt: Optional[CollectOptions]) -> EvidencePack:
    """
    估值优先级：
    1) 理杏仁 fundamental/non_financial（推荐，PB/PS/分位都能靠 metricsList 拿）
    2) Tushare daily_basic（fallback）
    3) 都不行 -> warnings
    """
    pack = EvidencePack(run_id=ctx.run_id, ticker=ctx.ticker, asof=ctx.asof)
    asof = ctx.asof or date.today()

    # ---------- 1) Lixinger ----------
    # 理杏仁接口用 stockCode（6位数字，不带 .SH/.SZ）
    stock_code = (ctx.ticker or "").strip()
    if "." in stock_code:
        stock_code = stock_code.split(".")[0]
    stock_code = re.sub(r"\D", "", stock_code)  # 只保留数字
    if LIXINGER_TOKEN and stock_code:
        # 你要的 current + 10y 分位，全部靠 metricsList
        metrics_list = [
            "mc", "pe_ttm", "pb", "ps_ttm", "dyr", "sp",
            "pe_ttm.y10.cvpos", "pb.y10.cvpos", "ps_ttm.y10.cvpos",
        ]
        payload = {
            "token": LIXINGER_TOKEN,
            "date": asof.isoformat(),
            "stockCodes": [stock_code],
            "metricsList": metrics_list,
        }

        resp = await lixinger_post(LIXINGER_ENDPOINT_FUNDAMENTAL_NON_FINANCIAL, payload)

        src = build_source(
            run_id=ctx.run_id,
            source_type="valuation_data",
            provider="lixinger",
            url=LIXINGER_BASE_URL.rstrip("/") + "/" + LIXINGER_ENDPOINT_FUNDAMENTAL_NON_FINANCIAL.lstrip("/"),
            title=f"{ctx.ticker} 估值&分位（理杏仁 fundamental/non_financial）",
            raw={"payload": payload, "resp_sample": resp},
            quote="估值与分位来自理杏仁 OpenAPI（raw 可审计）。",
            published_at=asof,
        )
        pack.sources.append(src)

        if not isinstance(resp, dict) or resp.get("code") != 1:
            pack.warnings.append(f"valuation：理杏仁返回异常：{resp.get('message') if isinstance(resp, dict) else 'invalid resp'}")
            return pack

        data = resp.get("data") or []
        if not data:
            pack.warnings.append("valuation：理杏仁 data 为空（可能日期无数据/权限/参数问题）")
            return pack

        row = data[0] if isinstance(data, list) else data
        # 只在取到数值时写 facts（不编造）
        def to_float(v) -> Optional[float]:
            try:
                if v is None:
                    return None
                return float(v)
            except Exception:
                return None

        # current
        mc = to_float(row.get("mc"))
        pe = to_float(row.get("pe_ttm"))
        pb = to_float(row.get("pb"))
        ps = to_float(row.get("ps_ttm"))
        dyr = to_float(row.get("dyr"))
        sp = to_float(row.get("sp"))

        # percentile (cvpos 通常是 0~1；你入库用 pct 建议乘 100，或保持 0~1 二选一)
        # 这里默认：保持 0~1（更忠实原始），unit 也写 pct01；你想要 0~100 再改一行
        pe_p = to_float(row.get("pe_ttm.y10.cvpos"))
        pb_p = to_float(row.get("pb.y10.cvpos"))
        ps_p = to_float(row.get("ps_ttm.y10.cvpos"))

        # 写入 facts
        mapping_current = {
            "total_mv": mc,          # 市值
            "pe_current": pe,
            "pb_current": pb,
            "ps_current": ps,
            "dividend_yield": dyr,
            "price": sp,
        }
        for metric, v in mapping_current.items():
            if v is None:
                continue
            pack.facts.append(FactRow(
                run_id=ctx.run_id,
                entity_kind="company",
                entity_ticker=ctx.ticker,
                entity_name=ctx.name,
                metric=metric,
                value=v,
                period="current",
                unit="cny" if metric == "total_mv" else ("times" if metric.endswith("_current") else ("pct" if metric == "dividend_yield" else "cny")),
                currency="CNY" if metric in ("total_mv", "price") else None,
                basis="snapshot",
                asof_date=asof,
                source_id=src.source_id
            ))

        mapping_pct = {
            "pe_percentile_10y": pe_p,
            "pb_percentile_10y": pb_p,
            "ps_percentile_10y": ps_p,
        }
        for metric, v in mapping_pct.items():
            if v is None:
                continue
            pack.facts.append(FactRow(
                run_id=ctx.run_id,
                entity_kind="company",
                entity_ticker=ctx.ticker,
                entity_name=ctx.name,
                metric=metric,
                value=v,
                period="10y",
                unit="pct01",  # 0~1 的分位
                currency=None,
                basis="history_percentile",
                asof_date=asof,
                source_id=src.source_id
            ))

        # 若拿到了响应但没任何数值落库，提示你检查 metricsList 或日期
        if not pack.facts:
            pack.warnings.append("valuation：理杏仁已请求但未解析出任何可用指标（检查 metricsList/date/权限）")

        return pack

    # ---------- 2) fallback to Tushare daily_basic ----------
    if not TUSHARE_TOKEN:
        pack.warnings.append("valuation：未配置 LIXINGER_TOKEN，且无 TUSHARE_TOKEN，未抓取。")
        return pack

    ts_code = normalize_ts_code(ctx.ticker)
    trade_date = asof.strftime("%Y%m%d")

    resp = await tushare_post("daily_basic", {"ts_code": ts_code, "trade_date": trade_date})
    row, fields, err = tushare_first_row(resp)

    src = build_source(
        run_id=ctx.run_id,
        source_type="valuation_data",
        provider="tushare",
        url=f"https://api.tushare.pro?api_name=daily_basic&ts_code={ts_code}&trade_date={trade_date}",
        title=f"{ts_code} 当前估值快照（Tushare daily_basic）",
        raw={"api": "daily_basic", "ts_code": ts_code, "trade_date": trade_date, "resp": resp, "fields_sample": fields[:60]},
        quote="结构化接口返回（raw 可审计）。",
        published_at=asof,
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
            basis="snapshot",
            asof_date=asof,
            source_id=src.source_id
        ))

    return pack

async def collect_news(ctx: RunContext, opt: Optional[CollectOptions]) -> EvidencePack:
    pack = EvidencePack(run_id=ctx.run_id, ticker=ctx.ticker, asof=ctx.asof)
    keyword = (ctx.name or "").strip() or ctx.ticker

    # -------- GDELT --------
    gdelt_days = (opt.gdelt_days if opt and opt.gdelt_days is not None else GDELT_TIMESPAN_DAYS)
    gdelt_max = (opt.gdelt_maxrecords if opt and opt.gdelt_maxrecords is not None else GDELT_MAXRECORDS)

    if keyword:
        params = {
            "query": f"\"{keyword}\"",
            "mode": "artlist",
            "format": "json",
            "sort": "datedesc",
            "maxrecords": str(gdelt_max),
            "timespan": f"{int(gdelt_days)}d",
        }

        try:
            data = await http_get_json("https://api.gdeltproject.org/api/v2/doc/doc", params=params, headers={"User-Agent": HTTP_UA})
            articles = data.get("articles") or []
        except Exception as e:
            pack.warnings.append(f"GDELT 请求失败：{e}")
            articles = []

        src_meta = build_source(
            run_id=ctx.run_id,
            source_type="news",
            provider="gdelt",
            url="https://api.gdeltproject.org/api/v2/doc/doc",
            title=f"{keyword} 新闻候选池（GDELT）",
            raw={"params": params, "count": len(articles), "resp_sample": (articles[:3] if articles else [])},
            quote="新闻元数据来自 GDELT DOC API（URL 可追溯）。",
        )
        pack.sources.append(src_meta)

        for a in articles:
            url = a.get("url")
            title = a.get("title") or "news"
            seendate = a.get("seendate")  # 20240101123000
            pub = parse_date_yyyymmdd(seendate[0:8]) if seendate and len(seendate) >= 8 else None
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

    # -------- RSS --------
    feeds: List[str] = []
    if opt and opt.rss_feeds:
        feeds = [x.strip() for x in opt.rss_feeds if x and x.strip()]
    elif RSS_FEEDS:
        feeds = [x.strip() for x in RSS_FEEDS.split(",") if x.strip()]

    rss_max_items = (opt.rss_max_items if opt and opt.rss_max_items is not None else RSS_MAX_ITEMS)

    if feeds:
        rss_src = build_source(
            run_id=ctx.run_id,
            source_type="news",
            provider="rss",
            url=",".join(feeds[:5]) + ("..." if len(feeds) > 5 else ""),
            title=f"{keyword} RSS 聚合（{len(feeds)} feeds）",
            raw={"feeds": feeds, "max_items": rss_max_items},
            quote="新闻元数据来自 RSS/Atom（URL 可追溯）。",
        )
        pack.sources.append(rss_src)

        seen_links = set()
        total_added = 0
        for feed_url in feeds:
            if total_added >= rss_max_items:
                break
            try:
                xml_text = await http_get_text(feed_url, headers={"User-Agent": HTTP_UA})
                items = parse_rss_or_atom(xml_text)
            except Exception as e:
                pack.warnings.append(f"RSS 拉取失败：{feed_url} -> {e}")
                continue

            for it in items:
                if total_added >= rss_max_items:
                    break
                link = (it.get("link") or "").strip()
                if not link or link in seen_links:
                    continue
                seen_links.add(link)
                title = it.get("title") or "news"
                pub = try_parse_pub_date(it.get("pub_raw") or "")
                summary = it.get("summary") or None

                s = build_source(
                    run_id=ctx.run_id,
                    source_type="news",
                    provider="rss",
                    url=link,
                    title=title,
                    raw={"feed": feed_url, "item": it},
                    published_at=pub,
                    quote=summary[:280] if summary else None,
                )
                pack.sources.append(s)
                total_added += 1

    if not keyword:
        pack.warnings.append("news：缺少 name/ticker 作为关键词，GDELT 未查询。")
    if not feeds and not RSS_FEEDS:
        pack.warnings.append("news：未配置 RSS_FEEDS（可选），仅使用 GDELT（如可用）。")

    return pack

async def collect_filings(ctx: RunContext, opt: Optional[CollectOptions]) -> EvidencePack:
    pack = EvidencePack(run_id=ctx.run_id, ticker=ctx.ticker, asof=ctx.asof)

    code = (ctx.ticker or "").strip()
    if not code:
        pack.warnings.append("filings：缺少 ticker")
        return pack

    cninfo_days = (opt.cninfo_days if opt and opt.cninfo_days is not None else 365)
    page_size = (opt.cninfo_page_size if opt and opt.cninfo_page_size is not None else CNINFO_PAGE_SIZE)
    max_pages = (opt.cninfo_max_pages if opt and opt.cninfo_max_pages is not None else CNINFO_MAX_PAGES)

    url = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
    headers = {
        "User-Agent": CNINFO_USER_AGENT,
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.cninfo.com.cn/",
        "Origin": "https://www.cninfo.com.cn",
    }
    if CNINFO_COOKIE:
        headers["Cookie"] = CNINFO_COOKIE

    end = (ctx.asof or date.today())
    start = end - timedelta(days=int(cninfo_days))
    seDate = f"{start.strftime('%Y-%m-%d')}~{end.strftime('%Y-%m-%d')}"

    all_ann = []
    last_error = None

    for page in range(1, max_pages + 1):
        form = {
            "pageNum": str(page),
            "pageSize": str(page_size),
            "tabName": "fulltext",
            "seDate": seDate,
            "stock": code,
        }
        try:
            resp = await http_post_form(url, data=form, headers=headers)
            ann = resp.get("announcements") or resp.get("data") or []
            if not ann:
                break
            all_ann.extend(ann)
        except Exception as e:
            last_error = e
            break

    if last_error and not all_ann:
        pack.warnings.append(f"CNINFO 请求失败（可能风控/限流/验证码）：{last_error}")
        return pack

    src_meta = build_source(
        run_id=ctx.run_id,
        source_type="announcement",
        provider="cninfo",
        url=url,
        title=f"{code} 公告列表（CNINFO 元数据）",
        raw={"seDate": seDate, "page_size": page_size, "pages": max_pages, "count": len(all_ann), "sample": all_ann[:3]},
        quote="公告元数据来自 CNINFO 列表接口（URL 可追溯）。",
    )
    pack.sources.append(src_meta)

    for x in all_ann:
        title = x.get("announcementTitle") or x.get("title") or "announcement"
        adjunct = x.get("adjunctUrl") or x.get("url")
        pub_ms = x.get("announcementTime")
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

async def collect_peers(ctx: RunContext, opt: Optional[CollectOptions]) -> EvidencePack:
    pack = EvidencePack(run_id=ctx.run_id, ticker=ctx.ticker, asof=ctx.asof)
    if not TUSHARE_TOKEN:
        pack.warnings.append("TUSHARE_TOKEN 未配置：peers 未抓取。")
        return pack

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

    peers = []
    for it in items:
        row = dict(zip(fields, it))
        if row.get("industry") == my_ind and row.get("ts_code") != my_ts:
            peers.append(row)
        if len(peers) >= 20:
            break

    src = build_source(
        run_id=ctx.run_id,
        source_type="industry_report",
        provider="tushare",
        url="https://api.tushare.pro?api_name=stock_basic",
        title=f"{ctx.ticker} 同行池（industry={my_ind}）",
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

    opt = req.options

    packs: List[EvidencePack] = []
    for c in req.collectors:
        fn = COLLECTOR_MAP.get(c)
        if not fn:
            raise HTTPException(status_code=400, detail=f"Unknown collector: {c}")
        packs.append(await fn(ctx, opt))

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
