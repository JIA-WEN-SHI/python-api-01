# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List, Literal, Optional, Dict, Any
from datetime import datetime, date
import hashlib, json, os

import httpx  # ✅ 用 httpx 避免 async 里 requests 阻塞

# =========================================================
# Config
# =========================================================
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "").strip()
LIXINGER_TOKEN = os.getenv("LIXINGER_TOKEN", "").strip()  # 预留
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "30"))

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
    支持传入：
    - 600519
    - 600519.SH
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
    return t  # 兜底

def make_source_id(run_id: str, provider: str, stype: str, url_hash: str) -> str:
    # ✅可复现：同 run_id + stype + url_hash => 同一个 source_id
    return f"src_{provider}_{stype}__{run_id}__{url_hash[:16]}"

async def tushare_post(api_name: str, params: Dict[str, Any], fields: str = "") -> Dict[str, Any]:
    """
    Tushare：POST https://api.tushare.pro
    返回：{"code":0,"msg":"","data":{"fields":[...],"items":[...]}}
    """
    if not TUSHARE_TOKEN:
        return {"code": -1, "msg": "TUSHARE_TOKEN missing", "data": None}

    payload: Dict[str, Any] = {
        "api_name": api_name,
        "token": TUSHARE_TOKEN,
        "params": params,
    }
    if fields:
        payload["fields"] = fields

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        r = await client.post("https://api.tushare.pro", json=payload)
        r.raise_for_status()
        return r.json()

# =========================================================
# Models (align DB)
# =========================================================
class RunContext(BaseModel):
    run_id: str
    ticker: str
    name: Optional[str] = None
    asof: Optional[date] = None

    # ✅解决 n8n 传空字符串导致 422
    @field_validator("asof", mode="before")
    @classmethod
    def empty_str_to_none(cls, v):
        if v == "" or v is None:
            return None
        return v

class CollectRequest(BaseModel):
    run_context: RunContext
    collectors: List[CollectorName] = Field(
        default_factory=lambda: ["financials", "valuation", "filings", "news", "peers"]
    )

class SourceRow(BaseModel):
    # sources 表字段（你库里 url_hash 是 generated，但这里返回给 n8n 方便写库/校验）
    source_id: str
    run_id: str
    type: str  # ✅统一用：annual_report / announcement / news / policy / industry_report / research_report / market_data / valuation_data ...
    title: Optional[str] = None
    url: str
    url_hash: str
    published_at: Optional[date] = None
    quote: Optional[str] = None
    retrieved_at: datetime
    content_hash: str
    raw: Optional[Dict[str, Any]] = None

class FactRow(BaseModel):
    # normalized_facts 表字段
    run_id: str
    entity_kind: str = "company"  # company|peer|industry|macro
    entity_ticker: Optional[str] = None
    entity_name: Optional[str] = None

    metric: str
    value: float
    period: str  # FY2023 / 2024Q1 / TTM / current

    unit: Optional[str] = None       # cny / pct / times / shares / ...
    currency: Optional[str] = None   # CNY / USD / ...
    basis: Optional[str] = None      # consolidated / parent / ...
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
app = FastAPI(title="Collector Service", version="0.3.0")

@app.get("/health")
async def health():
    return {"ok": True, "ts": datetime.utcnow().isoformat()}

def build_source(
    run_id: str,
    source_type: str,   # ✅用 sources.type 的语义（market_data / valuation_data / news / annual_report...）
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
    ✅最小闭环：Tushare income 三年（FY）+ 可审计 raw
    - sources.type = market_data（统一给 Phase6/Phase3 使用）
    - facts.period = FY2021 / FY2022 / FY2023（统一）
    """
    pack = EvidencePack(run_id=ctx.run_id, ticker=ctx.ticker, asof=ctx.asof)

    if not TUSHARE_TOKEN:
        pack.warnings.append("TUSHARE_TOKEN 未配置：financials 未抓取任何数据。")
        return pack

    ts_code = normalize_ts_code(ctx.ticker)
    if "." not in ts_code:
        pack.warnings.append(f"ticker 无法规范化为 ts_code：{ctx.ticker}")
        return pack

    years = [2021, 2022, 2023]  # MVP：先 3 年
    income_raw_all: List[Dict[str, Any]] = []

    for y in years:
        resp = await tushare_post(
            "income",
            {"ts_code": ts_code, "period": f"{y}1231"},
            fields="",  # 你可以后续收紧 fields
        )
        income_raw_all.append({"year": y, "resp": resp})

    # ✅source 归类：market_data（你也可以叫 financial_data，但要全系统统一）
    src_income = build_source(
        run_id=ctx.run_id,
        source_type="market_data",
        provider="tushare",
        url=f"https://api.tushare.pro?api_name=income&ts_code={ts_code}",
        title=f"{ts_code} 利润表（Tushare income）",
        raw={"provider": "tushare", "api": "income", "ts_code": ts_code, "years": years, "data": income_raw_all},
        published_at=None,
        quote="结构化财务接口返回（raw 可审计）。",
    )
    pack.sources.append(src_income)

    # facts 映射（只在可解析时写，避免“编造”）
    for item in income_raw_all:
        y = item["year"]
        resp = item["resp"] or {}
        if resp.get("code") != 0 or not resp.get("data"):
            pack.warnings.append(f"income FY{y} 拉取失败或无数据：{resp.get('msg')}")
            continue

        data = resp.get("data") or {}
        fields = data.get("fields") or []
        items = data.get("items") or []
        if not fields or not items:
            pack.warnings.append(f"income FY{y} 返回空 fields/items")
            continue

        # 有时会返回多行，取第一行（年末口径通常一行）
        row = dict(zip(fields, items[0]))

        # ✅字段名以 tushare 实际返回为准；拿不到就不写
        mapping = {
            "revenue": row.get("revenue"),
            "net_profit": row.get("n_income"),
            "net_profit_parent": row.get("n_income_attr_p"),
        }

        # 给你一个“自检提示”：第一次跑你就知道字段是否存在
        if y == years[-1]:
            pack.warnings.append(f"income fields sample: {fields[:30]}")

        for metric, val in mapping.items():
            if val is None:
                continue
            try:
                v = float(val)
            except Exception:
                continue

            pack.facts.append(
                FactRow(
                    run_id=ctx.run_id,
                    entity_kind="company",
                    entity_ticker=ctx.ticker,  # 保留原 ticker（你的系统内部主键）
                    entity_name=ctx.name,
                    metric=metric,
                    value=v,
                    period=f"FY{y}",
                    unit="cny",       # ✅unit != currency
                    currency="CNY",
                    basis="consolidated",
                    asof_date=date(y, 12, 31),
                    source_id=src_income.source_id,
                )
            )

    return pack

async def collect_valuation(ctx: RunContext) -> EvidencePack:
    pack = EvidencePack(run_id=ctx.run_id, ticker=ctx.ticker, asof=ctx.asof)
    pack.warnings.append("valuation 尚未接入真实数据源：未输出任何估值 facts。")
    return pack

async def collect_filings(ctx: RunContext) -> EvidencePack:
    pack = EvidencePack(run_id=ctx.run_id, ticker=ctx.ticker, asof=ctx.asof)
    pack.warnings.append("filings 尚未接入公告/年报来源：未输出 sources。")
    return pack

async def collect_news(ctx: RunContext) -> EvidencePack:
    pack = EvidencePack(run_id=ctx.run_id, ticker=ctx.ticker, asof=ctx.asof)
    pack.warnings.append("news 尚未接入新闻来源：未输出 sources。")
    return pack

async def collect_peers(ctx: RunContext) -> EvidencePack:
    pack = EvidencePack(run_id=ctx.run_id, ticker=ctx.ticker, asof=ctx.asof)
    pack.warnings.append("peers 尚未接入行业/成分股来源：未输出 peers。")
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

    # sources 去重（按 source_id）
    seen_sid = set()
    merged_sources: List[SourceRow] = []
    for p in packs:
        for s in p.sources:
            if s.source_id in seen_sid:
                continue
            seen_sid.add(s.source_id)
            merged_sources.append(s)
    out.sources = merged_sources

    # facts 合并（进一步去重交给 DB 唯一索引更稳）
    out.facts = [f for p in packs for f in p.facts]

    # peers 去重
    seen_peer = set()
    merged_peers: List[PeerRow] = []
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

    if not packs:
        return EvidencePack(run_id=ctx.run_id, ticker=ctx.ticker, asof=ctx.asof)

    out = merge_packs(packs)

    # 真实性护栏 1：facts.source_id 必须在 sources 里
    source_ids = {s.source_id for s in out.sources}
    bad_facts = [f for f in out.facts if f.source_id not in source_ids]
    if bad_facts:
        out.warnings.append(
            f"{len(bad_facts)} 条 facts 缺失 source_id 对应 sources：建议 n8n 拒绝写入 normalized_facts。"
        )

    # 真实性护栏 2：sources 必须有 url_hash/content_hash
    bad_src = [s for s in out.sources if not s.url_hash or not s.content_hash]
    if bad_src:
        out.warnings.append(
            f"{len(bad_src)} 条 sources 缺少 hash 字段：建议 n8n 拒绝写入 sources。"
        )

    return out
