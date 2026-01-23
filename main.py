# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pydantic import field_validator
from typing import List, Literal, Optional, Dict, Any
from datetime import datetime, date
import hashlib, json, os, requests

# -------------------------
# Config
# -------------------------
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "").strip()
LIXINGER_TOKEN = os.getenv("LIXINGER_TOKEN", "").strip()
# 你可以继续加：CNINFO_KEY / RSS_LIST / GDELT 等

CollectorName = Literal["financials", "valuation", "filings", "news", "peers"]

# -------------------------
# Helpers
# -------------------------
def md5_text(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)

def make_source_id(run_id: str, provider: str, stype: str, url_hash: str) -> str:
    # 让 source_id 可复现（同 run_id+url_hash 一定同一个）
    return f"src_{provider}_{stype}__{run_id}__{url_hash[:16]}"

# -------------------------
# Models (align DB)
# -------------------------
class RunContext(BaseModel):
    run_id: str
    ticker: str
    name: Optional[str] = None
    asof: Optional[date] = None

    # 解决 n8n 传空字符串导致 422
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
    # 对齐 sources 表字段
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
    # 对齐 normalized_facts 表字段
    run_id: str
    entity_kind: str = "company"  # company|peer|industry|macro
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
    sources: List[SourceRow] = []
    facts: List[FactRow] = []
    peers: List[PeerRow] = []
    warnings: List[str] = []

app = FastAPI(title="Collector Service", version="0.2.0")

# -------------------------
# Providers (skeleton)
# -------------------------

def tushare_post(api_name: str, params: Dict[str, Any], fields: str = "") -> Dict[str, Any]:
    """
    Tushare 官方接口是 POST 到 api.tushare.pro
    返回 JSON：code/msg/data
    """
    url = "https://api.tushare.pro"
    payload = {
        "api_name": api_name,
        "token": TUSHARE_TOKEN,
        "params": params,
    }
    if fields:
        payload["fields"] = fields
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()

def build_source(run_id: str, stype: str, provider: str, url: str,
                 title: str, raw: Dict[str, Any],
                 published_at: Optional[date] = None,
                 quote: Optional[str] = None) -> SourceRow:
    url_hash = md5_text(url)
    source_id = make_source_id(run_id, provider, stype, url_hash)
    retrieved_at = datetime.utcnow()
    content_hash = sha256_text(safe_json(raw))
    return SourceRow(
        source_id=source_id,
        run_id=run_id,
        type=stype,
        title=title,
        url=url,
        url_hash=url_hash,
        published_at=published_at,
        quote=quote,
        retrieved_at=retrieved_at,
        content_hash=content_hash,
        raw=raw,
    )

# -------------------------
# Collectors
# -------------------------

async def collect_financials(ctx: RunContext) -> EvidencePack:
    """
    目标：三表 + 财务指标（最小集即可），全部写 facts，并且 source 可追溯。
    """
    pack = EvidencePack(run_id=ctx.run_id, ticker=ctx.ticker, asof=ctx.asof)

    if not TUSHARE_TOKEN:
        pack.warnings.append("TUSHARE_TOKEN 未配置：financials 未抓取任何数据。")
        return pack

    # 这里用“tushare 返回 raw + 指标 facts”的思路：
    # 你可以从 3-5 年开始（例如 2020-2024），逐步扩展季度/TTM
    years = [2021, 2022, 2023]  # MVP：先 3 年

    # 1) 利润表 income
    income_raw_all = []
    for y in years:
        raw = tushare_post("income", {"ts_code": f"{ctx.ticker}.SH" if ctx.ticker.startswith("6") else f"{ctx.ticker}.SZ",
                                      "period": f"{y}1231"}, fields="")
        income_raw_all.append({"year": y, "resp": raw})

    src_income = build_source(
        run_id=ctx.run_id,
        stype="financial_data",
        provider="tushare",
        url=f"https://api.tushare.pro?api_name=income&ts_code={ctx.ticker}",
        title=f"{ctx.ticker} 利润表（Tushare）",
        raw={"api": "income", "years": years, "data": income_raw_all},
        published_at=None,
        quote="结构化财务接口返回（可审计 raw）。"
    )
    pack.sources.append(src_income)

    # 将 income 的关键字段映射为 facts（注意：tushare data 是 fields + items）
    # 为了不“编造”，只有在拿到可解析数据时才写 facts
    for item in income_raw_all:
        y = item["year"]
        resp = item["resp"]
        if resp.get("code") != 0 or not resp.get("data"):
            pack.warnings.append(f"income {y} 拉取失败或无数据：{resp.get('msg')}")
            continue
        data = resp["data"]
        fields = data.get("fields", [])
        items = data.get("items", [])
        if not fields or not items:
            pack.warnings.append(f"income {y} 返回空 fields/items")
            continue
        row = dict(zip(fields, items[0]))  # period=年末通常一条
        # 你可以按需要扩展更多字段
        mapping = {
            "revenue": row.get("revenue"),
            "net_profit": row.get("n_income"),
            "net_profit_parent": row.get("n_income_attr_p"),
        }
        for metric, val in mapping.items():
            if val is None:
                continue
            try:
                v = float(val)
            except Exception:
                continue
            pack.facts.append(FactRow(
                run_id=ctx.run_id,
                entity_kind="company",
                entity_ticker=ctx.ticker,
                entity_name=ctx.name,
                metric=metric,
                value=v,
                period=f"FY{y}",
                unit="CNY",
                currency="CNY",
                basis="consolidated",
                asof_date=date(y, 12, 31),
                source_id=src_income.source_id
            ))

    # 2) 现金流 cashflow（示例同理）
    # 你可以照抄 income 的模式再做 cashflow / balancesheet / fina_indicator

    return pack

async def collect_valuation(ctx: RunContext) -> EvidencePack:
    """
    估值：优先做“当前PE/PB/PS + 10y分位”（数据源你可选：理杏仁 / 自算 / 其他）
    未配置就返回 warnings，不输出假数据。
    """
    pack = EvidencePack(run_id=ctx.run_id, ticker=ctx.ticker, asof=ctx.asof)
    # 暂不强绑某一家，以免你没 token 就全是假
    pack.warnings.append("valuation 尚未接入真实数据源：未输出任何估值 facts。")
    return pack

async def collect_filings(ctx: RunContext) -> EvidencePack:
    """
    公告/年报：建议落到 sources（annual_report / announcement）
    这里先不写具体抓取（不同站点授权/反爬差异大），但接口结构已对齐。
    """
    pack = EvidencePack(run_id=ctx.run_id, ticker=ctx.ticker, asof=ctx.asof)
    pack.warnings.append("filings 尚未接入公告/年报来源：未输出 sources。")
    return pack

async def collect_news(ctx: RunContext) -> EvidencePack:
    """
    新闻：必须落 sources(type=news)，带 url/published_at/raw(元数据)
    """
    pack = EvidencePack(run_id=ctx.run_id, ticker=ctx.ticker, asof=ctx.asof)
    pack.warnings.append("news 尚未接入新闻来源：未输出 sources。")
    return pack

async def collect_peers(ctx: RunContext) -> EvidencePack:
    """
    同行：建议按行业分类/指数成分生成 peers 表
    """
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
    merged_sources = []
    for p in packs:
        for s in p.sources:
            if s.source_id in seen_sid:
                continue
            seen_sid.add(s.source_id)
            merged_sources.append(s)
    out.sources = merged_sources

    # facts（可再做更强去重：entity+metric+period+basis）
    out.facts = [f for p in packs for f in p.facts]

    # peers 去重
    seen_peer = set()
    peers = []
    for p in packs:
        for pe in p.peers:
            if pe.peer_ticker in seen_peer:
                continue
            seen_peer.add(pe.peer_ticker)
            peers.append(pe)
    out.peers = peers

    out.warnings = [w for p in packs for w in p.warnings]
    return out

@app.post("/collect/evidence-pack", response_model=EvidencePack)
async def collect_evidence(req: CollectRequest):
    ctx = req.run_context
    packs: List[EvidencePack] = []
    for c in req.collectors:
        fn = COLLECTOR_MAP.get(c)
        if not fn:
            raise HTTPException(status_code=400, detail=f"Unknown collector: {c}")
        packs.append(await fn(ctx))

    if not packs:
        return EvidencePack(run_id=ctx.run_id, ticker=ctx.ticker, asof=ctx.asof)

    out = merge_packs(packs)

    # 真实性护栏 1：facts 的 source_id 必须在 sources 里，否则打 warning（建议 n8n 直接拒绝入库 facts）
    source_ids = {s.source_id for s in out.sources}
    bad_facts = [f for f in out.facts if f.source_id not in source_ids]
    if bad_facts:
        out.warnings.append(f"{len(bad_facts)} 条 facts 缺失 source_id 对应 sources：建议拒绝写入 normalized_facts。")

    # 真实性护栏 2：sources 必须有 url_hash/content_hash
    bad_src = [s for s in out.sources if not s.url_hash or not s.content_hash]
    if bad_src:
        out.warnings.append(f"{len(bad_src)} 条 sources 缺少 hash 字段：建议拒绝写入 sources。")

    return out
