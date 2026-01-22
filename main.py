# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any
from datetime import datetime, date
import hashlib, json, os

CollectorName = Literal["financials","valuation","filings","news","peers"]

class RunContext(BaseModel):
    run_id: str
    ticker: str
    name: Optional[str] = None
    asof: Optional[date] = None

class CollectRequest(BaseModel):
    run_context: RunContext
    collectors: List[CollectorName] = Field(default_factory=lambda: ["financials","valuation","filings","news","peers"])

class Source(BaseModel):
    source_id: str
    run_id: str
    type: str
    title: str
    url: str
    published_at: Optional[date] = None
    quote: Optional[str] = None
    retrieved_at: datetime
    content_hash: str
    raw: Optional[Dict[str, Any]] = None

class Fact(BaseModel):
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

class Peer(BaseModel):
    peer_ticker: str
    peer_name: Optional[str] = None

class EvidencePack(BaseModel):
    run_id: str
    ticker: str
    asof: Optional[date] = None
    sources: List[Source] = []
    facts: List[Fact] = []
    peers: List[Peer] = []
    warnings: List[str] = []

app = FastAPI(title="Collector Service", version="0.1.0")

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def make_source_id(run_id: str, provider: str, stype: str, url: str, extra: Dict[str, Any]) -> str:
    h = sha256_text(url + "|" + json.dumps(extra, ensure_ascii=False, sort_keys=True))
    return f"src_{provider}_{stype}__{run_id}__{h[:16]}"

# ---- 下面这些先写成“占位”，你后面逐步补真实抓取 ----
async def collect_financials(ctx: RunContext) -> EvidencePack:
    pack = EvidencePack(run_id=ctx.run_id, ticker=ctx.ticker, asof=ctx.asof)
    # TODO: 接入 tushare / lixinger / jq / 自有API
    # 这里演示一条“可追溯”的 source + facts
    url = f"https://example.com/financials/{ctx.ticker}"
    raw = {"provider":"demo","kind":"financials","ticker":ctx.ticker}
    sid = make_source_id(ctx.run_id, "demo", "financials", url, raw)
    now = datetime.utcnow()
    pack.sources.append(Source(
        source_id=sid, run_id=ctx.run_id, type="annual_report",
        title=f"{ctx.ticker} 财务数据（示例）", url=url,
        published_at=None, quote="示例：该链接用于占位，后续替换为真实来源。",
        retrieved_at=now, content_hash=sha256_text(json.dumps(raw, ensure_ascii=False)),
        raw=raw
    ))
    pack.facts.append(Fact(
        run_id=ctx.run_id, entity_kind="company", entity_ticker=ctx.ticker, entity_name=ctx.name,
        metric="revenue", value=1.0, period="2023A", unit="CNY", currency="CNY",
        basis="consolidated", asof_date=date(2023,12,31), source_id=sid
    ))
    return pack

async def collect_valuation(ctx: RunContext) -> EvidencePack:
    pack = EvidencePack(run_id=ctx.run_id, ticker=ctx.ticker, asof=ctx.asof)
    # TODO: PE/PB/PS 当前 + 历史序列
    return pack

async def collect_filings(ctx: RunContext) -> EvidencePack:
    pack = EvidencePack(run_id=ctx.run_id, ticker=ctx.ticker, asof=ctx.asof)
    # TODO: 年报/公告抓取：只需要抽“可引用片段 quote”，不要整篇塞进 sources.quote
    return pack

async def collect_news(ctx: RunContext) -> EvidencePack:
    pack = EvidencePack(run_id=ctx.run_id, ticker=ctx.ticker, asof=ctx.asof)
    # TODO: 新闻/RSS/研报摘要：每条一条 source（source_id 唯一）
    return pack

async def collect_peers(ctx: RunContext) -> EvidencePack:
    pack = EvidencePack(run_id=ctx.run_id, ticker=ctx.ticker, asof=ctx.asof)
    # TODO: 先用静态映射/规则：行业=白酒 -> peers
    pack.peers = []
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
    seen_source = set()
    # 合并 sources 去重
    merged_sources = []
    for p in packs:
        for s in p.sources:
            if s.source_id in seen_source: 
                continue
            seen_source.add(s.source_id)
            merged_sources.append(s)
    out.sources = merged_sources
    # 合并 facts（你也可以按 (metric,period,entity) 去重）
    out.facts = [f for p in packs for f in p.facts]
    # 合并 peers 去重
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
    packs = []
    for c in req.collectors:
        fn = COLLECTOR_MAP.get(c)
        if not fn:
            raise HTTPException(status_code=400, detail=f"Unknown collector: {c}")
        packs.append(await fn(ctx))
    if not packs:
        return EvidencePack(run_id=ctx.run_id, ticker=ctx.ticker, asof=ctx.asof)
    out = merge_packs(packs)

    # 真实性护栏：facts 的 source_id 必须在 sources 里
    source_ids = {s.source_id for s in out.sources}
    bad = [f for f in out.facts if f.source_id not in source_ids]
    if bad:
        out.warnings.append(f"{len(bad)} facts 缺失 source_id 对应 sources，建议 n8n 拒绝入库。")
    return out
