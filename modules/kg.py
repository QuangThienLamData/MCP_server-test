"""Knowledge Graph extraction inspired by Hyper-Extract / LightRAG.

Two-stage entity→relationship extraction using OpenAI structured output,
stored in Pinecone (namespace ``knowledge_graph``) + SQLite.
Zero new dependencies — reuses existing OpenAI SDK + Pinecone + SQLite.
"""

import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timezone

from modules.rag import DB_PATH, OPENAI_API_KEY, PINECONE_API_KEY, _embed, _get_index, _get_openai

logger = logging.getLogger(__name__)

KG_NAMESPACE = "knowledge_graph"

# ---------------------------------------------------------------------------
# Prompts (adapted from LightRAG / Hyper-Extract two-stage pattern)
# ---------------------------------------------------------------------------

ENTITY_EXTRACTION_PROMPT = """\
You are an expert knowledge-graph extraction assistant for competitor \
intelligence in the fintech / e-wallet / digital-payment industry.

Extract ALL significant entities from the source text.

Entity types (pick the most specific):
  company      – companies, startups, banks, financial institutions
  product      – named products, apps, services, platforms
  feature      – specific product features or capabilities
  strategy     – business strategies, campaigns, go-to-market moves
  technology   – technologies, APIs, protocols, standards
  person       – key people, executives, founders
  market       – market segments, regions, user demographics
  partnership  – partnerships, alliances, integrations, M&A
  metric       – key numbers, KPIs, funding amounts, user counts (include value)
  regulation   – laws, regulations, licenses, compliance items

Rules:
• Use canonical names (e.g. "ZaloPay" not "zalopay").
• Description = 1-3 sentences capturing what the text says about this entity.
• Aim for 5-15 entities per article; skip trivial mentions.

Reply with JSON only:
{"entities": [{"name": "...", "type": "...", "description": "..."}]}"""

RELATIONSHIP_EXTRACTION_PROMPT = """\
You are an expert knowledge-graph extraction assistant.
Extract relationships between the Known Entities based on the source text.

Rules:
• source and target MUST be names from the Known Entities list below.
• description: explain the specific connection (1-2 sentences).
• keywords: comma-separated summary terms.
• strength: 1-10 (10 = core/defining relationship, 1 = passing mention).
• Extract 3-20 relationships; do NOT invent connections unsupported by the text.

Known Entities:
{entity_list}

Reply with JSON only:
{{"relationships": [{{"source":"...","target":"...","description":"...","keywords":"...","strength":8}}]}}"""

# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def _init_kg_db():
    """Create knowledge-graph tables (idempotent)."""
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS kg_entities (
            id              TEXT PRIMARY KEY,
            name            TEXT NOT NULL UNIQUE,
            type            TEXT NOT NULL,
            description     TEXT NOT NULL,
            competitor_names TEXT DEFAULT '[]',
            source_urls     TEXT DEFAULT '[]',
            updated_at      TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS kg_relationships (
            id          TEXT PRIMARY KEY,
            source      TEXT NOT NULL,
            target      TEXT NOT NULL,
            description TEXT NOT NULL,
            keywords    TEXT DEFAULT '',
            strength    INTEGER DEFAULT 5,
            source_urls TEXT DEFAULT '[]',
            updated_at  TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_kg_rel_source ON kg_relationships(source);
        CREATE INDEX IF NOT EXISTS idx_kg_rel_target ON kg_relationships(target);
    """)
    conn.commit()
    conn.close()


def _entity_id(name: str) -> str:
    return hashlib.md5(name.lower().strip().encode()).hexdigest()


def _rel_id(source: str, target: str) -> str:
    return hashlib.md5(f"{source.lower().strip()}->{target.lower().strip()}".encode()).hexdigest()


def _json_append(existing_json: str, value: str) -> str:
    """Append *value* to a JSON list string if not already present."""
    lst = json.loads(existing_json or "[]")
    if value and value not in lst:
        lst.append(value)
    return json.dumps(lst, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Two-stage LLM extraction
# ---------------------------------------------------------------------------

def _extract_entities(text: str) -> list[dict]:
    """Stage 1 — extract entities from text via GPT-4o-mini structured output."""
    try:
        resp = _get_openai().chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": ENTITY_EXTRACTION_PROMPT},
                {"role": "user", "content": f"Source Text:\n{text[:8000]}"},
            ],
        )
        data = json.loads(resp.choices[0].message.content)
        return [
            e for e in data.get("entities", [])
            if isinstance(e, dict) and e.get("name") and e.get("type") and e.get("description")
        ]
    except Exception as e:
        logger.warning(f"[kg] Entity extraction failed: {e}")
        return []


def _extract_relationships(text: str, entities: list[dict]) -> list[dict]:
    """Stage 2 — extract relationships constrained to known entities."""
    if len(entities) < 2:
        return []
    entity_list = "\n".join(f"- {e['name']} ({e['type']})" for e in entities)
    prompt = RELATIONSHIP_EXTRACTION_PROMPT.replace("{entity_list}", entity_list)
    try:
        resp = _get_openai().chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Source Text:\n{text[:8000]}"},
            ],
        )
        data = json.loads(resp.choices[0].message.content)
        names_lower = {e["name"].lower() for e in entities}
        return [
            r for r in data.get("relationships", [])
            if isinstance(r, dict)
            and r.get("source", "").lower() in names_lower
            and r.get("target", "").lower() in names_lower
            and r.get("description")
        ]
    except Exception as e:
        logger.warning(f"[kg] Relationship extraction failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Store & index
# ---------------------------------------------------------------------------

def _store_entities(entities: list[dict], competitor_name: str, source_url: str):
    """Upsert entities into SQLite + Pinecone (namespace knowledge_graph)."""
    if not entities:
        return
    conn = sqlite3.connect(DB_PATH)
    now = datetime.now(timezone.utc).isoformat()
    index = _get_index()

    texts_to_embed: list[str] = []
    vectors_meta: list[dict] = []

    for e in entities:
        eid = _entity_id(e["name"])
        row = conn.execute(
            "SELECT description, competitor_names, source_urls FROM kg_entities WHERE id = ?",
            (eid,),
        ).fetchone()

        if row:
            old_desc, old_comps_json, old_urls_json = row
            comps_json = _json_append(old_comps_json, competitor_name)
            urls_json = _json_append(old_urls_json, source_url)
            # Simple merge: keep longer description or append new details
            if e["description"].lower().strip() != old_desc.lower().strip():
                if len(e["description"]) > len(old_desc):
                    desc = e["description"]
                else:
                    desc = old_desc
            else:
                desc = old_desc
            conn.execute(
                "UPDATE kg_entities SET description=?, competitor_names=?, source_urls=?, updated_at=? WHERE id=?",
                (desc, comps_json, urls_json, now, eid),
            )
        else:
            desc = e["description"]
            comps_json = json.dumps([competitor_name] if competitor_name else [])
            urls_json = json.dumps([source_url] if source_url else [])
            conn.execute(
                "INSERT INTO kg_entities (id,name,type,description,competitor_names,source_urls,updated_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (eid, e["name"], e["type"], desc, comps_json, urls_json, now),
            )

        embed_text = f"{e['name']} ({e['type']}): {desc}"
        texts_to_embed.append(embed_text)
        vectors_meta.append({
            "id": f"entity_{eid}",
            "metadata": {
                "name": e["name"],
                "type": e["type"],
                "description": desc[:500],
                "competitor_names": comps_json,
                "source_url": source_url or "",
            },
        })

    conn.commit()
    conn.close()

    # Embed and upsert to Pinecone
    if texts_to_embed:
        try:
            embeddings = _embed(texts_to_embed)
            vectors = []
            for meta, emb in zip(vectors_meta, embeddings):
                meta["values"] = emb
                vectors.append(meta)
            for i in range(0, len(vectors), 100):
                index.upsert(vectors=vectors[i : i + 100], namespace=KG_NAMESPACE)
        except Exception as e:
            logger.warning(f"[kg] Entity embedding failed: {e}")


def _store_relationships(relationships: list[dict], source_url: str):
    """Upsert relationships into SQLite."""
    if not relationships:
        return
    conn = sqlite3.connect(DB_PATH)
    now = datetime.now(timezone.utc).isoformat()

    for r in relationships:
        rid = _rel_id(r["source"], r["target"])
        strength = min(max(int(r.get("strength", 5)), 1), 10)
        row = conn.execute("SELECT source_urls FROM kg_relationships WHERE id = ?", (rid,)).fetchone()

        if row:
            urls_json = _json_append(row[0], source_url)
            conn.execute(
                "UPDATE kg_relationships SET description=?, keywords=?, strength=?, source_urls=?, updated_at=? WHERE id=?",
                (r["description"], r.get("keywords", ""), strength, urls_json, now, rid),
            )
        else:
            urls_json = json.dumps([source_url] if source_url else [])
            conn.execute(
                "INSERT INTO kg_relationships (id,source,target,description,keywords,strength,source_urls,updated_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (rid, r["source"], r["target"], r["description"], r.get("keywords", ""), strength, urls_json, now),
            )

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_knowledge(text: str, competitor_name: str = "", source_url: str = ""):
    """Full two-stage extraction pipeline (LightRAG-style).

    Called during crawl to build the knowledge graph incrementally.
    """
    if not OPENAI_API_KEY or not PINECONE_API_KEY:
        return
    entities = _extract_entities(text)
    if not entities:
        return
    logger.info(f"[kg] Extracted {len(entities)} entities from {source_url or 'text'}")
    relationships = _extract_relationships(text, entities)
    logger.info(f"[kg] Extracted {len(relationships)} relationships")
    _store_entities(entities, competitor_name, source_url)
    _store_relationships(relationships, source_url)


def search_kg(query: str, entity_type: str = "", top_k: int = 10) -> list[dict]:
    """Search knowledge-graph entities by semantic similarity (bilingual)."""
    from rag_mcp import _bilingual_queries

    variants = _bilingual_queries(query)
    qvecs = _embed(variants)

    filt = {}
    if entity_type:
        filt["type"] = {"$eq": entity_type}

    index = _get_index()
    merged: dict = {}
    for qvec in qvecs:
        res = index.query(
            vector=qvec, top_k=top_k, include_metadata=True,
            filter=filt if filt else None,
            namespace=KG_NAMESPACE,
        )
        for m in res.matches:
            if m.id not in merged or m.score > merged[m.id].score:
                merged[m.id] = m

    results = sorted(merged.values(), key=lambda m: m.score, reverse=True)[:top_k]
    return [
        {
            "name": m.metadata.get("name", ""),
            "type": m.metadata.get("type", ""),
            "description": m.metadata.get("description", ""),
            "competitor_names": m.metadata.get("competitor_names", "[]"),
            "score": m.score,
        }
        for m in results
    ]


def get_relationships(entity_name: str) -> list[dict]:
    """Get all relationships involving *entity_name* from SQLite."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT source, target, description, keywords, strength "
        "FROM kg_relationships WHERE LOWER(source) = LOWER(?) OR LOWER(target) = LOWER(?)",
        (entity_name, entity_name),
    ).fetchall()
    conn.close()
    return [
        {"source": s, "target": t, "description": d, "keywords": k, "strength": st}
        for s, t, d, k, st in rows
    ]


def get_kg_stats() -> dict:
    """Return knowledge-graph statistics."""
    conn = sqlite3.connect(DB_PATH)
    try:
        ent_count = conn.execute("SELECT COUNT(*) FROM kg_entities").fetchone()[0]
        rel_count = conn.execute("SELECT COUNT(*) FROM kg_relationships").fetchone()[0]
        types = conn.execute(
            "SELECT type, COUNT(*) FROM kg_entities GROUP BY type ORDER BY COUNT(*) DESC"
        ).fetchall()
    except Exception:
        ent_count, rel_count, types = 0, 0, []
    conn.close()
    return {"entities": ent_count, "relationships": rel_count, "types": dict(types)}
