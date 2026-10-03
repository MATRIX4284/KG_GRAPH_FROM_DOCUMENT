"""
Document -> GPT-5.1 (Azure AI Foundry) -> triplets -> knowledge graph.

Usage:
    az login                                         # Entra ID sign-in
    python kg_pipeline.py my_document.pdf            # NetworkX + HTML viz
    python kg_pipeline.py my_document.pdf --neo4j    # also load into Neo4j

Outputs:
    triplets.json   - raw extracted triplets
    graph.graphml   - graph file (Gephi, yEd, Cytoscape)
    graph.html      - interactive graph in the browser
"""

import json
import os
import re
import sys

import networkx as nx
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import OpenAI
from pydantic import BaseModel, Field
from pypdf import PdfReader
from pyvis.network import Network

# ---------------------------------------------------------------------------
# 1. Azure AI Foundry client
# ---------------------------------------------------------------------------
# Signs in with Microsoft Entra ID (run `az login` first) - no API key needed.
ENDPOINT = os.environ.get(
    "AZURE_FOUNDRY_ENDPOINT",
    "https://kaustav-foundry-reaource.services.ai.azure.com/openai/v1",
)
DEPLOYMENT = os.environ.get("AZURE_FOUNDRY_DEPLOYMENT", "gpt-5.1")

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://ai.azure.com/.default"
)
client = OpenAI(base_url=ENDPOINT, api_key=token_provider)


# ---------------------------------------------------------------------------
# 2. Schema for structured output
# ---------------------------------------------------------------------------
class Triplet(BaseModel):
    subject: str = Field(description="Canonical entity name")
    subject_type: str = Field(description="Person, Organization, Product, Location, Event, Concept, Date, ...")
    predicate: str = Field(description="snake_case relation, e.g. founded_by, located_in")
    object: str = Field(description="Canonical entity name or literal value")
    object_type: str
    evidence: str = Field(description="Short verbatim span from the text supporting the fact")


class TripletList(BaseModel):
    triplets: list[Triplet]


SYSTEM_PROMPT = """You are an information-extraction engine building a knowledge graph.
Extract factual (subject, predicate, object) triplets from the text.
Rules:
- Only facts explicitly stated in the text; no outside knowledge.
- Resolve pronouns/coreferences to the full entity name.
- Atomic facts only: split compound statements.
- Concise snake_case predicates; reuse the same predicate for the same relation.
- Normalize entity names ("Microsoft Corp." -> "Microsoft").
- Use the full name without titles or suffixes ("Dr. Jane Smith" -> "Jane Smith").
- Never use generic phrases ("the company", "the firm") as entities; use the name.
- Every entity you mention should be linked to at least one other named entity:
  attach dates, amounts and places to the entity they describe.
- If an entity matches one in KNOWN ENTITIES, use exactly that name."""


class AliasGroup(BaseModel):
    canonical: str = Field(description="Best full name for the entity")
    aliases: list[str] = Field(description="Other names in the list that mean the same entity")


class AliasMap(BaseModel):
    groups: list[AliasGroup]


RESOLVE_PROMPT = """You are an entity-resolution engine.
Given a list of entity names from one document, group names that refer to the
SAME real-world entity (e.g. "Dr. Maya Okafor", "Okafor", "Maya Okafor";
"Austin", "Austin, Texas"). Pick the most complete name without titles as canonical.
Only group when you are confident. Do NOT merge related-but-different entities
(a company and its subsidiary, a product and its successor, a lab and its venture arm).
Omit entities that have no aliases."""

BRIDGE_PROMPT = """You are completing a knowledge graph built from the TEXT.
The graph is split into a MAIN group of entities and several ISOLATED groups.
Find relationships, stated in the TEXT, that connect each isolated group to
entities in the main group (or to each other). Use entity names exactly as listed.
Only return facts supported by the TEXT, with a verbatim evidence span.
If no supported link exists for a group, skip it."""


# ---------------------------------------------------------------------------
# 3. Extraction
# ---------------------------------------------------------------------------
def load_text(path: str) -> str:
    if path.lower().endswith(".pdf"):
        return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
    with open(path, encoding="utf-8") as f:
        return f.read()


def chunk(text: str, size: int = 6000, overlap: int = 500) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size - overlap)]


def extract_triplets(text: str, known_entities: list[str]) -> list[Triplet]:
    # Passing already-seen entities keeps naming consistent across chunks.
    known = "\n".join(known_entities[-300:]) or "(none yet)"
    resp = client.responses.parse(
        model=DEPLOYMENT,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"KNOWN ENTITIES:\n{known}\n\nTEXT:\n{text}"},
        ],
        text_format=TripletList,
        reasoning={"effort": "low"},  # none | low | medium | high
    )
    return resp.output_parsed.triplets


def extract_from_text(text: str) -> list[Triplet]:
    chunks = chunk(text)
    seen, results, entities = set(), [], {}
    for i, c in enumerate(chunks, 1):
        print(f"Extracting chunk {i}/{len(chunks)} ...")
        for t in extract_triplets(c, list(entities.values())):
            # Case-insensitive entity resolution: first spelling wins.
            t.subject = entities.setdefault(t.subject.casefold(), t.subject)
            t.object = entities.setdefault(t.object.casefold(), t.object)
            key = (t.subject, t.predicate, t.object)
            if key not in seen:
                seen.add(key)
                results.append(t)
    return results


def dedupe(triplets: list[Triplet]) -> list[Triplet]:
    seen, out = set(), []
    for t in triplets:
        key = (t.subject, t.predicate, t.object)
        if t.subject != t.object and key not in seen:
            seen.add(key)
            out.append(t)
    return out


def resolve_entities(triplets: list[Triplet]) -> list[Triplet]:
    """Merge alias nodes ("Okafor" / "Dr. Maya Okafor") - the main cause of islands."""
    names = sorted({t.subject for t in triplets} | {t.object for t in triplets})
    resp = client.responses.parse(
        model=DEPLOYMENT,
        input=[
            {"role": "system", "content": RESOLVE_PROMPT},
            {"role": "user", "content": "\n".join(names)},
        ],
        text_format=AliasMap,
        reasoning={"effort": "medium"},
    )
    mapping = {}
    for grp in resp.output_parsed.groups:
        for alias in grp.aliases:
            mapping[alias] = grp.canonical
    print(f"Entity resolution merged {len(mapping)} aliases")
    for t in triplets:
        t.subject = mapping.get(t.subject, t.subject)
        t.object = mapping.get(t.object, t.object)
    return dedupe(triplets)


def bridge_components(triplets: list[Triplet], text: str, max_rounds: int = 2) -> list[Triplet]:
    """Ask the model for text-supported links between disconnected subgraphs."""
    for _ in range(max_rounds):
        comps = sorted(nx.weakly_connected_components(build_graph(triplets)),
                       key=len, reverse=True)
        if len(comps) <= 1:
            break
        main = "\n".join(sorted(comps[0])[:300])
        isolated = "\n".join(f"Group {i}: " + "; ".join(sorted(c))
                             for i, c in enumerate(comps[1:50], 1))
        resp = client.responses.parse(
            model=DEPLOYMENT,
            input=[
                {"role": "system", "content": BRIDGE_PROMPT},
                {"role": "user", "content": f"MAIN GROUP:\n{main}\n\nISOLATED GROUPS:\n"
                                            f"{isolated}\n\nTEXT:\n{text[:150_000]}"},
            ],
            text_format=TripletList,
            reasoning={"effort": "medium"},
        )
        new = resp.output_parsed.triplets
        print(f"Bridging: {len(comps)} components, model proposed {len(new)} links")
        if not new:
            break
        triplets = dedupe(triplets + new)
    return triplets


def report_connectivity(g: nx.MultiDiGraph, label: str) -> None:
    comps = sorted(nx.weakly_connected_components(g), key=len, reverse=True)
    print(f"[{label}] {len(comps)} connected components; largest has "
          f"{len(comps[0]) if comps else 0}/{g.number_of_nodes()} nodes")


# ---------------------------------------------------------------------------
# 4. Build the knowledge graph (NetworkX)
# ---------------------------------------------------------------------------
def build_graph(triplets: list[Triplet]) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for t in triplets:
        g.add_node(t.subject, type=t.subject_type)
        g.add_node(t.object, type=t.object_type)
        g.add_edge(t.subject, t.object, key=t.predicate,
                   predicate=t.predicate, evidence=t.evidence)
    return g


PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
           "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]


def visualize(g: nx.MultiDiGraph, out: str = "graph.html") -> None:
    types = sorted({d["type"] for _, d in g.nodes(data=True)})
    color = {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(types)}
    net = Network(height="800px", width="100%", directed=True, bgcolor="#ffffff")
    for n, d in g.nodes(data=True):
        net.add_node(n, label=n, title=f"{n} ({d['type']})",
                     color=color[d["type"]], size=10 + 3 * g.degree(n))
    for s, o, d in g.edges(data=True):
        net.add_edge(s, o, label=d["predicate"], title=d["evidence"])
    net.force_atlas_2based()
    net.write_html(out, open_browser=False)


# ---------------------------------------------------------------------------
# 5. Optional: load into Neo4j  (pip install neo4j)
# ---------------------------------------------------------------------------
def load_into_neo4j(triplets: list[Triplet]) -> None:
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.environ.get("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"]),
    )
    driver.execute_query("CREATE CONSTRAINT entity_name IF NOT EXISTS "
                         "FOR (e:Entity) REQUIRE e.name IS UNIQUE")
    for t in triplets:
        # Relationship types can't be parameters in Cypher, so sanitize first.
        rel = re.sub(r"[^A-Za-z0-9_]", "_", t.predicate).upper() or "RELATED_TO"
        driver.execute_query(
            f"""
            MERGE (s:Entity {{name: $s}}) SET s.type = $st
            MERGE (o:Entity {{name: $o}}) SET o.type = $ot
            MERGE (s)-[r:`{rel}`]->(o) SET r.evidence = $ev
            """,
            s=t.subject, st=t.subject_type, o=t.object, ot=t.object_type, ev=t.evidence,
        )
    driver.close()
    print("Loaded into Neo4j. Try: MATCH (s)-[r]->(o) RETURN s, r, o LIMIT 100")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    doc = sys.argv[1]
    text = load_text(doc)
    triplets = extract_from_text(text)
    report_connectivity(build_graph(triplets), "after extraction")

    triplets = resolve_entities(triplets)
    report_connectivity(build_graph(triplets), "after entity resolution")

    triplets = bridge_components(triplets, text)
    report_connectivity(build_graph(triplets), "after bridging")

    with open("triplets.json", "w") as f:
        json.dump([t.model_dump() for t in triplets], f, indent=2)

    g = build_graph(triplets)
    nx.write_graphml(g, "graph.graphml")
    visualize(g)

    print(f"{len(triplets)} triplets -> {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    top = sorted(g.degree, key=lambda x: x[1], reverse=True)[:10]
    print("Most connected entities:", ", ".join(f"{n} ({d})" for n, d in top))

    if "--neo4j" in sys.argv:
        load_into_neo4j(triplets)