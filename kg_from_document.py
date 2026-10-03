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
- If an entity matches one in KNOWN ENTITIES, use exactly that name."""


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


def extract_from_document(path: str) -> list[Triplet]:
    chunks = chunk(load_text(path))
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
    triplets = extract_from_document(doc)

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