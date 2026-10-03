"""
GraphRAG: ask questions about a document using the knowledge graph.

How it works:
  1. Plan    - GPT-5.1 picks the graph entities the question is about and decides
               whether it is a specific question (local) or a broad one (global).
  2. Retrieve
       local  - walk the graph 2 hops out from those entities and collect the facts
                (with their evidence quotes) along the way.
       global - use the topic titles/summaries from kg_topics.py plus the most
                connected facts.
  3. Answer  - GPT-5.1 answers using only the retrieved facts and cites them [n].

Usage (run kg_pipeline.py first; kg_topics.py is optional, used for global questions):
    python kg_rag.py                                 # interactive
    python kg_rag.py "Who founded Helix Robotics?"   # single question
"""

import json
import os
import re
import sys
from typing import Literal

import networkx as nx
from pydantic import BaseModel, Field

from kg_pipeline import DEPLOYMENT, Triplet, build_graph, client

MAX_HOPS = 2
MAX_FACTS = 150


class QueryPlan(BaseModel):
    entities: list[str] = Field(description="Entity names from the list that the question is about")
    scope: Literal["local", "global"] = Field(
        description="local = about specific entities; global = broad/summary question about the whole document")


PLAN_PROMPT = """You route questions over a knowledge graph.
From ENTITIES, pick the ones the QUESTION is about (use names exactly as listed;
include obvious aliases or partial matches). Choose scope "global" for broad questions
(main themes, summary, overview) and "local" for questions about specific things."""

ANSWER_PROMPT = """Answer the question using ONLY the numbered facts provided.
Cite the facts you used like [3] or [2][5]. Be concise.
If the facts do not contain the answer, say you could not find it in the knowledge graph."""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_graph(path: str = "triplets.json") -> nx.MultiDiGraph:
    with open(path) as f:
        return build_graph([Triplet(**t) for t in json.load(f)])


def load_topics(path: str = "topics.json") -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. Plan: link the question to graph entities
# ---------------------------------------------------------------------------
def candidate_entities(g: nx.MultiDiGraph, question: str, limit: int = 400) -> list[str]:
    nodes = list(g.nodes)
    if len(nodes) <= limit:
        return nodes
    # Large graph: pre-filter by word overlap, then top up with hub entities.
    words = {w for w in re.findall(r"\w+", question.casefold()) if len(w) > 2}
    hits = [n for n in nodes if words & set(re.findall(r"\w+", n.casefold()))]
    hubs = sorted(nodes, key=g.degree, reverse=True)
    return list(dict.fromkeys(hits + hubs))[:limit]


def plan(g: nx.MultiDiGraph, question: str) -> QueryPlan:
    resp = client.responses.parse(
        model=DEPLOYMENT,
        input=[
            {"role": "system", "content": PLAN_PROMPT},
            {"role": "user", "content": "ENTITIES:\n" + "\n".join(candidate_entities(g, question))
                                        + f"\n\nQUESTION: {question}"},
        ],
        text_format=QueryPlan,
        reasoning={"effort": "low"},
    )
    p = resp.output_parsed
    p.entities = [e for e in p.entities if e in g]  # drop anything not in the graph
    return p


# ---------------------------------------------------------------------------
# 2. Retrieve
# ---------------------------------------------------------------------------
def fmt(s: str, o: str, d: dict) -> str:
    return f'{s} --{d["predicate"]}--> {o}  (evidence: "{d["evidence"]}")'


def local_context(g: nx.MultiDiGraph, seeds: list[str]) -> list[str]:
    """Facts within MAX_HOPS of the seed entities, nearest first."""
    ug = g.to_undirected(as_view=True)
    dist = {}
    for s in seeds:
        for n, d in nx.single_source_shortest_path_length(ug, s, cutoff=MAX_HOPS).items():
            dist[n] = min(d, dist.get(n, d))
    edges = [(s, o, d) for s, o, d in g.edges(data=True) if s in dist and o in dist]
    edges.sort(key=lambda e: min(dist[e[0]], dist[e[1]]))
    return [fmt(*e) for e in edges[:MAX_FACTS]]


def global_context(g: nx.MultiDiGraph, topics: list[dict]) -> list[str]:
    facts = [f'Topic "{t["title"]}": {t["summary"]} Key entities: {", ".join(t["key_entities"])}'
             for t in topics if t["title"] != "Other"]
    hubs = sorted(g.nodes, key=g.degree, reverse=True)[:10]
    hub_edges = [(s, o, d) for s, o, d in g.edges(data=True) if s in hubs or o in hubs]
    return facts + [fmt(*e) for e in hub_edges[:MAX_FACTS - len(facts)]]


# ---------------------------------------------------------------------------
# 3. Answer
# ---------------------------------------------------------------------------
def answer(question: str, facts: list[str]) -> str:
    numbered = "\n".join(f"[{i}] {f}" for i, f in enumerate(facts, 1))
    resp = client.responses.create(
        model=DEPLOYMENT,
        input=[
            {"role": "system", "content": ANSWER_PROMPT},
            {"role": "user", "content": f"FACTS:\n{numbered}\n\nQUESTION: {question}"},
        ],
        reasoning={"effort": "low"},
    )
    return resp.output_text


def ask(g: nx.MultiDiGraph, topics: list[dict], question: str, show_facts: bool = False) -> str:
    p = plan(g, question)
    if p.scope == "global" or not p.entities:
        facts = global_context(g, topics)
        print(f"  (global search, {len(facts)} facts)")
    else:
        facts = local_context(g, p.entities)
        print(f"  (local search from {', '.join(p.entities)}; {len(facts)} facts)")
    if show_facts:
        print("\n".join(f"  [{i}] {f}" for i, f in enumerate(facts, 1)))
    return answer(question, facts)


if __name__ == "__main__":
    g, topics = load_graph(), load_topics()
    print(f"Loaded graph: {g.number_of_nodes()} entities, {g.number_of_edges()} facts, "
          f"{len(topics)} topics")

    if len(sys.argv) > 1:
        print(ask(g, topics, " ".join(sys.argv[1:])))
        sys.exit()

    print('Ask a question (type "facts" before a question to see retrieved facts, "quit" to exit)')
    while True:
        q = input("\n> ").strip()
        if q.lower() in {"quit", "exit", ""}:
            break
        show = q.lower().startswith("facts ")
        print(ask(g, topics, q[6:] if show else q, show_facts=show))
