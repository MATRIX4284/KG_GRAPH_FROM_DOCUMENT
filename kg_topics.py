"""
Find the main topics in the knowledge graph built by kg_pipeline.py.

How it works:
  1. Community detection (Louvain) splits the graph into clusters of entities
     that are densely connected to each other - each cluster is a topic.
  2. Degree centrality ranks the most connected entities in each cluster.
  3. GPT-5.1 reads each cluster's facts and gives it a title and summary.

Usage (run kg_pipeline.py first so triplets.json exists):
    python kg_topics.py                 # uses triplets.json
    python kg_topics.py my_triplets.json

Outputs:
    topics.json   - topics with title, summary, key entities, member entities
    topics.html   - interactive graph coloured by topic
"""

import json
import sys

import networkx as nx
from pydantic import BaseModel, Field
from pyvis.network import Network

from kg_pipeline import DEPLOYMENT, PALETTE, Triplet, build_graph, client

MIN_TOPIC_SIZE = 3  # smaller clusters are grouped under "Other"


class TopicLabel(BaseModel):
    title: str = Field(description="Short topic name, 2-6 words")
    summary: str = Field(description="2-3 sentence summary of what these facts are about")


TOPIC_PROMPT = """You label clusters of a knowledge graph.
Given the facts in one cluster, return a short topic title and a 2-3 sentence summary.
Base both only on the facts given."""


def detect_communities(g: nx.MultiDiGraph) -> list[set[str]]:
    ug = nx.Graph(g)  # Louvain needs an undirected simple graph
    comms = nx.community.louvain_communities(ug, resolution=1.0, seed=42)
    return sorted(comms, key=len, reverse=True)


def label_topic(facts: list[str]) -> TopicLabel:
    resp = client.responses.parse(
        model=DEPLOYMENT,
        input=[
            {"role": "system", "content": TOPIC_PROMPT},
            {"role": "user", "content": "\n".join(facts[:400])},
        ],
        text_format=TopicLabel,
        reasoning={"effort": "low"},
    )
    return resp.output_parsed


def find_topics(g: nx.MultiDiGraph) -> list[dict]:
    rank = nx.degree_centrality(nx.Graph(g))
    topics, other = [], set()
    for comm in detect_communities(g):
        if len(comm) < MIN_TOPIC_SIZE:
            other |= comm
            continue
        facts = [f"{s} --{d['predicate']}--> {o}"
                 for s, o, d in g.edges(data=True) if s in comm and o in comm]
        key = sorted(comm, key=rank.get, reverse=True)[:5]
        print(f"Labelling topic {len(topics) + 1} ({len(comm)} entities) ...")
        label = label_topic(facts)
        topics.append({
            "title": label.title,
            "summary": label.summary,
            "key_entities": key,
            "entities": sorted(comm),
            "num_facts": len(facts),
        })
    if other:
        topics.append({"title": "Other", "summary": "Small clusters not large enough to form a topic.",
                       "key_entities": [], "entities": sorted(other), "num_facts": 0})
    return topics


def visualize_topics(g: nx.MultiDiGraph, topics: list[dict], out: str = "topics.html") -> None:
    topic_of = {e: i for i, t in enumerate(topics) for e in t["entities"]}
    net = Network(height="800px", width="100%", directed=True, bgcolor="#ffffff")
    for n in g.nodes:
        i = topic_of[n]
        net.add_node(n, label=n, title=f"{n}\nTopic: {topics[i]['title']}",
                     color=PALETTE[i % len(PALETTE)], size=10 + 3 * g.degree(n))
    for s, o, d in g.edges(data=True):
        net.add_edge(s, o, label=d["predicate"], title=d["evidence"])
    net.force_atlas_2based()
    net.write_html(out, open_browser=False)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "triplets.json"
    with open(path) as f:
        triplets = [Triplet(**t) for t in json.load(f)]

    g = build_graph(triplets)
    topics = find_topics(g)

    with open("topics.json", "w") as f:
        json.dump(topics, f, indent=2)
    visualize_topics(g, topics)

    print(f"\nFound {len(topics)} topics:\n")
    for i, t in enumerate(topics, 1):
        print(f"{i}. {t['title']}  ({len(t['entities'])} entities, {t['num_facts']} facts)")
        print(f"   {t['summary']}")
        if t["key_entities"]:
            print(f"   Key entities: {', '.join(t['key_entities'])}")
        print()
    print("Saved topics.json and topics.html")
