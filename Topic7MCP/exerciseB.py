import json
import os

import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

URL = "https://asta-tools.allen.ai/mcp/v1"
headers = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "x-api-key": os.environ["ASTA_API_KEY"],
}


def call_tool(name, arguments, call_id=1):
    payload = {
        "jsonrpc": "2.0",
        "id": call_id,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }
    resp = requests.post(URL, headers=headers, json=payload)
    # Response is SSE — extract the data: line
    data_line = next(l[6:] for l in resp.text.splitlines() if l.startswith("data: "))
    result = json.loads(data_line)["result"]
    if result.get("isError"):
        raise RuntimeError(f"Tool error: {result['content'][0]['text']}")
    # Each result item is a separate content entry
    return [json.loads(item["text"]) for item in result["content"]]


# ---------------------------------------------------------------------------
# Drill 1 — search_papers_by_relevance: Find recent LLM agent papers
# ---------------------------------------------------------------------------
def drill1():
    print("=" * 60)
    print("Drill 1 — search_papers_by_relevance: LLM agent papers")
    print("=" * 60)
    papers = call_tool(
        "search_papers_by_relevance",
        {"keyword": "large language model agents", "fields": "title,year,authors,abstract", "limit": 5},
        call_id=1,
    )
    for i, paper in enumerate(papers, 1):
        print(f"{i}. {paper.get('title')} ({paper.get('year')})")
    print()


# ---------------------------------------------------------------------------
# Drill 2 — get_citations: Trace impact of the BERT paper
# ---------------------------------------------------------------------------
def drill2():
    print("=" * 60)
    print("Drill 2 — get_citations: Papers citing BERT (2023+)")
    print("=" * 60)
    items = call_tool(
        "get_citations",
        {
            "paper_id": "ARXIV:1810.04805",
            "fields": "title,year,authors",
            "limit": 10,
            "publication_date_range": "2023-01-01:",
        },
        call_id=2,
    )
    print(f"Results returned: {len(items)}")
    for i, item in enumerate(items[:5], 1):
        paper = item.get("citingPaper", item)
        print(f"{i}. {paper.get('title')} ({paper.get('year')})")
    print()


# ---------------------------------------------------------------------------
# Drill 3 — get_paper (references field): ReAct paper's intellectual lineage
# ---------------------------------------------------------------------------
def drill3():
    print("=" * 60)
    print("Drill 3 — get_paper references: ReAct paper lineage by year")
    print("=" * 60)
    items = call_tool(
        "get_paper",
        {"paper_id": "ARXIV:2210.03629", "fields": "references.title,references.year"},
        call_id=3,
    )
    refs = items[0].get("references", [])
    refs_sorted = sorted((r for r in refs if r.get("year")), key=lambda r: r["year"])
    for ref in refs_sorted:
        print(f"  [{ref.get('year', 'N/A')}] {ref.get('title')}")
    print()


if __name__ == "__main__":
    drill1()
    drill2()
    drill3()
