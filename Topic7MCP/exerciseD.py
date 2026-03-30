"""
Exercise D: Citation Network Explorer Agent

Autonomous agent — Python orchestrates all MCP calls, LLM only generates the final report.

Usage:
    python exerciseD.py ARXIV:2210.03629
    python exerciseD.py ARXIV:2210.03629 > report.md
"""
import json
import os
import sys

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

URL = "https://asta-tools.allen.ai/mcp/v1"
asta_headers = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "x-api-key": os.environ["ASTA_API_KEY"],
}

client = OpenAI()

_call_id = 0


def _parse_sse(resp) -> dict:
    data_line = next(l[6:] for l in resp.text.splitlines() if l.startswith("data: "))
    return json.loads(data_line)


def call_tool(name: str, arguments: dict) -> list[dict]:
    global _call_id
    _call_id += 1
    payload = {
        "jsonrpc": "2.0",
        "id": _call_id,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }
    resp = requests.post(URL, headers=asta_headers, json=payload)
    resp.raise_for_status()
    result = _parse_sse(resp)["result"]
    if result.get("isError"):
        raise RuntimeError(f"Tool error: {result['content'][0]['text']}")
    return [json.loads(item["text"]) for item in result["content"]]


# ---------------------------------------------------------------------------
# Step 1: Seed paper — metadata + references in one call
# ---------------------------------------------------------------------------
def fetch_seed_paper(paper_id: str) -> dict:
    print(f"[1/4] Fetching seed paper: {paper_id}", file=sys.stderr)
    items = call_tool("get_paper", {
        "paper_id": paper_id,
        "fields": (
            "title,abstract,year,authors,fieldsOfStudy,"
            "references.paperId,references.title,references.year,references.citationCount"
        ),
    })
    return items[0]


# ---------------------------------------------------------------------------
# Step 2: Top 5 references by citation count — fetch abstracts for each
# ---------------------------------------------------------------------------
def fetch_top_references(references: list[dict]) -> list[dict]:
    print("[2/4] Fetching top reference abstracts", file=sys.stderr)
    # Filter to refs that have a paperId and sort by citationCount descending
    refs_with_id = [r for r in references if r.get("paperId")]
    top5 = sorted(refs_with_id, key=lambda r: r.get("citationCount") or 0, reverse=True)[:5]
    result = []
    for ref in top5:
        print(f"      → {ref.get('title', ref['paperId'])[:60]}", file=sys.stderr)
        items = call_tool("get_paper", {
            "paper_id": ref["paperId"],
            "fields": "title,abstract,year,authors",
        })
        paper = items[0]
        paper["citationCount"] = ref.get("citationCount")
        result.append(paper)
    return result


# ---------------------------------------------------------------------------
# Step 3: Recent citing papers (last 3 years)
# ---------------------------------------------------------------------------
def fetch_recent_citations(paper_id: str) -> list[dict]:
    print("[3/4] Fetching recent citing papers", file=sys.stderr)
    items = call_tool("get_citations", {
        "paper_id": paper_id,
        "fields": "title,year,authors,abstract",
        "limit": 10,
        "publication_date_range": "2023-01-01:",
    })
    # Unwrap citingPaper wrapper (same shape as exerciseB drill2)
    papers = [item.get("citingPaper", item) for item in items]
    return papers[:5]


# ---------------------------------------------------------------------------
# Step 4: Author profiles — most-cited work other than the seed paper
# ---------------------------------------------------------------------------
def fetch_author_profiles(authors: list[dict], seed_title: str) -> list[dict]:
    print("[4/4] Fetching author profiles", file=sys.stderr)
    profiles = []
    for author in authors:
        author_id = author.get("authorId")
        if not author_id:
            continue
        print(f"      → {author.get('name', author_id)}", file=sys.stderr)
        # get_author_papers only returns {paperId, title} — fields param is ignored
        items = call_tool("get_author_papers", {
            "author_id": author_id,
            "limit": 3,
        })
        # Enrich each paper with citationCount via get_paper (fields param works there)
        enriched = []
        for p in items:
            if p.get("paperId") and p.get("title") != seed_title:
                detail = call_tool("get_paper", {
                    "paper_id": p["paperId"],
                    "fields": "title,year,citationCount",
                })
                enriched.append(detail[0])
        enriched_sorted = sorted(enriched, key=lambda x: x.get("citationCount") or 0, reverse=True)
        profiles.append({
            "name": author.get("name"),
            "authorId": author_id,
            "top_paper": enriched_sorted[0] if enriched_sorted else None,
        })
    return profiles


# ---------------------------------------------------------------------------
# Report generation — single LLM call over all collected data
# ---------------------------------------------------------------------------
def generate_report(
    seed: dict,
    top_refs: list[dict],
    recent_cites: list[dict],
    author_profiles: list[dict],
) -> str:
    print("[LLM] Generating markdown report", file=sys.stderr)

    data_block = json.dumps({
        "seed_paper": seed,
        "top_5_references": top_refs,
        "recent_citing_papers": recent_cites,
        "author_profiles": author_profiles,
    }, indent=2)

    system_prompt = (
        "You are a research report writer. "
        "Given structured data about a paper and its citation network, "
        "write a well-formatted markdown report with exactly these four sections:\n\n"
        "1. **Summary** — a concise one-paragraph overview of the seed paper: "
        "what it proposes, why it matters, and its field of study.\n"
        "2. **Foundational Works** — the 5 most-cited references, "
        "each with title, year, authors, and a 1-2 sentence description of what it contributes.\n"
        "3. **Recent Developments** — up to 5 papers from the last 3 years that cite this work, "
        "each with title, year, authors, and a brief note on how it builds on the seed paper.\n"
        "4. **Author Profiles** — for each author of the seed paper, "
        "their name and their most notable other work (title, year, citation count).\n\n"
        "Use only the data provided. Do not hallucinate papers or authors."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Here is the citation network data:\n\n```json\n{data_block}\n```"},
        ],
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python exerciseD.py <paper_id>", file=sys.stderr)
        print("Example: python exerciseD.py ARXIV:2210.03629", file=sys.stderr)
        print("Export:  python exerciseD.py ARXIV:2210.03629 > report.md", file=sys.stderr)
        sys.exit(1)

    paper_id = sys.argv[1]
    print(f"Citation Network Explorer — seed: {paper_id}\n", file=sys.stderr)

    seed = fetch_seed_paper(paper_id)
    references = seed.get("references", [])
    authors = seed.get("authors", [])

    top_refs = fetch_top_references(references)
    recent_cites = fetch_recent_citations(paper_id)
    author_profiles = fetch_author_profiles(authors, seed.get("title", ""))

    print("", file=sys.stderr)
    report = generate_report(seed, top_refs, recent_cites, author_profiles)
    print(report)


if __name__ == "__main__":
    main()
