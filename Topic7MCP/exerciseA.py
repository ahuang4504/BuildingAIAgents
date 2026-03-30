# Exercise 1: Inspect Asta MCP tools/list endpoint
#
# Q: Which tool would you use to find all papers about "transformer attention mechanisms"?
# A: search_papers_by_relevance -> according to the fetched description, this tools helps 
# search for paper by keyword relevance. To find papers about 'transformer attention mechanisms',
# we can just input that phrase as the keyword query to this tool, and it will return papers that 
# are relevant to that topic.
#
# Q: Which tool would you use to find who else published in the same area as a specific author?
# A: One method is to use get_author_papers for retrieving the papers of a known author to understand their area and either collecting co-authors
#    or then further using search_papers_by_relevance with those keywords and looking for overlapped authors that are heavily 
#    involved in that research area
import json
import os

import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


def main():
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "x-api-key": os.environ["ASTA_API_KEY"],
    }
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {},
    }
    resp = requests.post(
        "https://asta-tools.allen.ai/mcp/v1",
        headers=headers,
        json=payload,
    )
    resp.raise_for_status()

    # Response is SSE: extract JSON from "data: {...}" line
    data_line = next(
        line[len("data: "):] for line in resp.text.splitlines() if line.startswith("data: ")
    )
    tools = json.loads(data_line)["result"]["tools"]

    for tool in tools:
        name = tool.get("name", "unknown")
        description = next((l for l in tool.get("description", "").splitlines() if l.strip()), "")

        schema = tool.get("inputSchema", {})
        properties = schema.get("properties", {})
        required_params = set(schema.get("required", []))

        required_parts = []
        optional_parts = []

        for param_name, param_schema in properties.items():
            param_type = param_schema.get("type", "any")
            entry = f"{param_name} ({param_type})"
            if param_name in required_params:
                required_parts.append(entry)
            else:
                optional_parts.append(entry)

        print(f"Tool: {name}")
        print(f"  Description: {description}")
        if required_parts:
            print(f"  Required: {', '.join(required_parts)}")
        else:
            print("  Required: (none)")
        if optional_parts:
            print(f"  Optional: {', '.join(optional_parts)}")
        print()


if __name__ == "__main__":
    main()
