import json
import os

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

SYSTEM_PROMPT = (
    "You are a research assistant with access to Semantic Scholar tools via MCP. "
    "Use the available tools to find papers, authors, citations, and references. "
    "Always use tools to look up real data rather than relying on your training knowledge."
)


def _parse_sse(resp) -> dict:
    """Extract JSON from the SSE 'data: ...' line."""
    data_line = next(l[6:] for l in resp.text.splitlines() if l.startswith("data: "))
    return json.loads(data_line)


def get_asta_tools() -> list[dict]:
    """Fetch tool schemas from MCP and convert to OpenAI function-calling format."""
    payload = {
        "jsonrpc": "2.0",
        "id": 0,
        "method": "tools/list",
        "params": {},
    }
    resp = requests.post(URL, headers=asta_headers, json=payload)
    resp.raise_for_status()
    mcp_tools = _parse_sse(resp)["result"]["tools"]

    # MCP: { name, description, inputSchema }
    # OpenAI: { type: "function", function: { name, description, parameters } }
    # inputSchema is already valid JSON Schema — no transformation needed
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("inputSchema", {}),
            },
        }
        for tool in mcp_tools
    ]


def call_asta_tool(name: str, arguments: dict) -> str:
    """Execute a tools/call and return the result as a JSON string."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }
    try:
        resp = requests.post(URL, headers=asta_headers, json=payload)
        resp.raise_for_status()
        result = _parse_sse(resp)["result"]
        if result.get("isError"):
            return f"Error: {result['content'][0]['text']}"
        # Each content item's text is a JSON-encoded string (double-encoded)
        parsed = [json.loads(item["text"]) for item in result["content"]]
        return json.dumps(parsed, indent=2)
    except Exception as e:
        return f"Error calling tool {name}: {e}"


def chat(user_message: str, messages: list[dict], tools: list[dict]) -> str:
    """One turn of the chatbot loop, handling tool calls until a final answer."""
    messages.append({"role": "user", "content": user_message})

    for _ in range(10):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        assistant_message = response.choices[0].message
        messages.append(assistant_message)

        if not assistant_message.tool_calls:
            return assistant_message.content

        for tool_call in assistant_message.tool_calls:
            name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"  [tool] {name}({json.dumps(arguments)})")
            result = call_asta_tool(name, arguments)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": name,
                "content": result,
            })

    return "Max iterations reached without a final answer."


def main():
    print("Fetching Asta tools from MCP server...")
    tools = get_asta_tools()
    print(f"Discovered {len(tools)} tools: {[t['function']['name'] for t in tools]}\n")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("Research assistant ready. Type 'quit' to exit.\n")
    print("Suggested queries:")
    print("  - Find recent papers about large language model agents")
    print("  - Who wrote Attention is All You Need and what else have they published?")
    print("  - What papers cite the original BERT paper?")
    print("  - Summarize the references used in the ReAct paper\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not user_input or user_input.lower() == "quit":
            break
        answer = chat(user_input, messages, tools)
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()
