"""
Persistent Conversation Agent with LangGraph Checkpointing

Demonstrates:
- LangGraph nodes/edges replacing the Python tool-calling loop
- SqliteSaver checkpointing for multi-turn context that survives restarts
- Interactive thread picker at startup when no CLI arg is given
- Resume from thread_id, graceful Ctrl+C, 'history' inspection mid-session
- Mermaid diagram generation for the graph structure

Compare with:
  langgraph-tool-handling.py  — graph-based loop but NO checkpointing (state resets per run)
  manual-tool-handling.py     — Python for-loop, no graph, no persistence
"""

import json
import os
import sys
import uuid
import importlib.util
import pathlib
from datetime import datetime
from typing import Annotated, Sequence, Literal
from typing_extensions import TypedDict

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver


# ============================================================================
# IMPORT TOOLS from langgraph-tool-handling.py (no duplication)
# The if __name__ == "__main__" guard in that file prevents test queries
# from running when we exec_module it here.
# ============================================================================

_spec = importlib.util.spec_from_file_location(
    "_tools", pathlib.Path(__file__).parent / "langgraph-tool-handling.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

get_weather, calculator, count_letter, unit_converter = (
    _mod.get_weather,
    _mod.calculator,
    _mod.count_letter,
    _mod.unit_converter,
)
TOOLS = [get_weather, calculator, count_letter, unit_converter]


# ============================================================================
# STATE
# add_messages reducer merges new messages into accumulated history —
# this is what makes multi-turn context work with checkpointing.
# ============================================================================

class ConversationState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ============================================================================
# LLM
# ============================================================================

llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
llm_with_tools = llm.bind_tools(TOOLS)


# ============================================================================
# NODES (2 total — minimal)
# ============================================================================

def call_model(state: ConversationState) -> dict:
    """Prepend SystemMessage on first turn, then invoke LLM with tools bound."""
    messages = list(state["messages"])
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [
            SystemMessage(
                content=(
                    "You are a helpful assistant. "
                    "Use the provided tools when they can help answer the question."
                )
            )
        ] + messages
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# ToolNode(TOOLS) is the second node — LangGraph prebuilt, automatically
# dispatches tool calls and returns ToolMessage results; no manual dispatch.


# ============================================================================
# ROUTING
# ============================================================================

def route_after_model(state: ConversationState) -> Literal["tools", "__end__"]:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END


# ============================================================================
# GRAPH
#
#   START → call_model
#   call_model → tools        (if tool_calls present)
#   call_model → END          (if no tool_calls)
#   tools      → call_model   (model sees tool results, decides next)
# ============================================================================

workflow = StateGraph(ConversationState)
workflow.add_node("call_model", call_model)
workflow.add_node("tools", ToolNode(TOOLS))
workflow.add_edge(START, "call_model")
workflow.add_conditional_edges(
    "call_model",
    route_after_model,
    {"tools": "tools", END: END},
)
workflow.add_edge("tools", "call_model")

DB_PATH = pathlib.Path(__file__).parent / "conversations.db"
checkpointer = SqliteSaver(sqlite3.connect(str(DB_PATH), check_same_thread=False))
graph = workflow.compile(checkpointer=checkpointer)


# ============================================================================
# THREAD REGISTRY
# threads.json stores metadata (created_at, first_message) for the picker.
# ============================================================================

THREADS_FILE = pathlib.Path(__file__).parent / "threads.json"


def load_threads() -> dict:
    if THREADS_FILE.exists():
        return json.load(THREADS_FILE.open())
    return {}


def register_thread(thread_id: str, first_message: str) -> None:
    threads = load_threads()
    if thread_id not in threads:
        threads[thread_id] = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "first_message": first_message[:60],
        }
        json.dump(threads, THREADS_FILE.open("w"), indent=2)


# ============================================================================
# HELPERS
# ============================================================================

def show_checkpoint(thread_id: str) -> None:
    """Debug dump: print every message stored in the checkpoint for this thread."""
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)
    msgs = snapshot.values.get("messages", [])
    print(f"\n── Checkpoint '{thread_id}' ({len(msgs)} messages) ──")
    for msg in msgs:
        role = type(msg).__name__.replace("Message", "")
        text = (msg.content or "[tool calls pending]")[:100]
        print(f"  [{role:10s}] {text}")
    print()


def show_conversation(thread_id: str) -> None:
    """Readable replay of a thread's Human/Assistant exchanges on recovery."""
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)
    msgs = snapshot.values.get("messages", [])
    if not msgs:
        return
    print(f"\n── Resuming thread '{thread_id}' — conversation history ──\n")
    for msg in msgs:
        if isinstance(msg, HumanMessage):
            print(f"  You:       {msg.content}")
        elif isinstance(msg, AIMessage) and msg.content:
            print(f"  Assistant: {msg.content}")
        # Skip ToolMessages and bare AIMessages with no text
    print("─" * 54)


def pick_thread() -> str | None:
    """Interactive thread picker. Returns a thread_id to resume, or None for new."""
    threads = load_threads()
    if not threads:
        return None

    items = list(threads.items())
    print("\n── Previous conversations ──")
    for i, (tid, meta) in enumerate(items, 1):
        created = meta.get("created_at", "")
        first = meta.get("first_message", "")
        print(f"  [{i}] {tid}  {created}  {first}")
    print()

    try:
        choice = input(
            "Enter number to resume, thread ID directly, or press Enter for new: "
        ).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None

    if not choice:
        return None

    # Numeric index
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(items):
            return items[idx][0]
        print(f"[Invalid number, starting new conversation]")
        return None

    # Raw thread_id
    if choice in threads:
        return choice

    print(f"[Unknown thread_id '{choice}', starting new conversation]")
    return None


def save_mermaid() -> None:
    """Generate and save the graph's Mermaid diagram to a .mmd file."""
    out = pathlib.Path(__file__).parent / "conversation_agent_graph.mmd"
    mermaid_str = graph.get_graph().draw_mermaid()
    out.write_text(mermaid_str)
    print(f"[Graph diagram saved to {out.name}]")


# ============================================================================
# CONVERSATION LOOP
# One graph.invoke() per user turn.
# The checkpointer automatically loads + appends to accumulated history.
# ============================================================================

_first_message_registered: set[str] = set()


def run_conversation(thread_id: str | None = None) -> str:
    if thread_id is None:
        thread_id = uuid.uuid4().hex[:8]
    config = {"configurable": {"thread_id": thread_id}}
    print(f"\n── Thread: {thread_id} ──")
    print("Commands: 'exit' to quit, 'history' to show checkpoint.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n[Interrupted. Resume: python conversation_agent.py {thread_id}]")
            return thread_id

        if user_input.lower() in ("exit", "quit"):
            print(f"[Ended. thread_id={thread_id}]")
            return thread_id
        if user_input.lower() == "history":
            show_checkpoint(thread_id)
            continue
        if not user_input:
            continue

        # Register thread on first real message
        if thread_id not in _first_message_registered:
            register_thread(thread_id, user_input)
            _first_message_registered.add(thread_id)

        result = graph.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )
        reply = next(
            (m for m in reversed(result["messages"])
             if isinstance(m, AIMessage) and m.content),
            None,
        )
        if reply:
            print(f"\nAssistant: {reply.content}\n")


# ============================================================================
# ENTRY POINT
#
#   python conversation_agent.py           → interactive picker or new conversation
#   python conversation_agent.py <thread>  → resume existing thread directly
# ============================================================================

if __name__ == "__main__":
    save_mermaid()
    if len(sys.argv) > 1:
        thread_id = sys.argv[1]
        show_conversation(thread_id)
    else:
        thread_id = pick_thread()
        if thread_id:
            show_conversation(thread_id)
    run_conversation(thread_id)
