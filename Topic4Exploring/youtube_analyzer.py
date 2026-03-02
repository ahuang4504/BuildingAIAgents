"""
YouTube Transcript Analyzer Agent

This program demonstrates a LangGraph application for analyzing YouTube videos:
- Fetches transcripts using the youtube-transcript-api (no API key needed)
- Provides summaries, key concepts, and quiz questions after fetching
- Manual tool calling loop with ToolNode (same pattern as toolnode_example.py)
- Graph-based looping (no Python loops or checkpointing)
- Automatic conversation history management (trimming after 100 messages)
- Verbose debugging output
"""

import asyncio
import os
from pathlib import Path
from typing import TypedDict, Annotated, Sequence, Literal
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv(Path(__file__).parent.parent / ".env")

SCRIPT_DIR = Path(__file__).parent


class VideoAnalyzerState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    verbose: bool
    command: str  # "exit", "verbose", "quiet", or None


@tool
async def get_youtube_transcript(video_id: str) -> str:
    """
    Fetch the transcript of a YouTube video given its video ID.
    The video_id is the part after 'v=' in a YouTube URL (e.g., 'dQw4w9WgXcQ').
    For a URL like https://www.youtube.com/watch?v=dQw4w9WgXcQ, the video_id is 'dQw4w9WgXcQ'.
    For a short URL like https://youtu.be/dQw4w9WgXcQ, the video_id is 'dQw4w9WgXcQ'.

    Args:
        video_id: The YouTube video ID string

    Returns:
        The full transcript text as a single string, or an error message
    """
    try:
        transcript = YouTubeTranscriptApi().fetch(video_id)
        return " ".join([snippet.text for snippet in transcript])
    except Exception as e:
        return f"Error fetching transcript: {str(e)}. Please check that the video ID is correct and that the video has captions available."


tools = [get_youtube_transcript]


def input_node(state: VideoAnalyzerState) -> VideoAnalyzerState:
    if state.get("verbose", True):
        print("\n" + "="*80)
        print("NODE: input_node")
        print("="*80)

    user_input = input("\nYou: ").strip()

    if user_input.lower() in ["quit", "exit"]:
        if state.get("verbose", True):
            print("[DEBUG] Exit command received")
        return {"command": "exit"}

    if user_input.lower() == "verbose":
        print("[SYSTEM] Verbose mode enabled")
        return {"command": "verbose", "verbose": True}

    if user_input.lower() == "quiet":
        print("[SYSTEM] Verbose mode disabled")
        return {"command": "quiet", "verbose": False}

    if state.get("verbose", True):
        print(f"[DEBUG] User input: {user_input}")

    return {"command": None, "messages": [HumanMessage(content=user_input)]}


def call_model(state: VideoAnalyzerState) -> VideoAnalyzerState:
    if state.get("verbose", True):
        print("\n" + "="*80)
        print("NODE: call_model")
        print("="*80)
        print(f"[DEBUG] Calling model with {len(state['messages'])} messages")

    messages = list(state["messages"])

    # Prepend system prompt if this is the first call
    system_added = False
    if not messages or not isinstance(messages[0], SystemMessage):
        system_prompt = SystemMessage(
            content=(
                "You are an educational video analyzer. Your job is to help users learn "
                "from YouTube videos by analyzing their transcripts.\n\n"
                "When a user provides a YouTube URL or video ID:\n"
                "1. Extract the video ID from the URL if a full URL is given "
                "(the video ID is the value after 'v=' in standard URLs, or the path "
                "segment in youtu.be short URLs)\n"
                "2. Call the get_youtube_transcript tool with the extracted video ID\n"
                "3. After receiving the transcript, ALWAYS provide:\n"
                "   a) Summary: A concise overview of the video content\n"
                "   b) Key Concepts: The main ideas, terms, or techniques covered\n"
                "   c) Quiz Questions: Exactly 3 multiple-choice or short-answer questions "
                "to test understanding of the material\n\n"
                "If the transcript fetch fails, explain the error clearly and suggest "
                "alternatives (e.g., check if the video has captions enabled)."
            )
        )
        messages = [system_prompt] + messages
        system_added = True
        if state.get("verbose", True):
            print("[DEBUG] Added system prompt")

    model = ChatOpenAI(model="gpt-4o", temperature=0.7)
    model_with_tools = model.bind_tools(tools)
    response = model_with_tools.invoke(messages)

    if state.get("verbose", True):
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"[DEBUG] Model requested {len(response.tool_calls)} tool call(s):")
            for tc in response.tool_calls:
                print(f"  - {tc['name']}({tc['args']})")
        else:
            print(f"[DEBUG] Model response (no tools): {response.content[:100]}...")

    if system_added:
        return {"messages": [messages[0], response]}
    else:
        return {"messages": [response]}


def output_node(state: VideoAnalyzerState) -> VideoAnalyzerState:
    if state.get("verbose", True):
        print("\n" + "="*80)
        print("NODE: output_node")
        print("="*80)

    # Walk backwards to find the last non-empty AI message (tool messages may be interspersed)
    last_ai_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            last_ai_message = msg
            break

    if last_ai_message:
        print(f"\nAssistant: {last_ai_message.content}")
    else:
        print("\n[WARNING] No assistant response found")

    return {}


def trim_history(state: VideoAnalyzerState) -> VideoAnalyzerState:
    messages = state["messages"]
    max_messages = 100

    if len(messages) > max_messages:
        if state.get("verbose", True):
            print(f"\n[DEBUG] History length: {len(messages)} messages")
            print(f"[DEBUG] Trimming to most recent {max_messages} messages")

        # Always keep the system message at index 0
        if messages and isinstance(messages[0], SystemMessage):
            trimmed = [messages[0]] + list(messages[-(max_messages - 1):])
        else:
            trimmed = list(messages[-max_messages:])

        return {"messages": trimmed}

    return {}


def route_after_input(state: VideoAnalyzerState) -> Literal["call_model", "end", "input"]:
    command = state.get("command")

    if command == "exit":
        if state.get("verbose", True):
            print("[DEBUG] Routing to END (exit requested)")
        return "end"

    # Verbose/quiet just toggle a flag — loop back without calling the model
    if command in ["verbose", "quiet"]:
        if state.get("verbose", True):
            print("[DEBUG] Routing back to input (verbose toggle)")
        return "input"

    if state.get("verbose", True):
        print("[DEBUG] Routing to call_model")
    return "call_model"


def route_after_model(state: VideoAnalyzerState) -> Literal["tools", "output"]:
    last_message = state["messages"][-1]

    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        if state.get("verbose", True):
            print("[DEBUG] Routing to tools")
        return "tools"

    if state.get("verbose", True):
        print("[DEBUG] Routing to output")
    return "output"


def create_analyzer_graph():
    """
    Graph structure:

        ┌──────────────────────────────────────────────────────┐
        │                                                      │
        ▼                                                      │
      input_node ──(check command)──> call_model              │
          ▲                              │                     │
          │                              ├──(has tools)──> tools
          │                              │                    │
          │                              └──(no tools)──> output_node
          │                                                    │
          │                                                    ▼
          └───(verbose/quiet)                           trim_history ──┘

          └─────(exit)──> END
    """
    workflow = StateGraph(VideoAnalyzerState)

    tool_node = ToolNode(tools)

    workflow.add_node("input", input_node)
    workflow.add_node("call_model", call_model)
    workflow.add_node("tools", tool_node)
    workflow.add_node("output", output_node)
    workflow.add_node("trim_history", trim_history)

    workflow.set_entry_point("input")

    workflow.add_conditional_edges(
        "input",
        route_after_input,
        {"call_model": "call_model", "input": "input", "end": END}
    )
    workflow.add_conditional_edges(
        "call_model",
        route_after_model,
        {"tools": "tools", "output": "output"}
    )

    workflow.add_edge("tools", "call_model")
    workflow.add_edge("output", "trim_history")
    workflow.add_edge("trim_history", "input")

    print("[SYSTEM] Video analyzer graph created successfully (using manual ToolNode)")
    return workflow.compile()


def visualize_graph(app):
    try:
        graph_png = app.get_graph().draw_mermaid_png()
        out_path = SCRIPT_DIR / "youtube_analyzer_graph.png"
        with open(out_path, "wb") as f:
            f.write(graph_png)
        print(f"[SYSTEM] Graph visualization saved to '{out_path}'")
    except Exception as e:
        print(f"[WARNING] Could not generate graph visualization: {e}")
        print("You may need to install: pip install pygraphviz or pip install grandalf")


async def main():
    print("="*80)
    print("YouTube Transcript Analyzer - Educational Video Analysis Agent")
    print("="*80)
    print("\nThis agent analyzes YouTube videos using their transcripts:")
    print("  - Paste a YouTube URL or video ID to get started")
    print("  - Agent extracts the video ID, fetches the transcript, then provides:")
    print("      * Summary of the video content")
    print("      * Key concepts covered")
    print("      * 3 quiz questions to test your understanding")
    print("  - Uses manual ToolNode for transcript fetching")
    print("  - Single persistent conversation across all turns")
    print("  - History managed automatically (trimmed after 100 messages)")
    print("\nCommands:")
    print("  - Type 'quit' or 'exit' to end the conversation")
    print("  - Type 'verbose' to enable detailed tracing")
    print("  - Type 'quiet' to disable detailed tracing")
    print("\nAvailable tools:")
    print("  - get_youtube_transcript(video_id): Fetch video transcript text")
    print("="*80)

    app = create_analyzer_graph()
    visualize_graph(app)

    initial_state = {"messages": [], "verbose": True, "command": None}

    print("\n[SYSTEM] Starting conversation...\n")

    try:
        # Invoke once — the graph loops internally via trim_history -> input
        await app.ainvoke(initial_state)
    except KeyboardInterrupt:
        print("\n\n[SYSTEM] Interrupted by user (Ctrl+C)")

    print("\n[SYSTEM] Conversation ended. Goodbye!\n")


if __name__ == "__main__":
    asyncio.run(main())
