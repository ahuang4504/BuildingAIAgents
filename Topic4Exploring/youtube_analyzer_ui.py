"""
YouTube Transcript Analyzer — Gradio UI

Features:
- Analyze YouTube videos: transcript, summary, key concepts
- Interactive quiz with multiple-choice questions and inline feedback
- Chat interface to ask questions about the video content

Architecture:
- LangGraph: START → call_model → [tool_calls?] → ToolNode → call_model (loop) → END
- MemorySaver checkpointer with per-session thread_id
- Structured output via tools (model calls tools; Gradio extracts the args)
- RAG: transcript is chunked and embedded at analysis time; top chunks injected per chat query
"""

import uuid
from pathlib import Path
from typing import TypedDict, Annotated, Sequence, Literal

import numpy as np
from dotenv import load_dotenv
from langchain_core.messages import (
    BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
)
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from openai import AsyncOpenAI
from youtube_transcript_api import YouTubeTranscriptApi
import gradio as gr

load_dotenv(Path(__file__).parent.parent / ".env")

# ── RAG store ──────────────────────────────────────────────────────────────────

_rag_store: dict[str, dict] = {}  # thread_id → {chunks: list[str], embeddings: np.ndarray}


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def get_youtube_transcript(video_id: str) -> str:
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
        return (
            f"Error fetching transcript: {str(e)}. "
            "Please check that the video ID is correct and that the video has captions available."
        )


@tool
def return_video_analysis(summary: str, key_concepts: str) -> str:
    """Return structured analysis of a YouTube video after fetching its transcript.
    Call this immediately after get_youtube_transcript.

    Args:
        summary: 2-3 sentence overview of the video
        key_concepts: newline-separated bullet points of key ideas
    """
    return "Analysis complete."


@tool
def generate_quiz_question(
    question: str, choices: list[str], correct_answer: str, explanation: str
) -> str:
    """Generate a multiple-choice quiz question about the video content.

    Args:
        question: the question text
        choices: exactly 4 answer options as a list of strings
        correct_answer: one of the choices (exact string match)
        explanation: why the correct answer is right
    """
    return "Question generated."


tools = [get_youtube_transcript, return_video_analysis, generate_quiz_question]


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an educational YouTube video analyzer.

When given a YouTube URL or video ID:
1. Call get_youtube_transcript to fetch the transcript
2. Immediately call return_video_analysis with a 2-3 sentence summary and \
key concepts as newline-separated bullet points
3. After calling return_video_analysis, respond with only: "Video analyzed."

When asked to generate a quiz question:
- Call generate_quiz_question with a question, exactly 4 choices (list of strings), \
the correct_answer (exact match to one choice), and a brief explanation

For all other messages:
- NEVER call any tools — answer directly using the conversation history
- Base your answer on the summary and key concepts already in the conversation\
"""


# ── Graph ──────────────────────────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def call_model(state: State) -> dict:
    messages = list(state["messages"])
    print(f"[call_model] messages={len(messages)}, last={type(messages[-1]).__name__}")
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    # After analysis is complete, truncate the large transcript ToolMessage
    analysis_done = any(
        isinstance(m, AIMessage) and m.tool_calls
        and any(tc["name"] == "return_video_analysis" for tc in m.tool_calls)
        for m in messages
    )
    if analysis_done:
        messages = [
            ToolMessage(content=m.content[:500] + "…[truncated]", tool_call_id=m.tool_call_id)
            if isinstance(m, ToolMessage) and len(m.content) > 500 else m
            for m in messages
        ]

    model = ChatOpenAI(model="gpt-4o", temperature=0.7)
    response = model.bind_tools(tools).invoke(messages)
    return {"messages": [response]}


def route_after_model(state: State) -> Literal["tools", "end"]:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"


def route_after_tools(state: State) -> Literal["call_model", "end"]:
    """Skip the final LLM call after terminal tools — avoids re-sending huge context."""
    messages = state["messages"]
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc["name"] in ("return_video_analysis", "generate_quiz_question"):
                    return "end"
            break  # last AIMessage found; none of its tools were terminal
    return "call_model"


builder = StateGraph(State)
builder.add_node("call_model", call_model)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "call_model")
builder.add_conditional_edges(
    "call_model",
    route_after_model,
    {"tools": "tools", "end": END},
)
builder.add_conditional_edges(
    "tools",
    route_after_tools,
    {"call_model": "call_model", "end": END},
)

memory = MemorySaver()
agent = builder.compile(checkpointer=memory)


# ── Helpers ────────────────────────────────────────────────────────────────────

def find_tool_args(messages, tool_name: str):
    """Search backwards through messages for the most recent tool_call matching tool_name."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc["name"] == tool_name:
                    return tc["args"]
    return None


def find_tool_message(messages, tool_name: str) -> str | None:
    """Find content of the ToolMessage produced by calling tool_name."""
    for i, msg in enumerate(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc["name"] == tool_name:
                    for later_msg in messages[i + 1:]:
                        if (
                            isinstance(later_msg, ToolMessage)
                            and later_msg.tool_call_id == tc["id"]
                        ):
                            return later_msg.content
    return None


def last_ai_text(messages) -> str:
    """Return the content of the last AIMessage that has non-empty text."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return ""


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


async def build_rag_index(transcript: str, thread_id: str) -> None:
    chunks = chunk_text(transcript)
    client = AsyncOpenAI()
    resp = await client.embeddings.create(model="text-embedding-3-small", input=chunks)
    embeddings = np.array([e.embedding for e in resp.data], dtype=np.float32)
    _rag_store[thread_id] = {"chunks": chunks, "embeddings": embeddings}


async def retrieve_context(query: str, thread_id: str, top_k: int = 2) -> str:
    if thread_id not in _rag_store:
        return ""
    store = _rag_store[thread_id]
    client = AsyncOpenAI()
    resp = await client.embeddings.create(model="text-embedding-3-small", input=[query])
    q_emb = np.array(resp.data[0].embedding, dtype=np.float32)
    embs = store["embeddings"]
    scores = embs @ q_emb / (np.linalg.norm(embs, axis=1) * np.linalg.norm(q_emb) + 1e-9)
    top_idx = np.argsort(scores)[-top_k:][::-1]
    return "\n---\n".join(store["chunks"][i] for i in top_idx)


# ── Event handlers ─────────────────────────────────────────────────────────────

async def analyze_video(url: str, thread_id: str, chatbot: list):
    if not url.strip():
        yield "", "", "", chatbot, gr.Column(visible=True), gr.Column(visible=True)
        return

    # Yield loading state immediately so the UI stays interactive
    loading_chatbot = (chatbot or []) + [
        {"role": "user", "content": f"Analyze this video: {url}"},
        {"role": "assistant", "content": "Analyzing video... please wait."},
    ]
    yield "", "", "", loading_chatbot, gr.Column(visible=True), gr.Column(visible=True)

    print(f"[analyze_video] invoking agent for: {url[:60]}")
    try:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=f"Analyze this video: {url}")]},
            config={"configurable": {"thread_id": thread_id}},
        )
    except Exception as e:
        error = f"Error during analysis: {e}"
        print(f"[analyze_video] ERROR: {e}")
        error_chatbot = (chatbot or []) + [
            {"role": "user", "content": f"Analyze this video: {url}"},
            {"role": "assistant", "content": error},
        ]
        yield "", "", "", error_chatbot, gr.Column(visible=True), gr.Column(visible=True)
        return

    messages = result["messages"]
    print(f"[analyze_video] got {len(messages)} messages")
    analysis_args = find_tool_args(messages, "return_video_analysis")

    if analysis_args is None:
        error = "Could not analyze the video. Please check the URL and try again."
        error_chatbot = (chatbot or []) + [
            {"role": "user", "content": f"Analyze this video: {url}"},
            {"role": "assistant", "content": error},
        ]
        yield "", "", "", error_chatbot, gr.Column(visible=True), gr.Column(visible=True)
        return

    transcript = find_tool_message(messages, "get_youtube_transcript") or ""
    summary = analysis_args.get("summary", "")
    key_concepts = analysis_args.get("key_concepts", "")
    print(f"[analyze_video] analysis_args={analysis_args is not None}, transcript_len={len(transcript)}")

    chat_content = "✓ Video analyzed.\n\n"
    if summary:
        chat_content += f"**Summary:**\n{summary}\n\n"
    if key_concepts:
        chat_content += f"**Key Concepts:**\n{key_concepts}\n\n"
    chat_content += "Full transcript shown below."

    new_chatbot = (chatbot or []) + [
        {"role": "user", "content": f"Analyze this video: {url}"},
        {"role": "assistant", "content": chat_content},
    ]

    await build_rag_index(transcript, thread_id)
    yield transcript, summary, key_concepts, new_chatbot, gr.Column(visible=True), gr.Column(visible=True)


async def next_question(thread_id: str, chatbot: list):
    # Yield loading state immediately
    loading_chatbot = (chatbot or []) + [
        {"role": "user", "content": "Generate a quiz question about the video"},
        {"role": "assistant", "content": "Generating quiz question..."},
    ]
    yield "Generating question...", gr.update(choices=[], value=None), None, "", loading_chatbot

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="Generate a quiz question about the video")]},
        config={"configurable": {"thread_id": thread_id}},
    )
    messages = result["messages"]
    quiz_args = find_tool_args(messages, "generate_quiz_question")

    if quiz_args is None:
        error_chatbot = (chatbot or []) + [
            {"role": "user", "content": "Generate a quiz question about the video"},
            {"role": "assistant", "content": "Could not generate a quiz question."},
        ]
        yield "Could not generate a quiz question.", gr.update(choices=[], value=None), None, "", error_chatbot
        return

    question = quiz_args.get("question", "")
    choices = quiz_args.get("choices", [])
    correct_answer = quiz_args.get("correct_answer", "")
    explanation = quiz_args.get("explanation", "")

    quiz_state = {"correct_answer": correct_answer, "explanation": explanation}

    choices_summary = "\n".join(f"- {c}" for c in choices)
    new_chatbot = (chatbot or []) + [
        {"role": "user", "content": "Generate a quiz question about the video"},
        {"role": "assistant", "content": f"**Quiz Question:** {question}\n\n{choices_summary}"},
    ]

    yield (
        f"**{question}**",
        gr.update(choices=choices, value=None),
        quiz_state,
        "",           # clear feedback
        new_chatbot,
    )


def check_answer(selected: str, quiz_state: dict):
    if not selected or not quiz_state:
        return ""
    correct = quiz_state["correct_answer"]
    explanation = quiz_state["explanation"]
    if selected == correct:
        return f'<p style="color: green; font-weight: bold;">✓ Correct! {explanation}</p>'
    return (
        f'<p style="color: red; font-weight: bold;">✗ Incorrect. '
        f"The correct answer was: <em>{correct}</em>. {explanation}</p>"
    )


async def send_chat(message: str, chatbot: list, thread_id: str):
    if not message.strip():
        yield chatbot, message
        return

    # Show user message with placeholder immediately
    responding_chatbot = (chatbot or []) + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": "..."},
    ]
    yield responding_chatbot, ""

    context = await retrieve_context(message, thread_id)
    augmented = message
    if context:
        augmented = f"{message}\n\n[Relevant transcript excerpt:\n{context}]"

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=augmented)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    ai_text = last_ai_text(result["messages"])
    if not ai_text:
        ai_text = "I wasn't able to generate a response. Please try again."
    yield (chatbot or []) + [
        {"role": "user", "content": message},   # show original message, not augmented
        {"role": "assistant", "content": ai_text},
    ], ""


# ── Gradio UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="YouTube Transcript Analyzer") as demo:
    thread_id_state = gr.State(value=lambda: str(uuid.uuid4()))
    quiz_state = gr.State(value=None)

    gr.Markdown("# YouTube Transcript Analyzer")

    with gr.Row():
        url_box = gr.Textbox(
            placeholder="Paste a YouTube URL or video ID...",
            show_label=False,
            scale=4,
        )
        analyze_btn = gr.Button("Analyze", scale=1)

    # Chat section
    gr.Markdown("## Chat")
    chatbot = gr.Chatbot(height=400, value=[])
    with gr.Row():
        msg_box = gr.Textbox(
            placeholder="Ask a question about the video...",
            show_label=False,
            scale=4,
        )
        send_btn = gr.Button("Send", scale=1)

    # Results and Quiz sections — always visible
    with gr.Column(visible=True) as results_col:
        gr.Markdown("## Results")
        transcript_box = gr.Textbox(label="Transcript", lines=10, interactive=False)
        summary_box = gr.Textbox(label="Summary", lines=4, interactive=False)
        concepts_box = gr.Textbox(label="Key Concepts", lines=5, interactive=False)

    with gr.Column(visible=True) as quiz_col:
        gr.Markdown("## Quiz")
        question_md = gr.Markdown("")
        answer_radio = gr.Radio(choices=[], label="Select your answer")
        with gr.Row():
            check_btn = gr.Button("Check Answer")
            next_btn = gr.Button("Next Question")
        feedback_md = gr.HTML("")

    # ── Wire up events ────────────────────────────────────────────────────────

    _analyze_inputs = [url_box, thread_id_state, chatbot]
    _analyze_outputs = [transcript_box, summary_box, concepts_box, chatbot, results_col, quiz_col]

    analyze_btn.click(analyze_video, inputs=_analyze_inputs, outputs=_analyze_outputs)
    url_box.submit(analyze_video, inputs=_analyze_inputs, outputs=_analyze_outputs)

    _quiz_outputs = [question_md, answer_radio, quiz_state, feedback_md, chatbot]

    next_btn.click(next_question, inputs=[thread_id_state, chatbot], outputs=_quiz_outputs)
    check_btn.click(check_answer, inputs=[answer_radio, quiz_state], outputs=[feedback_md])

    _chat_inputs = [msg_box, chatbot, thread_id_state]
    send_btn.click(send_chat, inputs=_chat_inputs, outputs=[chatbot, msg_box])
    msg_box.submit(send_chat, inputs=_chat_inputs, outputs=[chatbot, msg_box])


if __name__ == "__main__":
    demo.queue()
    demo.launch()
