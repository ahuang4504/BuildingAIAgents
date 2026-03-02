# Topic 4: Exploring LangGraph & Tools

Note: I had Claude Code help format this README.md since it's good at helping with documentation.

## Table of Contents

- [toolnode_example.py](#toolnode_examplepy)
- [react_agent_example.py](#react_agent_examplepy)
- [youtube_analyzer.py](#youtube_analyzerpy)
- [youtube_analyzer_ui.py](#youtube_analyzer_uipy)
  - [Requirements](#requirements)
  - [Usage](#usage)
  - [UI Features](#ui-features)
  - [Architecture](#architecture)
  - [Notes](#notes)

---

## toolnode_example.py

Demonstrates manual ToolNode dispatch in LangGraph. Tools are defined as async functions so LangGraph can run independent tool calls in parallel via asyncio.

Graph: `START → call_model → [tool calls?] → ToolNode → call_model (loop) → END`

Includes `verbose` and `exit` special commands handled through state-based routing.

---

## react_agent_example.py

Same agent built using `create_react_agent` from LangGraph prebuilt. The graph is simpler on the surface but the ReAct agent handles tool routing internally.

Useful for comparison with `toolnode_example.py` — identical behavior, much less boilerplate.

---

## youtube_analyzer.py

CLI YouTube transcript analyzer. Paste a YouTube URL or video ID at the prompt; the agent fetches the transcript and returns a summary, key concepts, and quiz questions.

**Special commands:** `verbose`, `quiet`, `exit`, `quit`

---

## youtube_analyzer_ui.py

Gradio web UI version of the YouTube analyzer. The main project file.

### Requirements

- Python 3.10+
- Virtual environment at `../.venv`
- `OPENAI_API_KEY` in `../.env`

**Dependencies:**
```
langchain-openai  langgraph  langchain-core
youtube-transcript-api  gradio  numpy  openai
```

### Usage

```bash
source ../.venv/bin/activate
python youtube_analyzer_ui.py
```

Open the local URL printed in the terminal (e.g. `http://127.0.0.1:7860`).

### UI Features

- **Analyze** — paste a YouTube URL or video ID to fetch transcript, summary, and key concepts
- **Chat** — ask follow-up questions; relevant transcript chunks are retrieved via RAG and injected into each query
- **Quiz** — generate multiple-choice questions with instant feedback

### Architecture

**LangGraph workflow:**
```
START → call_model → [tool calls?] → ToolNode → call_model (loop) → END
```

Three tools: `get_youtube_transcript`, `return_video_analysis`, `generate_quiz_question`. `MemorySaver` provides per-session conversation history via `thread_id`.

**RAG:** After analysis, the transcript is chunked (500 chars, 100 overlap) and embedded with `text-embedding-3-small`. Each chat query retrieves the top 2 relevant chunks by cosine similarity and injects them into the message.

**Token optimization:** Once `return_video_analysis` has been called, the large transcript `ToolMessage` is truncated to 500 chars in `call_model`, reducing token cost for all subsequent messages.

### Notes

- Videos must have captions enabled (auto-generated captions work).
- Only English captions are supported by default.
