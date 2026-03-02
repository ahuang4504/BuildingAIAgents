# Topic 6: Vision-Language Model Exercises

Note: I had Claude Code help format this README.md since it's good at helping with documentation.

## Table of Contents

- [Exercise 1: Vision-Language Chat Agent](#exercise-1-vision-language-chat-agent)
  - [Overview](#overview)
  - [Requirements](#requirements)
  - [Setup](#setup)
  - [Usage](#usage)
  - [Graph Structure](#graph-structure)
  - [Image Handling](#image-handling)
  - [Notes](#notes)
- [Exercise 2: Video Surveillance Agent](#exercise-2-video-surveillance-agent)
  - [Overview](#overview-1)
  - [Additional Dependency](#additional-dependency)
  - [Usage](#usage-1)
  - [How It Works](#how-it-works)
  - [Notes](#notes-1)

---

# Exercise 1: Vision-Language Chat Agent

A multi-turn chat agent that reasons about uploaded images using ollama's `llava` model — built with LangGraph and surfaced via a Gradio web UI.

## Overview

This agent lets you upload an image and hold a multi-turn conversation about it. Each message is processed by a LangGraph pipeline that:

1. Receives the full conversation history from a SQLite-backed checkpoint
2. Formats the messages for the `llava` vision-language model (only the most recent image is sent as bytes; prior image turns are sent as text context)
3. Calls `llava` via ollama and returns the response
4. Displays the reply in the Gradio chat panel

Conversations persist across server restarts via `chats.db` (created next to the script). You can copy a Thread ID and reload the full conversation — including image thumbnails — at any time.

## Requirements

- Python 3.10+
- [ollama](https://ollama.com) installed and running locally
- The `llava` model pulled via ollama

### Dependencies

```
ollama
pillow
langchain-core
langgraph
langgraph-checkpoint-sqlite
gradio
```

Install into the shared venv:

```bash
pip install ollama pillow langchain-core langgraph langgraph-checkpoint-sqlite gradio
```

## Setup

1. Start the ollama server:

```bash
ollama serve
```

2. Pull the llava model (one-time):

```bash
ollama pull llava
```

## Usage

```bash
python vlm_chat_agent.py
```

Then open the Gradio URL printed in the terminal (e.g., `http://127.0.0.1:7860`).

1. A **Thread ID** is auto-generated on page load — copy it to resume the conversation later
2. **Upload an image** using the image panel — it auto-submits with the prompt "Describe the picture"
3. **Type follow-up questions** in the text box and press Enter or click Send
4. To resume a past conversation: paste the Thread ID and click **Load**
5. To start fresh: click **New Chat** (generates a new Thread ID)

## Graph Structure

```
Gradio UI (image upload or text input)
    │
    ▼ upload_image() or send_message()
HumanMessage added to LangGraph state (stored in SQLite)
    │
    ▼ agent.invoke({messages}, thread_id)
LangGraph: START → call_vlm → END
    │
    ▼ result["messages"][-1].content
AI reply appended to Gradio chatbot display
```

The graph is a single `call_vlm` node. `SqliteSaver` checkpoints the full message history (including image bytes) to `chats.db` so conversations survive server restarts.

## Image Handling

Uploaded images are:

- Resized to a maximum of 512px on the longest side (maintains aspect ratio)
- Encoded as JPEG and stored as base64 in the LangGraph checkpoint
- Only the **most recent** image-bearing message sends image bytes to `llava` — earlier image turns send text only, avoiding multi-image confusion in LLaVA
- Displayed inline as chat bubbles in the Gradio UI
- Reconstructed from base64 into temp files when loading a saved thread

## Notes

- Requires ollama to be running locally before starting the app
- `llava` runs entirely on your machine — no data is sent to external APIs
- `chats.db` is created in `Topic6VLM/` on first run
- Each browser tab gets its own Thread ID — multiple sessions don't interfere

---

## Exercise 2: Video Surveillance Agent

### Overview

`video_surveillance_agent.py` analyzes a video clip to detect when a person enters and exits the scene. Because LLaVA cannot process video directly, the agent breaks the video into individual frames and queries LLaVA on each one.

### Additional Dependency

```bash
pip install opencv-python
```

### Usage

```bash
python video_surveillance_agent.py <path/to/video.mp4>
```

### How It Works

1. OpenCV extracts one frame every 2 seconds from the video
2. Each frame is resized to ≤512px (longest side), JPEG-encoded, and base64-encoded
3. LLaVA is asked: _"Is there a person visible in this image? Reply with only Yes or No."_
4. A state machine tracks No→Yes (person entered) and Yes→No (person exited) transitions
5. To handle LLaVA's occasional false negatives, **2 consecutive "No" frames** are required before an exit is logged (debouncing)
6. Results are printed with MM:SS timestamps

### Example Output

```
[00:00] Frame 1/60 — No person
[00:02] Frame 2/60 — Person detected
...
Person ENTERED at 00:02
Person EXITED at 00:14
```

### Notes

- Requires `ollama serve` running with the `llava` model pulled
- Frames are sampled at ~2-second intervals; sub-2-second appearances may be missed
- The debounce threshold (`DEBOUNCE = 2`) can be increased if spurious enter/exit pairs still appear
- LLaVA runs entirely on your machine — no data is sent to external APIs
- Processing speed depends on video length and hardware; expect ~1–5 seconds per frame
