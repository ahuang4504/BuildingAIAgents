"""
Vision-Language Chat Agent with LangGraph and Gradio

This program demonstrates a multi-turn chat agent that:
- Accepts an image upload via Gradio web UI (single image, auto-submits)
- Reasons about the image using ollama's llava model
- Maintains conversation history across multiple turns
- Uses LangGraph for structured state management (state + nodes + edges)
- Uses SqliteSaver for persistent conversation storage across server restarts
- Surfaces the interface via a Gradio web UI with chat recovery
"""

import io, base64, uuid, sqlite3, tempfile, os
from typing import TypedDict, Annotated, Sequence

import ollama
from PIL import Image
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import gradio as gr

MAX_SIZE = 512      # resize longest side to this to reduce latency
IMAGE_HEIGHT = 300  # px — change this to resize all image upload cells


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def pil_to_b64(img: Image.Image, max_size: int = MAX_SIZE) -> str:
    if max(img.size) > max_size:
        img = img.copy()
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def b64_to_tempfile(b64_str: str) -> str:
    data = base64.b64decode(b64_str)
    fd, path = tempfile.mkstemp(suffix=".jpg")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path

def call_vlm(state: State) -> dict:
    # node to call the llava VLM via ollama with the current conversation history.
    # LLaVA only handles one image context reliably: only send images for the
    # most recent human message that contains them; earlier turns use text only.
    messages = state["messages"]

    # Find the index of the last HumanMessage that carries images
    last_image_idx = -1
    for i, msg in enumerate(messages):
        if isinstance(msg, HumanMessage) and isinstance(msg.content, list):
            if any(b["type"] == "image" for b in msg.content):
                last_image_idx = i

    ollama_messages = []
    for i, msg in enumerate(messages):
        if isinstance(msg, HumanMessage):
            if isinstance(msg.content, str):
                entry = {"role": "user", "content": msg.content}
            else:  # list of {"type": "text"|"image", ...} blocks
                text = next((b["text"] for b in msg.content if b["type"] == "text"), "")
                entry = {"role": "user", "content": text}
                # Only attach image bytes for the most recent image-bearing message
                if i == last_image_idx:
                    images = [b["data"] for b in msg.content if b["type"] == "image"]
                    if images:
                        entry["images"] = images
            ollama_messages.append(entry)
        elif isinstance(msg, AIMessage):
            ollama_messages.append({"role": "assistant", "content": msg.content})

    response = ollama.chat(model="llava", messages=ollama_messages)
    return {"messages": [AIMessage(content=response["message"]["content"])]}


builder = StateGraph(State)
builder.add_node("call_vlm", call_vlm)
builder.add_edge(START, "call_vlm")
builder.add_edge("call_vlm", END)

DB_PATH = os.path.join(os.path.dirname(__file__), "chats.db")
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
memory = SqliteSaver(conn)
agent = builder.compile(checkpointer=memory)


def build_gradio_history(messages) -> list:
    history = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            if isinstance(msg.content, str):
                history.append({"role": "user", "content": msg.content})
            else:
                blocks = []
                for b in msg.content:
                    if b["type"] == "image":
                        path = b64_to_tempfile(b["data"])
                        blocks.append({"type": "file", "file": {"path": path, "mime_type": "image/jpeg", "orig_name": "image.jpg"}})
                    elif b["type"] == "text" and b["text"]:
                        blocks.append({"type": "text", "text": b["text"]})
                history.append({"role": "user", "content": blocks})
        elif isinstance(msg, AIMessage):
            history.append({"role": "assistant", "content": msg.content})
    return history


def upload_image(img: Image.Image, history: list, thread_id: str):
    b64 = pil_to_b64(img)
    content = [{"type": "text", "text": "Describe the picture"},
               {"type": "image", "data": b64}]
    result = agent.invoke(
        {"messages": [HumanMessage(content=content)]},
        config={"configurable": {"thread_id": thread_id}}
    )
    ai_response = result["messages"][-1].content
    path = b64_to_tempfile(b64)
    user_bubble = {"role": "user", "content": [
        {"type": "file", "file": {"path": path, "mime_type": "image/jpeg", "orig_name": "image.jpg"}},
        {"type": "text", "text": "Describe the picture"}
    ]}
    return history + [user_bubble, {"role": "assistant", "content": ai_response}], None


def send_message(message: str, history: list, thread_id: str):
    if not message.strip():
        return history, message
    result = agent.invoke(
        {"messages": [HumanMessage(content=message)]},
        config={"configurable": {"thread_id": thread_id}}
    )
    ai_response = result["messages"][-1].content
    return history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ai_response}
    ], ""


def load_thread(thread_id: str) -> list:
    state = agent.get_state({"configurable": {"thread_id": thread_id}})
    messages = state.values.get("messages", []) if state.values else []
    return build_gradio_history(messages)


def new_chat():
    return [], str(uuid.uuid4())


with gr.Blocks(title="VLM Chat Agent") as demo:
    gr.Markdown("# Vision-Language Chat Agent")

    with gr.Row():
        thread_id_box = gr.Textbox(label="Thread ID (copy to resume later)", interactive=True, scale=4)
        load_btn = gr.Button("Load", scale=1)
        new_btn = gr.Button("New Chat", scale=1)

    chatbot = gr.Chatbot(height=500)
    img_upload = gr.Image(type="pil", label="Upload image (auto-sends)", height=IMAGE_HEIGHT)

    with gr.Row():
        msg_box = gr.Textbox(placeholder="Type a message...", show_label=False, scale=4)
        send_btn = gr.Button("Send", scale=1)

    demo.load(lambda: str(uuid.uuid4()), [], [thread_id_box])

    img_upload.upload(upload_image, [img_upload, chatbot, thread_id_box], [chatbot, img_upload])
    send_btn.click(send_message, [msg_box, chatbot, thread_id_box], [chatbot, msg_box])
    msg_box.submit(send_message, [msg_box, chatbot, thread_id_box], [chatbot, msg_box])
    load_btn.click(load_thread, [thread_id_box], [chatbot])
    new_btn.click(new_chat, [], [chatbot, thread_id_box])


if __name__ == "__main__":
    demo.launch()
