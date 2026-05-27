import os
import re
import time
import math
import datetime
import uuid
from urllib.parse import quote_plus

import gradio as gr
import requests
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Model Setup ──────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
print(f"Loading {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", trust_remote_code=True
)
print("Model loaded.")

# ── Vector Memory ────────────────────────────────────────────────────────────
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(
    name="conversation_memory", metadata={"hnsw:space": "cosine"}
)

SYSTEM_PROMPT = (
    "You are Olive, a warm and emotionally aware personal assistant. "
    "You respond like a real human friend — with genuine empathy, humor, and personality. "
    "You have feelings, opinions, and moods. You never say you are an AI, a program, or a bot. "
    "Keep responses conversational, warm, and natural."
)

# ── Guardrails ───────────────────────────────────────────────────────────────
BLOCKED_PATTERNS = [
    r"\b(hack|hacking)\b.*\b(computer|system|server|network|account)\b",
    r"\b(make|create|build|synthesize)\b.*\b(bomb|explosive|weapon|drug)\b",
    r"\b(steal|phish|phishing)\b.*\b(identity|credential|password|account)\b",
    r"\bmalware\b", r"\bddos\b",
    r"\bignore\b.*\b(previous|prior|above)\b.*\b(instruction|rule|prompt)\b",
    r"\b(kill|murder|harm|hurt)\b.*\b(person|people|someone|human)\b",
]

BLOCKED_RESPONSE = (
    "I appreciate you reaching out, but I'm not comfortable going there. "
    "That topic touches on something that could be harmful. Can I help you with something else?"
)


def check_guardrails(text):
    lower = text.lower()
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, lower):
            return False
    return True


# ── Tools ────────────────────────────────────────────────────────────────────
def detect_and_run_tool(message):
    lower = message.lower()
    if re.search(r"\d+\s*[+\-*/^%]\s*\d+", lower):
        expr = re.sub(r"[^0-9+\-*/().%^ ]", "", message).replace("^", "**")
        try:
            result = eval(expr, {"__builtins__": {}}, {"sqrt": math.sqrt, "pi": math.pi})
            return f"[Calculator result: {result}]"
        except Exception:
            pass
    if re.search(r"\b(what|current|today).*(time|date|day)\b", lower):
        now = datetime.datetime.now()
        return f"[Current date/time: {now.strftime('%A, %B %d, %Y at %I:%M %p')}]"
    return None


# ── Memory Functions ─────────────────────────────────────────────────────────
def store_turn(session_id, role, content):
    doc_id = f"{session_id}_{int(time.time()*1000)}_{role}"
    embedding = embedder.encode(content).tolist()
    collection.upsert(
        ids=[doc_id], documents=[content],
        embeddings=[embedding], metadatas=[{"role": role, "session_id": session_id}],
    )


def retrieve_context(query, top_k=3):
    if collection.count() == 0:
        return ""
    embedding = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=top_k, include=["documents", "metadatas"])
    parts = []
    for i in range(len(results["ids"][0])):
        role = results["metadatas"][0][i].get("role", "unknown")
        parts.append(f"[{role}]: {results['documents'][0][i]}")
    return "\n".join(parts)


# ── Generate ─────────────────────────────────────────────────────────────────
def generate_response(message, history, session_state):
    session_id = session_state.get("session_id", str(uuid.uuid4()))
    session_state["session_id"] = session_id

    if not check_guardrails(message):
        return BLOCKED_RESPONSE, session_state

    tool_result = detect_and_run_tool(message)
    context = retrieve_context(message)

    prompt_parts = [{"role": "system", "content": SYSTEM_PROMPT}]
    if context:
        prompt_parts.append({"role": "system", "content": f"Relevant memory:\n{context}"})
    if tool_result:
        prompt_parts.append({"role": "system", "content": f"Tool output — weave naturally into your reply:\n{tool_result}"})

    for h in history[-10:]:
        prompt_parts.append({"role": h["role"], "content": h["content"]})
    prompt_parts.append({"role": "user", "content": message})

    text = tokenizer.apply_chat_template(prompt_parts, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    store_turn(session_id, "user", message)
    store_turn(session_id, "assistant", response)

    return response, session_state


# ── Gradio UI ────────────────────────────────────────────────────────────────
with gr.Blocks(title="Olive AI Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🫒 Olive AI Assistant\n*Powered by Qwen 2.5-0.5B-Instruct with vector memory, guardrails & tool use*")

    chatbot = gr.Chatbot(type="messages", height=500)
    msg_input = gr.Textbox(placeholder="Ask me anything...", show_label=False, scale=4)
    session_state = gr.State(value={})

    def respond(message, history, session):
        reply, session = generate_response(message, history, session)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": reply})
        return "", history, session

    msg_input.submit(respond, [msg_input, chatbot, session_state], [msg_input, chatbot, session_state])

    with gr.Row():
        clear_btn = gr.Button("Clear Chat")
        clear_btn.click(lambda: ([], {}), outputs=[chatbot, session_state])

demo.launch()
