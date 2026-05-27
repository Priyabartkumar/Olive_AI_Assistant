import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from memory.conversation import ConversationMemory
from models.oss_model import OSSModel
from models.frontier_model import FrontierModel
from guardrails import check_input, apply_guardrails
from tools import detect_tool, execute_tool
from observability import log_request, load_logs, get_summary

st.set_page_config(page_title="Olive AI Assistant", page_icon="🫒", layout="wide")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Olive AI Assistant")
    model_choice = st.selectbox(
        "Select Model",
        ["OSS (Qwen 2.5)", "Frontier (Gemini)"],
        key="model_choice",
    )

    if st.button("Clear Conversation"):
        st.session_state.pop("memory", None)
        st.session_state.pop("chat_display", None)
        st.rerun()

    st.divider()
    st.caption("**How it works**")
    st.caption(
        "Each message is embedded and stored in ChromaDB. "
        "When you send a new message, the assistant retrieves "
        "semantically similar past turns via vector search to "
        "build richer, more human-like responses."
    )

    st.divider()
    st.caption("**Features**")
    st.caption("🛡️ Guardrails — harmful input/output filtering")
    st.caption("🔧 Tools — calculator, date/time, web search")
    st.caption("📊 Observability — request logging & metrics")
    st.caption("🧠 Memory — vector search (ChromaDB + RAG)")

    if "memory" in st.session_state:
        st.divider()
        st.caption(f"Session: `{st.session_state.memory.session_id[:8]}...`")
        st.caption(f"Stored turns: {st.session_state.memory.store.count}")

    # ── Observability panel ──────────────────────────────────────────────────
    with st.expander("📊 Observability Dashboard"):
        logs = load_logs()
        summary = get_summary(logs)
        st.metric("Total Requests", summary["total_requests"])
        st.metric("Avg Latency", f"{summary['avg_latency_ms']} ms")
        st.metric("Guardrails Triggered", summary["guardrails_triggered"])
        st.metric("Errors", summary["errors"])
        if summary["models_used"]:
            st.caption("**Requests by model:**")
            for model_name, count in summary["models_used"].items():
                st.caption(f"  {model_name}: {count}")
        if summary["tools_used"]:
            st.caption("**Tools used:**")
            for tool_name, count in summary["tools_used"].items():
                st.caption(f"  {tool_name}: {count}")

# ── Init state ───────────────────────────────────────────────────────────────
if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()
if "chat_display" not in st.session_state:
    st.session_state.chat_display = []
if "models" not in st.session_state:
    st.session_state.models = {}


def get_model(choice: str):
    if choice not in st.session_state.models:
        if choice == "OSS (Qwen 2.5)":
            st.session_state.models[choice] = OSSModel()
        else:
            st.session_state.models[choice] = FrontierModel()
    return st.session_state.models[choice]


# ── Chat display ─────────────────────────────────────────────────────────────
for msg in st.session_state.chat_display:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── User input ───────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask me anything..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_display.append({"role": "user", "content": user_input})

    memory: ConversationMemory = st.session_state.memory
    model = get_model(model_choice)

    start_time = time.time()
    tool_used = None
    guardrail_triggered = False
    error_msg = None

    # Step 1: Check input guardrails
    safe_input, blocked_msg = check_input(user_input)

    if not safe_input:
        response = blocked_msg
        guardrail_triggered = True
    else:
        # Step 2: Check if a tool should handle this
        tool_used = detect_tool(user_input)

        if tool_used:
            tool_result = execute_tool(tool_used, user_input)
            # Let the model incorporate the tool result naturally
            messages = memory.build_prompt(user_input)
            messages.append({
                "role": "system",
                "content": (
                    f"Tool '{tool_used}' returned this result — weave it naturally "
                    f"into your response as if you knew it yourself:\n{tool_result}"
                ),
            })
        else:
            messages = memory.build_prompt(user_input)

        # Step 3: Generate response
        with st.chat_message("assistant"):
            with st.spinner(f"Thinking ({model.name})..."):
                try:
                    raw_response = model.generate(messages)
                    response = apply_guardrails(user_input, raw_response)
                    if response != raw_response:
                        guardrail_triggered = True
                except Exception as e:
                    response = f"Oops, something went wrong on my end. Let me try again — could you rephrase that?"
                    error_msg = str(e)
            st.markdown(response)
            if tool_used:
                st.caption(f"🔧 Used tool: {tool_used}")

    if safe_input is False:
        with st.chat_message("assistant"):
            st.markdown(response)

    latency_ms = (time.time() - start_time) * 1000

    # Step 4: Log for observability
    log_request(
        model_name=model.name,
        user_message=user_input,
        response=response,
        latency_ms=latency_ms,
        tool_used=tool_used,
        guardrail_triggered=guardrail_triggered,
        error=error_msg,
    )

    # Step 5: Store in vector memory
    memory.add_turn("user", user_input)
    memory.add_turn("assistant", response)
    st.session_state.chat_display.append({"role": "assistant", "content": response})
