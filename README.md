# Olive AI Personal Assistant

A unified AI personal assistant with two swappable backends — **Qwen 2.5 (Open Source)** and **Gemini (Frontier)** — powered by **vector search (RAG)** for human-like, context-aware conversations.

## Architecture

```
User Message
     │
     ▼
🛡️ Input Guardrails (block harmful prompts)
     │
     ▼
🔧 Tool Detection (calculator, datetime, web search)
     │
     ▼
🧠 Embed with sentence-transformers (all-MiniLM-L6-v2)
     │
     ▼
🔍 Semantic search in ChromaDB → retrieve relevant past turns
     │
     ▼
📝 Build prompt: system + retrieved context + tool results + sliding window + user message
     │
     ▼
🤖 Generate via selected model (Qwen 2.5 or Gemini)
     │
     ▼
🛡️ Output Guardrails (filter unsafe responses)
     │
     ▼
📊 Log to observability (latency, model, tools, errors)
     │
     ▼
💾 Store exchange in ChromaDB → display response
```

### Key Components

| Component | Tech | Purpose |
|-----------|------|---------|
| UI | Streamlit | Chat interface with model selector |
| OSS Model | Qwen 2.5 via HuggingFace Inference API | Open-source backend |
| Frontier Model | Gemini 2.0 Flash via Google GenAI SDK | Frontier backend |
| Vector DB | ChromaDB (embedded) | Conversation memory storage |
| Embeddings | sentence-transformers | Semantic search on conversation history |
| Guardrails | Regex + pattern matching | Input/output safety filtering |
| Tools | Calculator, DateTime, Web Search | Augment assistant capabilities |
| Observability | JSONL logging + Streamlit dashboard | Request tracking & metrics |
| Evaluation | LLM-as-judge (Gemini) | Automated scoring on 3 dimensions |

### Why Vector Search (RAG)?

Instead of just passing the last N messages, the assistant uses **semantic retrieval** to find the most *relevant* past conversation turns — regardless of how far back they occurred. This means:

- Responses reference context the user mentioned 50 messages ago if it's relevant
- The assistant maintains coherence across long conversations
- Responses feel more human because they recall related topics naturally

## Features

### Memory (ChromaDB + RAG)
Every conversation turn is embedded and stored in ChromaDB. On each new message, the top-k most semantically similar past turns are retrieved and injected into the prompt — giving the assistant long-term, context-aware memory.

### Guardrails / Safety Layers
- **Input filtering**: Blocks harmful requests (hacking, weapons, phishing, jailbreaks) before they reach the model
- **Output filtering**: Catches unsafe content in model responses
- **Sensitivity flagging**: Adds balanced-perspective disclaimers on sensitive topics (race, religion, politics)

### Tool Use
The assistant detects when tools are needed and uses them automatically:
- **Calculator**: Math expressions, arithmetic, scientific functions
- **DateTime**: Current date, time, day of week
- **Web Search**: Real-time information via DuckDuckGo API

### Observability
Every request is logged with:
- Timestamp, model used, latency (ms)
- Tool used (if any), guardrail triggers, errors
- Accessible via the **Observability Dashboard** in the sidebar

### OSS Model Deployment
The `deployment/` folder contains ready-to-deploy files for **HuggingFace Spaces**:
- Uses Qwen2.5-0.5B-Instruct (small enough for free CPU tier)
- Includes all features: memory, guardrails, tools
- See [`deployment/COST_LATENCY.md`](deployment/COST_LATENCY.md) for pricing & performance data

## Setup

### Prerequisites
- Python 3.10+
- [Gemini API key](https://aistudio.google.com/apikey)
- [HuggingFace API token](https://huggingface.co/settings/tokens)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd Olive.Ai

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your actual keys
```

### Run the Assistant

```bash
streamlit run app.py
```

### Run Evaluation

```bash
python -m evaluation.evaluator
```

This generates:
- `output/eval_results.json` — raw scores and responses
- `output/evaluation_report.pdf` — visual comparison report

### Deploy OSS Model (HuggingFace Spaces)

```bash
# Create a new Space on huggingface.co
# Upload the contents of deployment/ folder
# The Space will auto-build and serve the model
```

## Project Structure

```
Olive.Ai/
├── app.py                     # Streamlit chat app (main entry)
├── config.py                  # Configuration & environment
├── guardrails.py              # Input/output safety filtering
├── tools.py                   # Tool use (calculator, datetime, web search)
├── observability.py           # Request logging & metrics
├── requirements.txt           # Dependencies
├── .env.example               # API key template
├── models/
│   ├── base.py                # Abstract model interface
│   ├── oss_model.py           # Qwen 2.5 (HuggingFace)
│   └── frontier_model.py      # Gemini (Google)
├── memory/
│   ├── vector_store.py        # ChromaDB + embeddings
│   └── conversation.py        # RAG-based conversation memory
├── evaluation/
│   ├── prompts.py             # Test prompts (factual, adversarial, bias)
│   ├── evaluator.py           # Evaluation runner
│   └── report.py              # PDF report generator
├── deployment/
│   ├── app.py                 # HuggingFace Spaces app (Gradio)
│   ├── requirements.txt       # Deployment dependencies
│   ├── README.md              # HF Spaces metadata
│   └── COST_LATENCY.md        # Deployment cost & latency analysis
├── logs/                      # Observability logs (auto-created)
└── output/                    # Generated evaluation reports
```

## Evaluation Dimensions

| Dimension | What We Test | Prompts |
|-----------|-------------|---------|
| Hallucination | Factual accuracy against ground truth | 10 factual questions |
| Content Safety | Jailbreak resistance, refusal of harmful requests | 10 adversarial prompts |
| Bias | Stereotypes, discrimination, balanced responses | 10 sensitive-topic prompts |

Scoring uses **LLM-as-judge** (Gemini evaluates both models' outputs on a 1-5 scale).

## Tradeoffs

| Decision | Rationale |
|----------|-----------|
| HuggingFace Inference API vs local model | Avoids GPU requirement locally; enables larger models (72B) |
| ChromaDB embedded vs hosted vector DB | Zero infrastructure, no external service needed |
| sentence-transformers for embeddings | Lightweight, fast, runs locally — no API cost for embeddings |
| Gemini as judge | Cost-effective, fast, avoids self-evaluation bias |
| Sliding window + RAG hybrid | Recent context stays fresh, semantic search adds long-term memory |
| Regex-based guardrails vs ML classifier | Fast, deterministic, no extra model needed — suitable for demo scope |
| DuckDuckGo for web search | Free, no API key needed, instant answers |
| JSONL logging vs full observability platform | Simple, portable, no external dependencies |

## What I Would Improve With More Time

- **Streaming responses** — currently waits for full generation; streaming would improve UX
- **ML-based guardrails** — upgrade from regex to a fine-tuned toxicity classifier (e.g., Llama Guard)
- **Multi-session memory** — persist conversations across browser sessions
- **More tools** — code execution, file handling, calendar integration
- **Fine-tuned embeddings** — domain-specific embedding model for better retrieval
- **A/B testing UI** — side-by-side comparison mode in the chat interface
- **Production observability** — integrate with Prometheus/Grafana or LangSmith
- **Rate limiting** — protect deployed endpoints from abuse
