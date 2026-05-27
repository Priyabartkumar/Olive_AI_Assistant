# Cost & Latency Table — OSS Model Deployment

## Current Deployment: Qwen2.5-72B-Instruct via HuggingFace Inference API

### Architecture
- **Model**: Qwen2.5-72B-Instruct (runs on HuggingFace's servers, not on the Space)
- **Space**: Hosts only the Gradio UI + lightweight logic (~200MB RAM)
- **Dependencies**: gradio, huggingface-hub (no torch/transformers needed)

### Deployment Options

| Platform | Tier | Cost | Cold Start | Avg Latency (per request) | Notes |
|----------|------|------|------------|---------------------------|-------|
| **HuggingFace Spaces + Inference API** | Free | $0/month | 30-60s (on sleep) | 3-5s | Sleeps after 48h inactivity, API rate-limited |
| **HuggingFace Spaces + Inference API** | PRO ($9/mo) | $9/month | None | 2-3s | Always-on Space, higher API rate limits |
| **HuggingFace Inference Endpoints** | Dedicated | $1.30/hr (A10G) | None | 0.5-1.5s | Dedicated GPU, no rate limits, production-grade |

### Estimated Monthly Costs (1000 requests/day)

| Setup | Monthly Cost | Latency Profile |
|-------|-------------|-----------------|
| HF Spaces Free + Inference API Free | **$0** | 3-5s avg, 60s cold start after sleep |
| HF Spaces Free + Inference API PRO | **$9** | 2-3s avg, higher throughput |
| HF Inference Endpoints (A10G) | **~$936** | 0.5-1.5s avg, production-ready |

### Previous Approach (Local 0.5B Model) — Abandoned

| Aspect | Local 0.5B | API-based 72B (current) |
|--------|-----------|------------------------|
| Model quality | Poor (ignored persona) | Excellent (follows system prompt) |
| Response time | 15-30s on CPU | 3-5s |
| Build time | ~10 min (4GB downloads) | ~1 min (200MB) |
| RAM usage | ~4GB | ~200MB |
| Multi-turn | Timed out on 2nd message | Works reliably |
| Dependencies | 7 packages (torch, transformers, etc.) | 2 packages |

### Why the Switch?

The 0.5B model running locally on free CPU:
1. Was too small to follow the Olive persona consistently
2. Timed out on multi-turn conversations due to slow CPU inference
3. sentence-transformers embedding added 3-5s per call on CPU
4. Required ~4GB of downloads, making builds slow

The Inference API approach offloads heavy computation to HuggingFace's GPU servers while the Space only serves the lightweight UI.

### Recommendation

**HuggingFace Spaces (Free) + Inference API** is the best setup for demos:
- Zero cost
- Fast responses (3-5s)
- No infrastructure management
- 72B model quality (much better than local 0.5B)
- Upgrade to PRO ($9/mo) for higher rate limits and always-on Space
