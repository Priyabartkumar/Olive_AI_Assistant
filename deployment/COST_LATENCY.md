# Cost & Latency Table — OSS Model Deployment

## Model: Qwen2.5-0.5B-Instruct

### Deployment Options

| Platform | Tier | Cost | Cold Start | Avg Latency (per request) | GPU | Notes |
|----------|------|------|------------|---------------------------|-----|-------|
| **HuggingFace Spaces** | Free | $0/month | 30-60s (on sleep) | 2-5s | CPU (2 vCPU, 16GB RAM) | Sleeps after 48h inactivity |
| **HuggingFace Spaces** | Upgraded | $9/month | None | 1-3s | T4 GPU (16GB VRAM) | Always-on, faster inference |
| **Modal** | Serverless | ~$0.0003/request | 5-15s (cold) | 0.5-2s | T4/A10G | Pay-per-use, scales to zero |
| **RunPod** | Serverless | $0.00016/s (GPU) | 10-30s | 0.5-1.5s | A40/A100 available | Community cloud, cheapest GPU |
| **Replicate** | Serverless | ~$0.0002/request | 5-10s | 1-3s | T4/A40 | Simple API, pay-per-prediction |
| **Ollama (self-hosted)** | Self-hosted | Hardware cost only | None | 0.5-2s | Any local GPU | No ongoing cost, full control |

### Why Qwen2.5-0.5B-Instruct?

- **0.5B parameters** — small enough to run on CPU (free HF Spaces tier)
- Strong instruction-following despite small size
- Fast inference: ~50-100 tokens/sec on CPU, ~200+ tokens/sec on T4 GPU
- Fits in <2GB VRAM — works on virtually any GPU

### Estimated Monthly Costs (1000 requests/day)

| Platform | Monthly Cost | Latency Profile |
|----------|-------------|-----------------|
| HF Spaces (Free) | **$0** | 2-5s avg, 60s cold start after sleep |
| HF Spaces (T4) | **$9** | 1-3s avg, no cold start |
| Modal | **~$9** | 0.5-2s avg, 15s cold start |
| RunPod | **~$5-14** | 0.5-1.5s avg, 30s cold start |
| Replicate | **~$6** | 1-3s avg, 10s cold start |
| Ollama (local) | **$0** (+ hardware) | 0.5-2s avg, no cold start |

### Recommendation

**HuggingFace Spaces (Free tier)** is the best starting point:
- Zero cost for demos and evaluation
- No infrastructure management
- Direct integration with HF model hub
- Upgrade to T4 ($9/mo) when you need consistent performance

For production scale, **Modal** or **RunPod** offer the best cost-to-performance ratio with serverless scaling.
