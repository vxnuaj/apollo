```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FRONTEND (Next.js)                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │  Session 1  │  │  Session 2  │  │  Session 3  │  │   ...100    │       │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘       │
└─────────┼─────────────────┼─────────────────┼─────────────────┼─────────────┘
          │                 │                 │                 │
          │ HTTP POST       │ HTTP POST       │ HTTP POST       │
          │ /chat/          │ /chat/          │ /chat/          │
          │ completions     │ completions     │ completions     │
          │                 │                 │                 │
          ▼                 ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FASTAPI SERVER (Uvicorn)                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         API Routes                                     │  │
│  │  • POST /v1/chat/completions       (streaming SSE)                    │  │
│  │  • GET  /v1/sessions/{id}/messages (get history)                      │  │
│  │  • POST /v1/sessions               (create new session)               │  │
│  │  • GET  /health                                                        │  │
│  │  • GET  /metrics                                                       │  │
│  └────────────────────────────┬──────────────────────────────────────────┘  │
│                                │                                             │
│                                ▼                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      Request Handler                                   │  │
│  │  • Parse request                                                       │  │
│  │  • Validate session_id                                                 │  │
│  │  • Load conversation history from DB ────────────┐                    │  │
│  │  • Add to request queue                          │                    │  │
│  └────────────────────────────┬─────────────────────┼────────────────────┘  │
└─────────────────────────────────┼─────────────────────┼─────────────────────┘
                                  │                     │
                                  ▼                     ▼
                                                ┌──────────────────────────────┐
                                                │   PostgreSQL Database        │
                                                │                              │
                                                │  conversations               │
                                                │  ├─ id (UUID, PK)           │
                                                │  ├─ session_id (UUID)       │
                                                │  ├─ title (TEXT)            │
                                                │  ├─ created_at              │
                                                │  └─ updated_at              │
                                                │                              │
                                                │  messages                    │
                                                │  ├─ id (UUID, PK)           │
                                                │  ├─ conversation_id (FK)    │
                                                │  ├─ role (TEXT)             │
                                                │  ├─ content (TEXT)          │
                                                │  └─ created_at              │
                                                └──────────────┬───────────────┘
                                                               │
                                  ┌────────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         REQUEST QUEUE MANAGER                                │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │  Priority Queue (asyncio.Queue)                                     │     │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐          │     │
│  │  │ Request1 │→│ Request2 │→│ Request3 │→│ Request4 │→ ...      │     │
│  │  │ User: 1  │  │ User: 2  │  │ User: 3  │  │ User: 4  │          │     │
│  │  │ Token: 0 │  │ Token: 5 │  │ Token: 2 │  │ Token: 0 │          │     │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘          │     │
│  └────────────────────────────┬───────────────────────────────────────┘     │
│                                │                                             │
│                                ▼                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      Batch Scheduler                                   │  │
│  │  • Collect requests (up to batch_size=32)                             │  │
│  │  • Pad sequences to same length                                       │  │
│  │  • Create attention masks                                             │  │
│  │  • Send batch to inference engine                                     │  │
│  └────────────────────────────┬──────────────────────────────────────────┘  │
└─────────────────────────────────┼───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        INFERENCE ENGINE (JAX/Flax)                           │
│                                                                               │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      Session Manager                                   │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │  Session Cache (LRU)                                             │  │  │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │  │  │
│  │  │  │ Session 1   │  │ Session 2   │  │ Session 3   │  ...       │  │  │
│  │  │  │ KV Cache    │  │ KV Cache    │  │ KV Cache    │            │  │  │
│  │  │  │ History     │  │ History     │  │ History     │            │  │  │
│  │  │  └─────────────┘  └─────────────┘  └─────────────┘            │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────┬──────────────────────────────────────────┘  │
│                                │                                             │
│                                ▼                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    Gemma Model (JAX/Flax)                              │  │
│  │                                                                         │  │
│  │  Input: [B, L] token IDs (batched, padded)                            │  │
│  │         └─> B = batch_size (up to 32 users)                           │  │
│  │         └─> L = sequence_length (variable per user)                   │  │
│  │                                                                         │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │  Embedder                                                        │  │  │
│  │  │    • Lookup embeddings                                           │  │  │
│  │  │    • Scale by sqrt(embed_dim)                                    │  │  │
│  │  └──────────────────────────┬──────────────────────────────────────┘  │  │
│  │                              │                                          │  │
│  │                              ▼                                          │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │  Transformer Layers (18x)                                        │  │  │
│  │  │  ┌───────────────────────────────────────────────────────────┐  │  │  │
│  │  │  │ TransformerBlock                                           │  │  │  │
│  │  │  │   • Pre-attention RMSNorm                                  │  │  │  │
│  │  │  │   • Grouped Query Attention (4 heads, 1 KV head)          │  │  │  │
│  │  │  │     - Q/K/V projections                                    │  │  │  │
│  │  │  │     - QK RMSNorm                                           │  │  │  │
│  │  │  │     - RoPE (Rotary Position Embeddings)                   │  │  │  │
│  │  │  │     - Attention computation                                │  │  │  │
│  │  │  │   • Post-attention RMSNorm                                 │  │  │  │
│  │  │  │   • Residual connection                                    │  │  │  │
│  │  │  │   • Pre-FFN RMSNorm                                        │  │  │  │
│  │  │  │   • GeGLU (Gated Linear Unit)                             │  │  │  │
│  │  │  │   • Post-FFN RMSNorm                                       │  │  │  │
│  │  │  │   • Residual connection                                    │  │  │  │
│  │  │  └───────────────────────────────────────────────────────────┘  │  │  │
│  │  │  (Repeat 18 times)                                               │  │  │
│  │  └──────────────────────────┬──────────────────────────────────────┘  │  │
│  │                              │                                          │  │
│  │                              ▼                                          │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │  Final Layer                                                     │  │  │
│  │  │    • Final RMSNorm                                               │  │  │
│  │  │    • Logits projection (weight tying with embedder)             │  │  │
│  │  └──────────────────────────┬──────────────────────────────────────┘  │  │
│  │                              │                                          │  │
│  │                              ▼                                          │  │
│  │  Output: [B, L, vocab_size] logits                                    │  │
│  │           └─> vocab_size = 262,144 tokens                             │  │
│  └────────────────────────────┬──────────────────────────────────────────┘  │
│                                │                                             │
│                                ▼                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      Token Sampler                                     │  │
│  │  • Get logits for last position [B, vocab_size]                       │  │
│  │  • Apply temperature scaling                                          │  │
│  │  • Top-k / Top-p sampling (nucleus sampling)                          │  │
│  │  • Sample next token for each user in batch                           │  │
│  │  • Return: [B] token IDs                                              │  │
│  └────────────────────────────┬──────────────────────────────────────────┘  │
└─────────────────────────────────┼───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RESPONSE DISTRIBUTOR                                 │
│  • Decode tokens to text (SentencePiece)                                    │
│  • Route each token back to correct user session                            │
│  • Send via SSE stream                                                       │
│  • Check for EOS (end of sequence) per user                                 │
│  • Save completed message to database ───────────────────────────┐          │
│  • Update session state                                          │          │
└────────────────────────────┬─────────────────────────────────────┼──────────┘
                             │                                     │
                             ▼                                     ▼
                                                        ┌──────────────────────┐
                                                        │  PostgreSQL Database │
                                                        │  • Save user message │
                                                        │  • Save AI response  │
                                                        │  • Update timestamp  │
                                                        └──────────────────────┘
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SSE (Server-Sent Events) Streams                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  User 1      │  │  User 2      │  │  User 3      │  │  User 100    │   │
│  │  data: token │  │  data: token │  │  data: token │  │  data: token │   │
│  │  data: token │  │  data: token │  │  data: token │  │  data: token │   │
│  │  data: done  │  │  data: token │  │  data: done  │  │  data: token │   │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FRONTEND (Next.js)                                 │
│  • Receive tokens via SSE                                                    │
│  • Append to message content                                                 │
│  • Display in chat interface                                                 │
└─────────────────────────────────────────────────────────────────────────────┘


MONITORING & METRICS (Prometheus)
═════════════════════════════════
┌──────────────────────────────────────┐
│ • Requests/sec                       │
│ • Avg latency (TTFT, total)         │
│ • Active sessions                    │
│ • Queue depth                        │
│ • Batch utilization                  │
│ • GPU memory usage                   │
│ • Token throughput                   │
└──────────────────────────────────────┘
```