┌───────────────────────────────────────────────────────────────┐
│                       FRONTEND (Next.js)                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│  │ Session 1   │  │ Session 2   │  │ Session 3   │  ...      │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘           │
└─────────┼───────────────┼───────────────┼───────────────┘
          │               │               │
          │ HTTP POST     │ HTTP POST     │ HTTP POST
          │ /chat/        │ /chat/        │ /chat/
          ▼               ▼               ▼
┌───────────────────────────────────────────────┐
│            CPU Backend / Orchestrator         │
│                                               │
│  ┌─────────────────────────────────────────┐  │
│  │ API Routes / Request Handler             │  │
│  │ • Validate session, parse request       │  │
│  │ • Load conversation history from DB     │  │
│  │ • Add request to queue / batching       │  │
│  └───────────────┬─────────────────────────┘  │
│                  │                               
│                  ▼                               
│  ┌─────────────────────────────────────────┐  
│  │ Request Queue / Batch Scheduler         │  
│  │ • Collect requests (batch_size=32)     │  
│  │ • Pad sequences, create attention masks│  
│  │ • Forward batch to GPU worker(s)       │  
│  └───────────────┬─────────────────────────┘  
└──────────────────┼───────────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────────────┐
│               GPU Worker Node(s)              │
│  ┌─────────────────────────────────────────┐  │
│  │ Checkpoint Loading (Startup)            │  │
│  │ • Load model weights into GPU memory    │  │
│  │ • Initialize JAX/Flax session           │  │
│  └───────────────┬─────────────────────────┘  │
│                  │
│                  ▼
│  ┌─────────────────────────────────────────┐
│  │ Inference Engine (JAX/Flax)             │
│  │ • Session Manager (KV caches, histories)│
│  │ • Gemma Model (batched input tokens)   │
│  │ • Transformer layers, embeddings, FFN  │
│  │ • Token sampling (temperature, top-k/p)│
│  └───────────────┬─────────────────────────┘
│                  │
│                  ▼
│  ┌─────────────────────────────────────────┐
│  │ Response Distributor                      │
│  │ • Decode tokens → text (SentencePiece)  │
│  │ • Route tokens to correct session       │
│  │ • Stream via SSE to frontend            │
│  │ • Save completed messages to DB         │
│  └─────────────────────────────────────────┘
└───────────────────────────────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────────────┐
│ PostgreSQL Database                            │
│ • Store conversations, sessions, messages     │
└───────────────────────────────────────────────┘

MONITORING & METRICS (Prometheus)
═════════════════════════════════
┌──────────────────────────────────────┐
│ • Requests/sec                       │
│ • Avg latency (TTFT, total)          │
│ • Active sessions                     │
│ • Queue depth                         │
│ • Batch utilization                   │
│ • GPU memory usage                    │
│ • Token throughput                    │
└──────────────────────────────────────┘


-----
### Inference Server

Client/Backend
      │
      ▼
WebSocket Layer
      │  (receives prompt, streams tokens)
      ▼
Session Manager
      │  (tracks tokens generated, session metadata)
      ▼
Token Generator
      │  (autoregressive loop)
      ▼
Model Loader / Model Weights
      │  (apply model to current token sequence on GPU)
      ▼
Output Token
      │
      └─────────> streamed back to backend
