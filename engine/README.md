### Init.

Build the Architecture for the model in order to load weights.

### Core / Must-Have Features
1. Continuous / Dynamic Batching  
   - Dynamically form batches at every decode step  
   - Add new requests as slots free up, evict finished sequences immediately  

2. Paged / Block-based KV Cache (inspired by PagedAttention)  
   - Split KV cache into fixed-size non-contiguous blocks/pages  
   - Use block table (Python dict + JAX arrays) to map logical → physical blocks  
   - Enables near-100% memory utilization, no padding waste  

3. Efficient KV Cache Management & State  
   - Functional updates via carry in `jax.lax.scan` or careful mutable arrays  
   - Incremental growth during decode without full reallocation  

4. Separate Prefill vs. Decode Paths  
   - Two distinct JIT-compiled functions:  
     - Prefill: prompt processing (compute-heavy, full attention)  
     - Decode: token-by-token generation (memory-bound, cache append)  

5. Prefix / Radix / Trie-based Prefix Caching  
   - Cache & reuse KV for shared prompt prefixes (system prompts, few-shot, instructions)  
   - Hash prefixes in Python, store reusable KV segments, append user-specific tokens  

### Strong Throughput & Scalability Multipliers
6. Quantization Support (weights + activations + KV cache)  
   - Minimum: int8 / fp8 weights, bfloat16 compute  
   - Bonus: quantized KV cache (int8/fp8) to fit more concurrent users  
   - Dequant during matmul using JAX primitives  

7. Grouped-Query Attention (GQA) / Multi-Query Attention (MQA) awareness  
   - Reduced KV heads, shared projections → dramatically smaller KV cache  

8. Speculative Decoding  
   - Use smaller draft model or assisted generation to guess several tokens ahead  
   - Verify in one forward pass → 2–4× effective tokens/second  

9. Model Sharding / Tensor Parallelism (JAX sharding primitives)  
   - Shard weights across devices (hidden dim or heads)  
   - Use `jax.sharding.NamedSharding` or `jax.pmap` + manual collectives  
   - Replicate activations where possible to minimize decode communication  

10. Request Scheduler with Priority / Fairness  
    - Round-robin or length-aware scheduling  
    - Optional: token budgets, rate limiting, SLA-aware priority queues  

### Production Polish & Reliability Features
11. Streaming Output  
    - Yield tokens as generated (Python generator + async queue)  

12. OpenAI-compatible API Endpoint  
    - `/v1/chat/completions` style interface (FastAPI/asyncio server)  

13. Batched Sampling Methods  
    - Greedy, top-p, top-k, temperature — vectorized with `jax.lax.top_k`, etc.  

14. Memory Eviction & LRU Cache  
    - Automatic eviction of idle KV caches (weakref or timed LRU)  

15. Profiling & Metrics  
    - Prometheus-style: tokens/sec, TTFT, TPOT, GPU util, batch size, memory  
    - Integrate `jax.profiler` traces for debugging  

### Suggested Prioritization Roadmap
- Phase 1 (basic serving at decent speed): 1, 3, 4  
- Phase 2 (high throughput with many users): 2, 5, 6  
- Phase 3 (production quality & extreme perf): 7–15