from flax import linen as nn

MODEL_REGISTRY = {
    "gemma": {
        "gemma-3-270m": {
            "model_path": "gemma/gemma-3-270m",
            "tokenizer_path": "gemma/gemma-3-270m/tokenizer.model",
            "context_len": 32_000,
            "config": {
                "vocab_size": 262144,
                "embed_dim": 640,
                "dtype": "fp32",
                "param_dtype": "fp32",
                "num_q_heads": 4,
                "num_kv_heads": 1,
                "head_dim": 256,
                "ffn_hidden_dim": 2048,
                "num_layer": 18,
                "embedding_initializer": nn.initializers.normal(stddev = .02),
                "num_query_heads": 4,
                "num_kv_heads": 1, # multiquery attention,
                "head_dim": 256,
                "rope_base_frequency": 10000,
                "rope_scale_factor": 1,
                "rope_proportion": 1,
                "num_layers": 18
            }
        }
    }    
}

def gemma_3_270m_config():
    return MODEL_REGISTRY["gemma"]["gemma-3-270m"]["config"]