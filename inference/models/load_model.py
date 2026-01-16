from pprint import pprint
from model_registry import MODEL_REGISTRY, gemma_3_270m_config

def load_model(model:str): # model= Union["gemma/gemma-270m", "deepseek/deepseek-r1", etc]
    model_name, model_param_name = model.split("/")
    try:
        MODEL_REGISTRY[model_name][model_param_name]
    except Exception as e:
        print(f"{model} not found in model_registry")
        
    model = _load_model_instance(model)        
        
    return model

def _load_model_instance(model:str): # model= Union["gemma/gemma-270m", "deepseek/deepseek-r1", etc]
    model_name, model_param_name = model.split("/") 
    if model_name == "gemma":
        from gemma._gemma import Gemma
        from gemma._gemma import GemmaConfig, EmbeddingConfig, TransformerBlockConfig, AttentionConfig
        if model_param_name == "gemma-3-270m": model_config = gemma_3_270m_config()         
  
        embedding_config = EmbeddingConfig(
            num_embeddings = model_config["vocab_size"],
            features = model_config["embed_dim"],
            dtype = model_config["dtype"],
            param_dtype = model_config["param_dtype"],
            embedding_init = model_config["embedding_initializer"]
        )
       
        attn_config = AttentionConfig(
            num_query_heads = model_config["num_query_heads"],
            num_kv_heads = model_config["num_kv_heads"],
            head_dim = model_config["head_dim"],
            rope_base_frequency = model_config["rope_base_frequency"],
            rope_scale_factor = model_config["rope_scale_factor"],
            rope_proportion = model_config["rope_proportion"]
        ) 
        
        transformer_block_config = TransformerBlockConfig(
            attn_config = attn_config,
            ffn_hidden_dim = model_config["ffn_hidden_dim"],
            embed_dim = model_config["embed_dim"]
        )
        
        model_config = GemmaConfig(
            embedding_config = embedding_config,
            transformer_block_config = transformer_block_config,
            num_layers = model_config["num_layers"]
        ) 
        
        model = Gemma(
            config = model_config,
        )
    
    return model

if __name__ == "__main__":
    model = "gemma/gemma-3-270m"
    import jax 
    import jax.numpy as jnp
    x = jnp.ones((2, 10), dtype = jnp.int32)
    print(_load_model_instance(model).tabulate(jax.random.key(0), x))