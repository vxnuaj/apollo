import jax
import jax.numpy as jnp
from flax import linen as nn
import orbax.checkpoint as ocp
import absl.logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from _gemma import Gemma, TransformerBlock, Transformer
from _config import GemmaConfig, EmbeddingConfig, AttentionConfig, TransformerBlockConfig
from test_tokenizer import GemmaTokenizer

absl.logging.set_verbosity(absl.logging.ERROR)

def load_gemma_checkpoint(checkpoint_path: str):
    """Load Gemma checkpoint from OCDBT format."""
    checkpoint_path = os.path.abspath(checkpoint_path)
    checkpointer = ocp.StandardCheckpointer()
    checkpoint = checkpointer.restore(checkpoint_path)
    return checkpoint

def restructure_checkpoint(flat_checkpoint):
    """
    Convert flat checkpoint to Flax nested dict structure.
    
    Checkpoint format: 
      - {'transformer/embedder': {'input_embedding': array}}
      - {'transformer/layer_0/attn/q_einsum': {'w': array}}
      - {'transformer/layer_0/pre_attention_norm': {'scale': array}}
    
    Flax expects: 
      - {'transformer': {'input_embedding': array}}  # embedder becomes input_embedding  
      - {'transformer': {'layer_0': {'attn': {'q_einsum': array}}}}
      - {'transformer': {'layer_0': {'pre_attention_norm': {'scale': array}}}}
    
    The parameter names ('w', 'scale', etc.) need to be kept as a nested level!
    """
    nested = {}
    
    for key, value_dict in flat_checkpoint.items():
        parts = key.split('/')
        
        # Extract param name and array
        param_keys = list(value_dict.keys())
        if not param_keys:
            continue
            
        param_name = param_keys[0]  # 'w', 'scale', 'input_embedding'
        array_value = value_dict[param_name]
        
        # Special case: embedder should be renamed to input_embedding (no extra nesting)
        if parts[-1] == 'embedder':
            parts[-1] = param_name  # Replace 'embedder' with 'input_embedding'
            # Navigate and store directly (no param_name nesting for embedder)
            current = nested
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = array_value
        else:
            # For all other params, keep the param_name as an extra level
            # Navigate to the module level
            current = nested
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Special handling for einsum weights - extract 'w' directly
            if parts[-1].endswith('_einsum') or parts[-1] == 'linear':
                # For einsum weights and linear, store array directly (model uses param name directly)
                current[parts[-1]] = array_value
            else:
                # For norms and other params, keep the param_name nesting
                if parts[-1] not in current:
                    current[parts[-1]] = {}
                current[parts[-1]][param_name] = array_value
    
    return nested

def create_gemma_config():
    """Create Gemma-270M configuration."""
    vocab_size = 262144
    embed_dim = 640
    num_q_heads = 4
    num_kv_heads = 1
    head_dim = 256
    ffn_hidden_dim = 2048
    num_layers = 18
    
    return GemmaConfig(
        embedding_config = EmbeddingConfig(
            num_embeddings = vocab_size,
            features = embed_dim,
            dtype = jnp.float32,
            param_dtype = jnp.float32,
            embedding_init = nn.initializers.normal(stddev=0.02),
        ),
        transformer_block_config = TransformerBlockConfig(
            attn_config = AttentionConfig(
                num_query_heads = num_q_heads,
                num_kv_heads = num_kv_heads,
                head_dim = head_dim,
            ),
            ffn_hidden_dim = ffn_hidden_dim,
            embed_dim = embed_dim
        ),
        num_layers = num_layers
    )

def run_inference(model, params, input_ids, tokenizer=None, temperature=1.0, max_new_tokens=50, verbose=True, debug=False):
    """
    Run basic inference with the model.
    
    Args:
        model: The Gemma model
        params: Model parameters
        input_ids: Input token IDs [B, L]
        tokenizer: Optional tokenizer for decoding tokens
        temperature: Sampling temperature
        max_new_tokens: Number of tokens to generate
        verbose: Whether to print progress
        debug: Whether to print debug information
        
    Returns:
        Generated token IDs
    """
    def sample_token(logits, temperature, rng):
        """Sample next token from logits."""
        logits = logits / temperature
        probs = jax.nn.softmax(logits, axis=-1)
        return jax.random.categorical(rng, logits=jnp.log(probs))
    
    generated = input_ids.tolist()[0] if len(input_ids.shape) == 2 else input_ids.tolist()
    
    if verbose and tokenizer:
        print(f"\nPrompt: {tokenizer.decode(generated)}")
        print("\nGenerating:")
    
    # Debug first forward pass
    if debug:
        current_ids = jnp.array([generated])
        logits = model.apply({'params': params}, current_ids)
        print(f"\n[DEBUG] First forward pass:")
        print(f"  Input shape: {current_ids.shape}")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Logits dtype: {logits.dtype}")
        
        # Convert to float32 for display
        logits_f32 = logits.astype(jnp.float32)
        print(f"  Logits range: [{float(logits_f32.min()):.2f}, {float(logits_f32.max()):.2f}]")
        print(f"  Logits mean: {float(logits_f32.mean()):.2f}, std: {float(logits_f32.std()):.2f}")
        
        next_token_logits = logits[0, -1, :]
        top_5_indices = jnp.argsort(next_token_logits)[-5:][::-1]
        print(f"\n[DEBUG] Top 5 predictions for next token:")
        for idx in top_5_indices:
            token_text = tokenizer.decode([int(idx)])
            logit_val = float(next_token_logits[int(idx)])
            print(f"  Token {int(idx)}: '{token_text}' (logit: {logit_val:.2f})")
        print()
    
    for i in range(max_new_tokens):
        # Get current sequence
        current_ids = jnp.array([generated])
        
        # Forward pass
        logits = model.apply({'params': params}, current_ids)
        
        # Get logits for last token
        next_token_logits = logits[0, -1, :]
        
        # Sample next token
        rng = jax.random.PRNGKey(i)
        next_token = sample_token(next_token_logits, temperature, rng)
        
        # Append to sequence
        generated.append(int(next_token))
        
        # Decode and print the new token
        if verbose:
            if tokenizer:
                token_text = tokenizer.id_to_piece(int(next_token))
                print(f"{token_text}", end='', flush=True)
            else:
                print(f"Token {i+1}/{max_new_tokens}: {next_token}")
        
        # Stop if we hit EOS token
        if tokenizer and int(next_token) == tokenizer.eos_id:
            if verbose:
                print("\n[EOS]")
            break
    
    if verbose:
        print("\n")
    
    return jnp.array(generated)

if __name__ == "__main__":
    print("Loading Gemma-270M...")
    print("="*60)
    
    # Paths
    base_path = os.path.join(os.path.dirname(__file__), "..")
    checkpoint_path = os.path.join(base_path, "gemma-3-270m")
    tokenizer_path = os.path.join(checkpoint_path, "tokenizer.model")
    
    # Load tokenizer
    print("\n1. Loading tokenizer...")
    try:
        tokenizer = GemmaTokenizer(tokenizer_path)
        print(f"   ✓ Tokenizer loaded (vocab size: {tokenizer.vocab_size})")
    except Exception as e:
        print(f"   ✗ Error loading tokenizer: {e}")
        exit(1)
    
    # Load checkpoint
    print("\n2. Loading checkpoint...")
    try:
        flat_checkpoint = load_gemma_checkpoint(checkpoint_path)
        print(f"   ✓ Checkpoint loaded ({len(flat_checkpoint)} parameter groups)")
        
        checkpoint = restructure_checkpoint(flat_checkpoint)
        print(f"   ✓ Checkpoint restructured")
    except Exception as e:
        print(f"   ✗ Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Create model
    print("\n3. Initializing model...")
    try:
        config = create_gemma_config()
        model = Gemma(config=config)
        
        dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
        rng = jax.random.PRNGKey(0)
        variables = model.init(rng, dummy_input)
        
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(variables['params']))
        print(f"   ✓ Model initialized ({param_count:,} parameters)")
        
        # Debug: Check if checkpoint params match model structure
        print("\n4. Validating checkpoint compatibility...")
        checkpoint_param_count = sum(x.size for x in jax.tree_util.tree_leaves(checkpoint))
        print(f"   Checkpoint params: {checkpoint_param_count:,}")
        print(f"   Model params: {param_count:,}")
        
        # Check a specific weight to see if it's loading
        if 'transformer' in checkpoint and 'input_embedding' in checkpoint['transformer']:
            embed_shape = checkpoint['transformer']['input_embedding'].shape
            print(f"   Checkpoint embedder shape: {embed_shape}")
            print(f"   Checkpoint embedder dtype: {checkpoint['transformer']['input_embedding'].dtype}")
            
            # Check if a layer weight exists
            if 'layer_0' in checkpoint['transformer']:
                if 'q_einsum' in checkpoint['transformer']['layer_0'].get('attn', {}):
                    q_shape = checkpoint['transformer']['layer_0']['attn']['q_einsum'].shape
                    print(f"   Checkpoint layer_0 q_einsum shape: {q_shape}")
        
    except Exception as e:
        print(f"   ✗ Error initializing model: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Test that weights are actually different from random init
    print("\n5. Verifying checkpoint weights are being used...")
    try:
        test_input = jnp.array([[1, 2, 3]])
        
        # Get logits with checkpoint weights
        logits_checkpoint = model.apply({'params': checkpoint}, test_input)
        
        # Get logits with random init weights
        logits_random = model.apply({'params': variables['params']}, test_input)
        
        # They should be different
        diff = jnp.abs(logits_checkpoint - logits_random).mean()
        print(f"   Mean difference between checkpoint and random: {diff:.6f}")
        
        if diff < 0.01:
            print("   ⚠️  WARNING: Checkpoint and random weights are very similar!")
            print("   This suggests the checkpoint may not be loading correctly.")
        else:
            print("   ✓ Checkpoint weights are being used")
            
    except Exception as e:
        print(f"   ✗ Error testing weights: {e}")
        import traceback
        traceback.print_exc()
    
    # Test inference
    print("\n" + "="*60)
    print("Running inference...")
    print("="*60)
    
    # Test prompts
    test_prompts = [
        "The quick brown fox",
        "Once upon a time",
        "Hello, world!",
    ]
    
    try:
        for idx, prompt in enumerate(test_prompts):
            print(f"\n{'='*60}")
            print(f"Prompt: \"{prompt}\"")
            print('='*60)
            
            # Tokenize input
            input_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
            input_ids = jnp.array([input_ids])
            
            # Generate
            generated_ids = run_inference(
                model, 
                checkpoint, 
                input_ids,
                tokenizer=tokenizer,
                temperature=0.8, 
                max_new_tokens=30000,
                verbose=True,
                debug=(idx == 0)  # Debug only first prompt
            )
            
            # Decode full sequence
            full_text = tokenizer.decode(generated_ids.tolist())
            print(f"\nFull generated text:\n{full_text}")
        
        print("\n" + "="*60)
        print("✓ Inference completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Error during inference: {e}")
        import traceback
        traceback.print_exc()
