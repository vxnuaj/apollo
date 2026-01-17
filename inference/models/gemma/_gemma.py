from flax import linen as nn
from flax.linen.initializers import xavier_normal, zeros_init
from flax.typing import Initializer
import jax
from models.gemma._config import GemmaConfig, EmbeddingConfig, AttentionConfig, TransformerBlockConfig
from models.gemma._math import apply_rope, create_causal_mask
from dataclasses import asdict
import jax.numpy as jnp

class Gemma(nn.Module):
    config: GemmaConfig
    
    @nn.compact
    def __call__(self, x: jax.Array):
        """
        Full Gemma model.
        
        Args:
            x: Input token IDs of shape [B, L]
            
        Returns:
            Logits of shape [B, L, vocab_size]
        """
        transformer = Transformer(config=self.config, name='transformer')
        return transformer(x)
    
class Transformer(nn.Module):
    config: GemmaConfig
    
    @nn.compact
    def __call__(self, x: jax.Array):
        """
        Transformer matching Gemma checkpoint structure.
        
        Args:
            x: Input token IDs of shape [B, L]
            
        Returns:
            Logits of shape [B, L, vocab_size]
        """
        embedder = self.param(
            'input_embedding', 
            nn.initializers.normal(stddev=0.02),
            (self.config.embedding_config.num_embeddings, self.config.embedding_config.features)
        )
        
        # Lookup embeddings and scale by sqrt(embed_dim) as per Gemma spec
        x = embedder[x]
        x = x * jnp.sqrt(self.config.embedding_config.features).astype(x.dtype)
        
        for i in range(self.config.num_layers):
            layer = TransformerBlock(
                config=self.config.transformer_block_config,
                name=f'layer_{i}'
            )
            x = layer(x)
        
        x = RMSNorm(name='final_norm')(x)
        
        logits = jnp.einsum('bld,vd->blv', x, embedder)
        
        return logits

class TransformerBlock(nn.Module):
    config: TransformerBlockConfig
    
    @nn.compact
    def __call__(self, x: jax.Array):
        """
        Transformer block matching Gemma checkpoint structure.
        
        Args:
            x: Input tensor of shape [B, L, embed_dim]
            
        Returns:
            Output tensor of shape [B, L, embed_dim]
        """
        batch_size, seq_len = x.shape[:2]
        
        positions = jnp.arange(seq_len)[jnp.newaxis, :].repeat(batch_size, axis=0)
        mask = create_causal_mask(seq_len)
        
        residual = x
        x = RMSNorm(name='pre_attention_norm')(x)
        
        attn = Attention(attn_config=self.config.attn_config, name='attn')
        attn_output = attn(x, positions, mask)
        
        attn_output = RMSNorm(name='post_attention_norm')(attn_output)
        x = residual + attn_output
        
        residual = x
        x = RMSNorm(name='pre_ffw_norm')(x)
        
        geglu = GeGLU(
            ffn_hidden_dim=self.config.ffn_hidden_dim,
            embed_dim=self.config.embed_dim,
            name='mlp'
        )
        ffn_output = geglu(x)
        
        ffn_output = RMSNorm(name='post_ffw_norm')(ffn_output)
        x = residual + ffn_output
        
        return x
   
class Attention(nn.Module):
    attn_config: AttentionConfig
    
    @nn.compact
    def __call__(self, x: jax.Array, positions: jax.Array, mask: jax.Array = None):
        """
        Full attention module matching Gemma checkpoint structure.
        
        Args:
            x: Input tensor of shape [B, L, embed_dim]  (already normalized by TransformerBlock)
            positions: Position indices of shape [B, L]
            mask: Optional attention mask of shape [B, L, L]
            
        Returns:
            Output tensor of shape [B, L, embed_dim]
        """
        _, _, embed_dim = x.shape
        num_q_heads = self.attn_config.num_query_heads
        num_kv_heads = self.attn_config.num_kv_heads
        head_dim = self.attn_config.head_dim
        
        q_einsum = self.param('q_einsum', nn.initializers.xavier_normal(), (num_q_heads, embed_dim, head_dim))
        kv_einsum = self.param('kv_einsum', nn.initializers.xavier_normal(), (2, num_kv_heads, embed_dim, head_dim))
        
        q = jnp.einsum('bld,hde->blhe', x, q_einsum)
        kv = jnp.einsum('bld,khde->kblhe', x, kv_einsum)
        k, v = kv[0], kv[1]
        
        q = RMSNorm(name='_query_norm')(q)
        k = RMSNorm(name='_key_norm')(k)
        
        q = apply_rope(
            q, 
            positions,
            base_frequency=self.attn_config.rope_base_frequency,
            scale_factor=self.attn_config.rope_scale_factor,
            rope_proportion=self.attn_config.rope_proportion
        )
        
        k = apply_rope(
            k,
            positions,
            base_frequency=self.attn_config.rope_base_frequency,
            scale_factor=self.attn_config.rope_scale_factor,
            rope_proportion=self.attn_config.rope_proportion
        )
        
        q = q * (head_dim ** -0.5)
        
        num_groups = num_q_heads // num_kv_heads
        k = jnp.repeat(k, repeats=num_groups, axis=2)
        v = jnp.repeat(v, repeats=num_groups, axis=2)
        
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        
        attn_scores = jnp.einsum('bhqd,bhkd->bhqk', q, k)
        if mask is not None:
            attn_scores = jnp.where(mask, attn_scores, -1e10)
        
        attn_weights = jax.nn.softmax(attn_scores, axis=-1)
        attn_output = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, v)
        
        attn_vec_einsum = self.param('attn_vec_einsum', nn.initializers.xavier_normal(), (num_q_heads, head_dim, embed_dim))
        output = jnp.einsum('blhd,hde->ble', jnp.transpose(attn_output, (0, 2, 1, 3)), attn_vec_einsum)
        
        return output

class EinsumLinear(nn.Module):
    shape: tuple[int, ...] # shape of the weight matrix as (in_features, out_features)
    use_bias: bool = True
    initializer: Initializer = xavier_normal()
    bias_initializer: Initializer = zeros_init()
    dtype: jax.typing.DTypeLike = None
   
    @nn.compact
    def __call__(self, x, eqn: str):
        w = self.param('w', init_fn = self.initializer, shape = (self.shape[0], self.shape[1]), dtype = self.dtype if self.dtype is not None else x.dtype)
        if self.use_bias:
            b = self.param('b', init_fn = self.bias_initializer, shape = (self.shape[1],), dtype = self.dtype if self.dtype is not None else x.dtype)
        out = jnp.einsum(eqn, x, w)
        if self.use_bias:
            out = out + b
        return out.astype(x.dtype)

class RMSNorm(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Get the feature dimension as a concrete value
        features = x.shape[-1]
        scale = self.param('scale', nn.initializers.zeros_init(), (features,))
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        normed_inputs = x * jax.lax.rsqrt(var + 1e-06)
        scale = jnp.expand_dims(scale, axis=range(len(x.shape) - 1))
        normed_inputs = normed_inputs * (1 + scale)
        return normed_inputs

class GeGLU(nn.Module):
    ffn_hidden_dim: int
    embed_dim: int
    
    @nn.compact
    def __call__(self, x: jax.Array):
        """
        GeGLU matching Gemma checkpoint structure.
        
        Args:
            x: Input tensor of shape [B, L, embed_dim]
            
        Returns:
            Output tensor of shape [B, L, embed_dim]
        """
        gating_einsum = self.param('gating_einsum', nn.initializers.xavier_normal(), (2, self.ffn_hidden_dim, self.embed_dim))
        
        gate_value = jnp.einsum('bld,hed->hble', x, gating_einsum)
        gate, value = gate_value[0], gate_value[1]
        
        gated = jax.nn.gelu(gate) * value
        
        linear = self.param('linear', nn.initializers.xavier_normal(), (self.ffn_hidden_dim, self.embed_dim))
        output = jnp.einsum('ble,ed->bld', gated, linear)
        
        return output       
        
if __name__ == "__main__": # for testing
    # Gemma-270M parameters from checkpoint
    vocab_size = 262144
    embed_dim = 640
    num_q_heads = 4
    num_kv_heads = 1
    head_dim = 256
    ffn_hidden_dim = 2048
    num_layers = 18
    
    gemma = Gemma(config = GemmaConfig(
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
    ))
   
    x = jnp.ones((2, 10), dtype = jnp.int32)
    print(gemma.tabulate(jax.random.key(0), x))