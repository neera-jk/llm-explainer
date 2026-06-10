export const MODEL_REGISTRY = {
    gpt2: {
        key: 'gpt2',
        label: 'GPT-2',
        desc: 'decoder-only · 12 layers · 768 dims · 117M params',
        d_model: 768,
        d_ff: 3072,
        n_heads: 12,
        n_layers: 12,
        head_dim: 64,
        vocab_size: 50257,
        pos_encoding: 'Learned absolute positional embeddings',
        norm_type: 'LayerNorm',
        attn_type: 'Multi-Head Attention (MHA)',
        ffn_activation: 'GELU',
    },
    llama3: {
        key: 'llama3',
        label: 'Llama 3',
        desc: 'decoder-only · 32 layers · 4096 dims · 8B params',
        d_model: 4096,
        d_ff: 14336,
        n_heads: 32,
        n_kv_heads: 8,
        n_layers: 32,
        head_dim: 128,
        vocab_size: 128256,
        pos_encoding: 'RoPE',
        norm_type: 'RMSNorm',
        attn_type: 'Grouped Query Attention (GQA)',
        ffn_activation: 'SwiGLU',
    },
    kimi: {
        key: 'kimi',
        label: 'Kimi K2',
        desc: 'MoE · 61 layers · 7168 dims · 1T params (32B active)',
        d_model: 7168,
        d_ff: 18432,
        n_heads: 64,
        n_layers: 61,
        head_dim: 112,
        vocab_size: 160000,
        n_experts: 384,
        n_active_experts: 8,
        pos_encoding: 'RoPE',
        norm_type: 'RMSNorm',
        attn_type: 'MLA (Multi-head Latent Attention)',
        ffn_activation: 'SwiGLU + MoE (384 experts, 8 active per token)',
    },
}

export const MODEL_KEYS = Object.keys(MODEL_REGISTRY)
export const DEFAULT_MODEL = 'gpt2'

export function makeEmbeddingVector(tokenId, dims) {
    const vec = []
    for (let i = 0; i < dims; i++) {
        const x = Math.sin(tokenId * 127.1 + i * 311.7) * 43758.5453
        vec.push(parseFloat(((x - Math.floor(x)) * 2 - 1).toFixed(3)))
    }
    return vec
}

export function softmax(logits, temperature = 1.0) {
    const scaled = logits.map(v => v / temperature)
    const max = Math.max(...scaled)
    const exps = scaled.map(v => Math.exp(v - max))
    const sum = exps.reduce((a, b) => a + b, 0)
    return exps.map(v => v / sum)
}