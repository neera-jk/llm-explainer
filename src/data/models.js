export const MODEL_FAMILIES = {
    gpt2: {
        key: 'gpt2',
        label: 'GPT-2 family',
        norm: 'LayerNorm',
        attn: 'MHA',
        ffn: 'GELU',
        posEncoding: 'learned',
        hasBias: true,
        tiedEmbeddings: true,
    },
    llama: {
        key: 'llama',
        label: 'Llama 3 family',
        norm: 'RMSNorm',
        attn: 'GQA',
        ffn: 'SwiGLU',
        posEncoding: 'rope',
        hasBias: false,
        tiedEmbeddings: false,
    },
    moe: {
        key: 'moe',
        label: 'MoE family',
        norm: 'RMSNorm',
        attn: 'MLA',
        ffn: 'SwiGLU + MoE',
        posEncoding: 'rope',
        hasBias: false,
        tiedEmbeddings: false,
    },
}

export const MODEL_REGISTRY = {
    'gpt2-small': {
        key: 'gpt2-small',
        label: 'GPT-2 small',
        family: 'gpt2',
        params: '117M',
        desc: 'decoder-only, 12 layers, 768 dims, 117M params',
        d_model: 768,
        d_ff: 3072,
        n_heads: 12,
        n_kv_heads: 12,
        n_layers: 12,
        head_dim: 64,
        vocab_size: 50257,
        context_length: 1024,
    },
    'gpt2-medium': {
        key: 'gpt2-medium',
        family: 'gpt2',
        label: 'GPT-2 medium',
        params: '345M',
        desc: 'decoder-only, 24 layers, 1024 dims, 345M params',
        d_model: 1024,
        d_ff: 4096,
        n_heads: 16,
        n_kv_heads: 16,
        n_layers: 24,
        head_dim: 64,
        vocab_size: 50257,
        context_length: 1024,
    },
    'gpt2-large': {
        key: 'gpt2-large',
        family: 'gpt2',
        label: 'GPT-2 large',
        params: '762M',
        desc: 'decoder-only, 36 layers, 1280 dims, 762M params',
        d_model: 1280,
        d_ff: 5120,
        n_heads: 20,
        n_kv_heads: 20,
        n_layers: 36,
        head_dim: 64,
        vocab_size: 50257,
        context_length: 1024,
    },
    'gpt2-xl': {
        key: 'gpt2-xl',
        family: 'gpt2',
        label: 'GPT-2 XL',
        params: '1.5B',
        desc: 'decoder-only, 48 layers, 1600 dims, 1.5B params',
        d_model: 1600,
        d_ff: 6400,
        n_heads: 25,
        n_kv_heads: 25,
        n_layers: 48,
        head_dim: 64,
        vocab_size: 50257,
        context_length: 1024,
    },
    'llama3-8b': {
        key: 'llama3-8b',
        family: 'llama',
        label: 'Llama 3 8B',
        params: '8B',
        desc: 'decoder-only, 32 layers, 4096 dims, GQA, RoPE, SwiGLU',
        d_model: 4096,
        d_ff: 14336,
        n_heads: 32,
        n_kv_heads: 8,
        n_layers: 32,
        head_dim: 128,
        vocab_size: 128256,
        context_length: 8192,
    },
    'llama3-70b': {
        key: 'llama3-70b',
        family: 'llama',
        label: 'Llama 3 70B',
        params: '70B',
        desc: 'decoder-only, 80 layers, 8192 dims, GQA, RoPE, SwiGLU',
        d_model: 8192,
        d_ff: 28672,
        n_heads: 64,
        n_kv_heads: 8,
        n_layers: 80,
        head_dim: 128,
        vocab_size: 128256,
        context_length: 8192,
    },
    'llama31-405b': {
        key: 'llama31-405b',
        family: 'llama',
        label: 'Llama 3.1 405B',
        params: '405B',
        desc: 'decoder-only, 126 layers, 16384 dims, GQA, RoPE, SwiGLU',
        d_model: 16384,
        d_ff: 53248,
        n_heads: 128,
        n_kv_heads: 8,
        n_layers: 126,
        head_dim: 128,
        vocab_size: 128256,
        context_length: 131072,
    },
    'kimi-k2': {
        key: 'kimi-k2',
        family: 'moe',
        label: 'Kimi K2',
        params: '1T (32B active)',
        desc: 'MoE, 61 layers, 7168 dims, MLA, 384 experts (8 active)',
        d_model: 7168,
        d_ff: 18432,
        n_heads: 64,
        n_kv_heads: 64,
        n_layers: 61,
        head_dim: 112,
        vocab_size: 160000,
        context_length: 131072,
        n_experts: 384,
        n_active_experts: 8,
    },
}

export const MODEL_GROUPS = [
    {
        label: 'GPT-2 family',
        family: 'gpt2',
        models: ['gpt2-small', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
    },
    {
        label: 'Llama 3 family',
        family: 'llama',
        models: ['llama3-8b', 'llama3-70b', 'llama31-405b'],
    },
    {
        label: 'MoE family',
        family: 'moe',
        models: ['kimi-k2'],
    },
]

export const DEFAULT_MODEL = 'gpt2-small'

export function getFamily(model) {
    return MODEL_FAMILIES[model.family] || MODEL_FAMILIES.gpt2
}

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