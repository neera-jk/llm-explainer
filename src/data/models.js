// Each model lives in ./models/<key>.json. Everything except key, label, family
// and params is generated from the model's HF config by scripts/import-hf.mjs.
const modelFiles = import.meta.glob('./models/*.json', { eager: true, import: 'default' })

// A family is only the dropdown group (and prompt tokenizer). Architecture
// features come from each model's own config.
export const MODEL_FAMILIES = {
    gpt2: { key: 'gpt2', label: 'GPT-2 family' },
    llama: { key: 'llama', label: 'Llama 3 family' },
    mixtral: { key: 'mixtral', label: 'Mixtral family' },
    deepseek: { key: 'deepseek', label: 'DeepSeek family' },
    qwen: { key: 'qwen', label: 'Qwen 3 family' },
    llama4: { key: 'llama4', label: 'Llama 4 family' },
}

export const MODEL_REGISTRY = Object.fromEntries(
    Object.values(modelFiles).map(m => [m.key, { ...m, desc: describe(m) }])
)

// rough size, only used to list smaller models first within a family
function approxSize(m) {
    return m.n_layers * m.d_model ** 2 * Math.max(1, m.n_experts)
}

export const MODEL_GROUPS = Object.values(MODEL_FAMILIES)
    .map(f => ({
        label: f.label,
        family: f.key,
        models: Object.values(MODEL_REGISTRY)
            .filter(m => m.family === f.key)
            .sort((a, b) => approxSize(a) - approxSize(b))
            .map(m => m.key),
    }))
    .filter(g => g.models.length > 0)

export const DEFAULT_MODEL = 'gpt2-small'

export function getFamily(model) {
    const family = MODEL_FAMILIES[model.family] || MODEL_FAMILIES.gpt2
    return { ...family, ...model.features }
}

export function isMoE(model) {
    return (model.n_experts || 0) > 1
}

// MoE diagrams draw a single expert, so they use the expert width
export function ffnWidth(model) {
    return isMoE(model) ? model.d_ff_expert : model.d_ff
}

// V is narrower than Q/K in MLA models
export function vHeadDim(model) {
    return model.mla ? model.mla.v_head_dim : model.head_dim
}

function describe(m) {
    const f = m.features
    const parts = [isMoE(m) ? 'MoE' : 'decoder-only', `${m.n_layers} layers`, `${m.d_model} dims`]
    if (f.attn !== 'MHA') parts.push(f.attn)
    if (f.posEncoding === 'rope') parts.push('RoPE')
    if (f.posEncoding === 'irope') parts.push('iRoPE')
    if (f.qkNorm) parts.push('QK-Norm')
    if (m.sliding_window) parts.push('SWA')
    if (isMoE(m)) {
        const shared = m.n_shared_experts ? `+${m.n_shared_experts}` : ''
        parts.push(`${m.n_experts}${shared} experts top-${m.n_active_experts}`)
    } else if (f.ffn !== 'GELU') {
        parts.push(f.ffn)
    }
    return parts.join(', ')
}

export function getFFNLabel(model, family) {
    if (isMoE(model)) {
        return family.ffn + ' + MoE'
    }
    return family.ffn
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
