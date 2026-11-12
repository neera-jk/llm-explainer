const HF = 'https://huggingface.co'

export class UnknownArchitectureError extends Error {
    constructor(config) {
        super(`no adapter for model_type "${config.model_type}" (${(config.architectures || []).join(', ')})`)
        this.modelType = config.model_type
    }
}

async function request(url, token) {
    const res = await fetch(url, { headers: token ? { Authorization: `Bearer ${token}` } : {} })
    if (!res.ok) {
        const err = new Error(`${res.status} ${res.statusText} for ${url}`)
        err.status = res.status
        throw err
    }
    return res
}

async function getJSON(url, token) {
    return (await request(url, token)).json()
}

export function getModelInfo(repo, token = process.env.HF_TOKEN) {
    return getJSON(`${HF}/api/models/${repo}`, token)
}

// just the fields the watcher needs to filter uploads
const LIST_FIELDS = ['createdAt', 'pipeline_tag', 'gated', 'tags', 'config', 'safetensors', 'likes']
const expandFields = () => LIST_FIELDS.map(f => `expand[]=${f}`).join('&')

// what the whole HF community is looking at right now, for one kind of model
export function listTrending(pipeline, { limit = 100, token = process.env.HF_TOKEN } = {}) {
    return getJSON(`${HF}/api/models?sort=trendingScore&direction=-1&limit=${limit}&filter=${encodeURIComponent(pipeline)}&${expandFields()}`, token)
}

// An org's uploads since a date, newest first. Busy orgs can upload more than
// one page in a month, so keep following the next-page link until we pass `since`.
export async function listModelsSince(org, since, { token = process.env.HF_TOKEN } = {}) {
    let url = `${HF}/api/models?author=${encodeURIComponent(org)}&sort=createdAt&direction=-1&limit=100&${expandFields()}`
    const models = []
    while (url) {
        const res = await request(url, token)
        for (const m of await res.json()) {
            if (new Date(m.createdAt) < since) return models
            models.push(m)
        }
        url = res.headers.get('link')?.match(/<([^>]+)>;\s*rel="next"/)?.[1]
    }
    return models
}

async function fetchPinned(repo, token) {
    // the model info endpoint works for gated repos too, so we always get the official sha
    const info = await getJSON(`${HF}/api/models/${repo}`, token)
    const config = await getJSON(`${HF}/${repo}/resolve/${info.sha}/config.json`, token)
    return { config, revision: info.sha }
}

// Gated repos (Llama) need HF_TOKEN. Without one we fall back to an ungated mirror
// and record both, so provenance still points at the official repo.
export async function fetchConfig(repo, { mirror, token = process.env.HF_TOKEN } = {}) {
    try {
        const { config, revision } = await fetchPinned(repo, token)
        return { config, source: { repo, revision } }
    } catch (err) {
        if (!mirror || ![401, 403].includes(err.status)) throw err
        const info = await getJSON(`${HF}/api/models/${repo}`, token)
        const pinned = await fetchPinned(mirror, token)
        return {
            config: pinned.config,
            source: { repo, revision: info.sha, config_from: mirror, config_revision: pinned.revision },
        }
    }
}

const DENSE = {
    d_ff_expert: null,
    n_experts: 0,
    n_active_experts: 0,
    n_shared_experts: 0,
    n_dense_layers: 0,
    alternating_moe: false,
}

function slidingWindow(c) {
    // Qwen configs ship sliding_window: 4096 with use_sliding_window: false
    if (c.use_sliding_window === false) return 0
    return c.sliding_window || 0
}

function gluName(act) {
    if (act === 'silu' || act === 'swish') return 'SwiGLU'
    if (act && act.startsWith('gelu')) return 'GeGLU'
    return `GLU(${act})`
}

function llamaLike(c) {
    return {
        d_model: c.hidden_size,
        n_layers: c.num_hidden_layers,
        n_heads: c.num_attention_heads,
        n_kv_heads: c.num_key_value_heads ?? c.num_attention_heads,
        head_dim: c.head_dim ?? c.hidden_size / c.num_attention_heads,
        vocab_size: c.vocab_size,
        context_length: c.max_position_embeddings,
        d_ff: c.intermediate_size,
        sliding_window: slidingWindow(c),
        ...DENSE,
        features: {
            norm: 'RMSNorm',
            ffn: gluName(c.hidden_act),
            posEncoding: 'rope',
            hasBias: !!(c.attention_bias || c.mlp_bias),
            tiedEmbeddings: c.tie_word_embeddings ?? false,
            qkNorm: false,
        },
    }
}

function deepseekLike(c) {
    const base = llamaLike(c)
    return {
        ...base,
        // MLA scores use the full qk dim (nope + rope part) for the sqrt scaling
        head_dim: c.qk_nope_head_dim + c.qk_rope_head_dim,
        d_ff_expert: c.moe_intermediate_size,
        n_experts: c.n_routed_experts,
        n_active_experts: c.num_experts_per_tok,
        n_shared_experts: c.n_shared_experts ?? 0,
        n_dense_layers: c.first_k_dense_replace ?? 0,
        alternating_moe: (c.moe_layer_freq ?? 1) > 1,
        mla: {
            kv_lora_rank: c.kv_lora_rank,
            q_lora_rank: c.q_lora_rank ?? null,
            qk_nope_head_dim: c.qk_nope_head_dim,
            qk_rope_head_dim: c.qk_rope_head_dim,
            v_head_dim: c.v_head_dim,
        },
    }
}

function qwen3(c) {
    const base = llamaLike(c)
    return { ...base, features: { ...base.features, qkNorm: true } }
}

const ADAPTERS = {
    gpt2: c => ({
        d_model: c.n_embd,
        n_layers: c.n_layer,
        n_heads: c.n_head,
        n_kv_heads: c.n_head,
        head_dim: c.n_embd / c.n_head,
        vocab_size: c.vocab_size,
        context_length: c.n_positions,
        d_ff: c.n_inner ?? 4 * c.n_embd,
        sliding_window: 0,
        ...DENSE,
        features: {
            norm: 'LayerNorm',
            ffn: 'GELU',
            posEncoding: 'learned',
            hasBias: true,
            tiedEmbeddings: c.tie_word_embeddings ?? true,
            qkNorm: false,
        },
    }),
    llama: llamaLike,
    mistral: llamaLike,
    mixtral: c => ({
        ...llamaLike(c),
        d_ff: null,
        d_ff_expert: c.intermediate_size,
        n_experts: c.num_local_experts,
        n_active_experts: c.num_experts_per_tok,
    }),
    qwen3,
    qwen3_moe: c => {
        const dense = c.mlp_only_layers || []
        return {
            ...qwen3(c),
            d_ff: dense.length ? c.intermediate_size : null,
            d_ff_expert: c.moe_intermediate_size,
            n_experts: c.num_experts,
            n_active_experts: c.num_experts_per_tok,
            n_dense_layers: dense.length,
            alternating_moe: (c.decoder_sparse_step ?? 1) > 1,
        }
    },
    deepseek_v2: deepseekLike,
    deepseek_v3: deepseekLike,
    kimi_k2: deepseekLike,
    llama4: outer => {
        const c = outer.text_config ?? outer
        const base = llamaLike(c)
        const alternating = (c.interleave_moe_layer_step ?? 1) > 1
        return {
            ...base,
            d_ff: alternating ? c.intermediate_size_mlp : null,
            d_ff_expert: c.intermediate_size,
            n_experts: c.num_local_experts,
            n_active_experts: c.num_experts_per_tok,
            // Llama 4 MoE layers always add one shared expert (hardcoded in the modeling code)
            n_shared_experts: 1,
            alternating_moe: alternating,
            attention_chunk_size: c.attention_chunk_size ?? 0,
            features: { ...base.features, posEncoding: 'irope', qkNorm: !!c.use_qk_norm },
        }
    },
}

function attnType(m) {
    if (m.mla) return 'MLA'
    if (m.n_kv_heads === m.n_heads) return 'MHA'
    if (m.n_kv_heads === 1) return 'MQA'
    return 'GQA'
}

export function normalize(config) {
    const adapter = ADAPTERS[config.model_type]
    if (!adapter) throw new UnknownArchitectureError(config)
    const { features, ...dims } = adapter(config)
    return {
        model_type: config.model_type,
        architectures: config.architectures || [],
        ...dims,
        features: { attn: attnType(dims), ...features },
    }
}

export function hasAdapter(modelType) {
    return Object.hasOwn(ADAPTERS, modelType)
}

// the generated part of a model file: where it came from plus the normalized config
export async function importModel(repo, mirror) {
    const { config, source } = await fetchConfig(repo, { mirror })
    return { source: { repo, ...(mirror && { mirror }), ...source }, ...normalize(config) }
}

export function isMoE(m) {
    return (m.n_experts || 0) > 1
}
