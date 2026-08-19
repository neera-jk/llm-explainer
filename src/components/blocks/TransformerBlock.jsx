import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { getFamily, isMoE, getFFNLabel } from '../../data/models'
import AttentionDetail from './AttentionDetail'
import FFNDetail from './FFNDetail'
import MatmulViz from './MatmulViz'

function FlowArrow({ color }) {
    return (
        <div className="lf-arrow">
            <svg width="22" height="10" viewBox="0 0 22 10">
                <line x1="0" y1="5" x2="15" y2="5" stroke="var(--border-soft)" strokeWidth="1" />
                <line x1="0" y1="5" x2="15" y2="5" className={`lf-pulse ${color}`} strokeWidth="1.5" />
                <polygon points="14,2 22,5 14,8" fill="var(--border-soft)" />
            </svg>
        </div>
    )
}

function PlainArrow() {
    return (
        <div className="lf-arrow">
            <svg width="22" height="10" viewBox="0 0 22 10">
                <line x1="0" y1="5" x2="15" y2="5" stroke="var(--border-soft)" strokeWidth="1" />
                <polygon points="14,2 22,5 14,8" fill="var(--border-soft)" />
            </svg>
        </div>
    )
}

function ArchArrow({ color }) {
    return (
        <div className="arch-arrow">
            <svg width="18" height="10" viewBox="0 0 18 10">
                <line x1="0" y1="5" x2="12" y2="5" stroke="var(--border-soft)" strokeWidth="1" />
                <line x1="0" y1="5" x2="12" y2="5" className={`lf-pulse ${color}`} strokeWidth="1.5" />
                <polygon points="12,2 18,5 12,8" fill="var(--border-soft)" />
            </svg>
        </div>
    )
}

function ArchBrace() {
    return (
        <div className="arch-brace">
            <svg width="14" height="60" viewBox="0 0 14 60">
                <path d="M0,0 C12,0 14,8 14,15 L14,22" className="arch-brace-path" />
                <path d="M0,60 C12,60 14,52 14,45 L14,38" className="arch-brace-path" />
                <path d="M0,30 L14,30" className="arch-brace-path" />
            </svg>
        </div>
    )
}

function AddCircle({ tipText }) {
    return (
        <div className="lf-add">
            +
            <span className="lf-add-tip">{tipText}</span>
        </div>
    )
}

function AttentionArch({ model, fam, T }) {
    const isGQA = fam.attn === 'GQA'
    const isMLA = fam.attn === 'MLA'
    const isIRoPE = fam.posEncoding === 'irope'
    const hasRoPE = fam.posEncoding === 'rope' || isIRoPE
    const hasQKNorm = fam.qkNorm === true
    const hasSWA = (model.sliding_window || 0) > 0
    const kvHeads = model.n_kv_heads || model.n_heads
    const groupSize = isGQA ? model.n_heads / kvHeads : 1

    return (
        <div className="arch-body">
            <div className="arch-row">
                <div className="arch-node arch-node--purple">
                    <span className="arch-node-title">input</span>
                    <span className="arch-node-sub">[{T}x{model.d_model}]</span>
                </div>
                <ArchArrow color="lf-pulse--purple" />
                <div className="arch-node">
                    <span className="arch-node-title">W_Q, W_K, W_V</span>
                    <span className="arch-node-sub">project</span>
                </div>
                <ArchBrace />
                <div className="arch-fan">
                    <div className="arch-fan-item arch-fan-item--q">Q <span className="arch-fan-dim">{model.n_heads} x {model.head_dim}d</span></div>
                    <div className="arch-fan-item arch-fan-item--k">K <span className="arch-fan-dim">{kvHeads} x {model.head_dim}d</span></div>
                    <div className="arch-fan-item arch-fan-item--v">V <span className="arch-fan-dim">{kvHeads} x {model.head_dim}d</span></div>
                </div>
                {hasQKNorm && (
                    <>
                        <ArchArrow color="lf-pulse--purple" />
                        <div className="arch-node" style={{ borderColor: 'rgba(42,157,143,0.35)', background: 'rgba(42,157,143,0.08)' }}>
                            <span className="arch-node-title" style={{ color: '#2A9D8F' }}>QK-Norm</span>
                            <span className="arch-node-sub">RMSNorm Q, K</span>
                        </div>
                    </>
                )}
                {hasRoPE && (
                    <>
                        <ArchArrow color="lf-pulse--purple" />
                        <div className="arch-node" style={{ borderColor: 'rgba(26,138,101,0.3)', background: 'rgba(26,138,101,0.06)' }}>
                            <span className="arch-node-title" style={{ color: '#1a8a65' }}>{isIRoPE ? 'iRoPE' : 'RoPE'}</span>
                            <span className="arch-node-sub">rotate Q, K</span>
                        </div>
                    </>
                )}
                <ArchArrow color="lf-pulse--purple" />
                <div className="arch-node arch-node--amber">
                    <span className="arch-node-title">Q x K^T</span>
                    <span className="arch-node-sub">scores [{T}x{T}]</span>
                </div>
                <ArchArrow color="lf-pulse--purple" />
                <div className="arch-node">
                    <span className="arch-node-title">scale + softmax</span>
                    <span className="arch-node-sub">div sqrt({model.head_dim})</span>
                </div>
                <ArchArrow color="lf-pulse--purple" />
                <div className="arch-node arch-node--amber">
                    <span className="arch-node-title">wt x V</span>
                    <span className="arch-node-sub">context</span>
                </div>
                <ArchArrow color="lf-pulse--purple" />
                <div className="arch-node">
                    <span className="arch-node-title">concat</span>
                    <span className="arch-node-sub">[{T}x{model.d_model}]</span>
                </div>
                <ArchArrow color="lf-pulse--purple" />
                <div className="arch-node arch-node--purple">
                    <span className="arch-node-title">W_O</span>
                    <span className="arch-node-sub">output</span>
                </div>
            </div>
            <div className="arch-pills">
                {isGQA && <span className="arch-pill arch-pill--green">{groupSize} Q heads share each K/V pair</span>}
                {isMLA && <span className="arch-pill arch-pill--green">latent KV compression</span>}
                {hasRoPE && <span className="arch-pill arch-pill--green">{isIRoPE ? 'iRoPE' : 'RoPE'} encodes position via rotation</span>}
                {!hasRoPE && <span className="arch-pill arch-pill--purple">position from learned embeddings</span>}
                {hasQKNorm && <span className="arch-pill arch-pill--green">QK-Norm stabilizes training</span>}
                {isIRoPE && <span className="arch-pill arch-pill--green">iRoPE: RoPE + NoPE layers interleaved</span>}
                {hasSWA && <span className="arch-pill arch-pill--amber">sliding window: {model.sliding_window} tokens</span>}
                <span className="arch-pill arch-pill--purple">{model.n_heads} Q heads, {kvHeads} KV heads</span>
                <span className="arch-pill arch-pill--amber">KV cache: {kvHeads} x {model.head_dim}d per layer</span>
            </div>
            <p className="arch-hint">
                {isGQA
                    ? `Grouped Query Attention: ${model.n_heads} query heads grouped into ${kvHeads} groups of ${groupSize}. Each group shares one K and one V, making the KV cache ${groupSize}x smaller than standard MHA.`
                    : isMLA
                        ? 'Multi-head Latent Attention: K and V are compressed into a low-dimensional latent space before caching, dramatically reducing memory for long contexts.'
                        : `Multi-Head Attention: all ${model.n_heads} heads have their own Q, K, and V. Simple but the KV cache stores ${model.n_heads} separate K and V per layer.`
                }
                {hasQKNorm ? ' QK-Norm applies RMSNorm to the Q and K vectors after projection but before rotation, preventing attention scores from exploding as the model scales up.' : ''}
                {isIRoPE ? ' iRoPE interleaves RoPE layers (with chunked local attention) and NoPE layers (no positional encoding, full causal mask), which is key to reaching extreme context lengths.' : ''}
                {hasSWA ? ` Sliding window attention limits each token to the previous ${model.sliding_window} tokens instead of the full sequence, cutting memory from quadratic to linear in sequence length.` : ''}
            </p>
        </div>
    )
}

function FFNArch({ model, fam, T }) {
    const isSwiGLU = fam.ffn === 'SwiGLU'
    const isMoEModel = isMoE(model)
    const hasSharedExperts = (model.n_shared_experts || 0) > 0
    const isAlternating = model.alternating_moe === true
    const ratio = (model.d_ff / model.d_model).toFixed(1)

    return (
        <div className="arch-body">
            <div className="arch-row">
                <div className="arch-node arch-node--purple" style={{ height: 52 }}>
                    <span className="arch-node-title">input</span>
                    <span className="arch-node-sub">[{T}x{model.d_model}]</span>
                </div>
                <ArchArrow color="lf-pulse--amber" />
                {isMoEModel && (
                    <>
                        <div className="arch-node" style={{ height: 52, borderColor: 'rgba(26,138,101,0.3)', background: 'rgba(26,138,101,0.06)' }}>
                            <span className="arch-node-title" style={{ color: '#1a8a65' }}>router</span>
                            <span className="arch-node-sub">top-{model.n_active_experts} of {model.n_experts}{hasSharedExperts ? ' + shared' : ''}</span>
                        </div>
                        <ArchArrow color="lf-pulse--amber" />
                    </>
                )}
                {isSwiGLU ? (
                    <>
                        <div className="arch-fan">
                            <div className="arch-fan-item" style={{ background: 'rgba(184,84,40,0.08)', border: '1px solid rgba(184,84,40,0.25)', color: '#b85428' }}>
                                W_gate <span className="arch-fan-dim">[{model.d_model}x{model.d_ff}]</span>
                            </div>
                            <div className="arch-fan-item" style={{ background: 'rgba(127,119,221,0.08)', border: '1px solid rgba(127,119,221,0.25)', color: '#7F77DD' }}>
                                W_up <span className="arch-fan-dim">[{model.d_model}x{model.d_ff}]</span>
                            </div>
                        </div>
                        <ArchArrow color="lf-pulse--amber" />
                        <div className="arch-node arch-node--amber" style={{ height: 64 }}>
                            <span className="arch-node-title">swish(gate) x up</span>
                            <span className="arch-node-sub">[{T}x{model.d_ff}]</span>
                            <span className="arch-expand-note">{ratio}x wider</span>
                        </div>
                    </>
                ) : (
                    <>
                        <div className="arch-node" style={{ height: 52 }}>
                            <span className="arch-node-title">W_up</span>
                            <span className="arch-node-sub">[{model.d_model}x{model.d_ff}]</span>
                        </div>
                        <ArchArrow color="lf-pulse--amber" />
                        <div className="arch-node arch-node--amber arch-node--tall">
                            <span className="arch-node-title">GELU</span>
                            <span className="arch-node-sub">[{T}x{model.d_ff}]</span>
                            <span className="arch-expand-note">{ratio}x wider</span>
                        </div>
                    </>
                )}
                <ArchArrow color="lf-pulse--amber" />
                <div className="arch-node" style={{ height: 52 }}>
                    <span className="arch-node-title">W_down</span>
                    <span className="arch-node-sub">[{model.d_ff}x{model.d_model}]</span>
                </div>
                <ArchArrow color="lf-pulse--amber" />
                <div className="arch-node arch-node--purple" style={{ height: 52 }}>
                    <span className="arch-node-title">output</span>
                    <span className="arch-node-sub">[{T}x{model.d_model}]</span>
                </div>
            </div>
            <div className="arch-pills">
                {isSwiGLU && <span className="arch-pill arch-pill--amber">SwiGLU: 3 weight matrices (gate, up, down)</span>}
                {!isSwiGLU && <span className="arch-pill arch-pill--amber">GELU: 2 weight matrices (up, down)</span>}
                {isMoEModel && <span className="arch-pill arch-pill--green">only {model.n_active_experts} of {model.n_experts} experts active per token</span>}
                {hasSharedExperts && <span className="arch-pill arch-pill--green">+{model.n_shared_experts} shared expert always active</span>}
                {isAlternating && <span className="arch-pill arch-pill--amber">dense and MoE layers alternate</span>}
                {(model.n_dense_layers || 0) > 0 && <span className="arch-pill arch-pill--amber">first {model.n_dense_layers} layers are dense</span>}
            </div>
            <p className="arch-hint">
                {isSwiGLU
                    ? `SwiGLU uses three weight matrices instead of two. W_gate and W_up project in parallel, then the gate output (after swish activation) controls which features from W_up pass through.`
                    : `Standard FFN: expand to ${model.d_ff} dimensions with W_up, apply GELU activation, then compress back to ${model.d_model} with W_down.`
                }
                {isMoEModel ? ` With MoE, each token is routed to only ${model.n_active_experts} of ${model.n_experts} available expert FFNs. Massive total capacity, but only a fraction runs per token.` : ''}
                {hasSharedExperts ? ` ${model.n_shared_experts} shared expert also runs on every token, giving a stable baseline on top of the routed experts.` : ''}
                {isAlternating ? ' Dense and MoE layers alternate through the stack instead of using MoE everywhere, which helps stabilize training.' : ''}
                {(model.n_dense_layers || 0) > 0 ? ` The first ${model.n_dense_layers} layers stay fully dense before MoE layers begin.` : ''}
            </p>
        </div>
    )
}

export default function TransformerBlock({ model, tokens, layerIndex, family }) {
    const [activeSubBlock, setActiveSubBlock] = useState(null)
    const [activeTab, setActiveTab] = useState('main')
    const T = tokens.length
    const fam = family || getFamily(model)
    const isMoEModel = isMoE(model)
    const isIRoPE = fam.posEncoding === 'irope'
    const hasRoPE = fam.posEncoding === 'rope' || isIRoPE
    const hasSWA = (model.sliding_window || 0) > 0
    const hasQKNorm = fam.qkNorm === true
    const hasSharedExperts = (model.n_shared_experts || 0) > 0
    const isAlternating = model.alternating_moe === true

    function handleSubClick(block) {
        if (activeSubBlock === block) {
            setActiveSubBlock(null)
            setActiveTab('main')
        } else {
            setActiveSubBlock(block)
            setActiveTab('main')
        }
    }

    const attnLabel = fam.attn === 'MHA' ? 'self-attention' : fam.attn
    const attnSub = fam.attn === 'GQA'
        ? `${model.n_heads}Q / ${model.n_kv_heads}KV heads`
        : fam.attn === 'MLA'
            ? `${model.n_heads} heads, latent KV`
            : `${model.n_heads} heads x ${model.head_dim}d`

    const ffnLabel = getFFNLabel(model, fam)
    const ffnSub = isMoEModel
        ? `top-${model.n_active_experts} of ${model.n_experts} experts`
        : `${model.d_model} to ${model.d_ff} to ${model.d_model}`

    return (
        <div className="layer-detail">
            <p className="layer-detail-intro">
                <strong>Layer {layerIndex}</strong> of {model.n_layers}. Data flows left to right.
                First, the input is normalized using <strong>{fam.norm}</strong>
                {fam.norm === 'RMSNorm' ? ' (a simpler, faster alternative to LayerNorm that skips mean subtraction)' : ' (stabilizes values by subtracting the mean and dividing by standard deviation)'}.
                Then <strong>{attnLabel}</strong> lets every token look at every other token to gather context.
                {fam.attn === 'GQA' ? ` In GQA, groups of ${model.n_heads / model.n_kv_heads} query heads share the same key and value, making the KV cache ${model.n_heads / model.n_kv_heads}x smaller.` : ''}
                {hasQKNorm ? ' QK-Norm applies RMSNorm to the query and key vectors before rotation, keeping attention scores stable at large scale.' : ''}
                {hasRoPE && !isIRoPE ? ' RoPE (Rotary Position Embeddings) is applied inside attention to encode where each token sits in the sequence.' : ''}
                {isIRoPE ? ' iRoPE interleaves RoPE layers with NoPE layers (no positional encoding) inside attention, which unlocks extreme context lengths.' : ''}
                {hasSWA ? ` Sliding window attention limits each token to the previous ${model.sliding_window} tokens, keeping memory linear in sequence length.` : ''}
                {' '}The highlighted zones show <strong>residual connections</strong>: the original input
                is added back to the sub-block's output at the "+" circle. This helps the model train
                by giving gradients a shortcut through the network.
                Then another {fam.norm}, followed by the <strong>{ffnLabel}</strong>
                {fam.ffn === 'SwiGLU' && !isMoEModel ? ' which uses a gating mechanism with three weight matrices' : ''}
                {isMoEModel ? ` where a router sends each token to ${model.n_active_experts} of ${model.n_experts} expert networks` : ''}.
                {hasSharedExperts ? ` A shared expert also runs on every token regardless of routing, providing a stable baseline.` : ''}
                {isAlternating ? ' In this model, dense and MoE layers alternate through the stack to stabilize training.' : ''}
                {(model.n_dense_layers || 0) > 0 ? ` The first ${model.n_dense_layers} layers are fully dense before MoE begins.` : ''}
                {' '}Another residual add, and the output has the same shape: <strong>[{T} x {model.d_model}]</strong>.
            </p>

            <div className="layer-flow-scroll">
                <div className="layer-flow-row">
                    <div className="lf-node lf-node--io">input<sub>[{T}x{model.d_model}]</sub></div>
                    <FlowArrow color="lf-pulse--purple" />
                    <div className="lf-node">{fam.norm}<sub>normalize</sub></div>
                    <FlowArrow color="lf-pulse--purple" />

                    <div className="lf-residual-zone lf-residual-zone--attn">
                        <div style={{ position: 'relative' }}>
                            {hasRoPE && <span className="lf-rope-badge">+ {isIRoPE ? 'iRoPE' : 'RoPE'}</span>}
                            {hasSWA && (
                                <span className="lf-rope-badge"
                                    style={{ color: '#BA7517', background: 'rgba(186,117,23,0.08)', borderColor: 'rgba(186,117,23,0.2)', top: hasRoPE ? '-26px' : '-12px' }}>
                                    sliding window
                                </span>
                            )}
                            <div
                                className={`lf-node lf-node--attn ${activeSubBlock === 'attn' ? 'lf-node--active' : ''}`}
                                onClick={() => handleSubClick('attn')}
                            >
                                {attnLabel}<sub>{attnSub}</sub>
                            </div>
                        </div>
                        <PlainArrow />
                        <AddCircle tipText="adds the original input back (residual connection)" />
                    </div>

                    <FlowArrow color="lf-pulse--amber" />
                    <div className="lf-node">{fam.norm}<sub>normalize</sub></div>
                    <FlowArrow color="lf-pulse--amber" />

                    <div className="lf-residual-zone lf-residual-zone--ffn">
                        <div
                            className={`lf-node lf-node--ffn ${activeSubBlock === 'ffn' ? 'lf-node--active' : ''}`}
                            onClick={() => handleSubClick('ffn')}
                        >
                            {ffnLabel}<sub>{ffnSub}</sub>
                        </div>
                        <PlainArrow />
                        <AddCircle tipText="adds the original input back (residual connection)" />
                    </div>

                    <PlainArrow />
                    <div className="lf-node lf-node--io">output<sub>[{T}x{model.d_model}]</sub></div>
                </div>
            </div>
            <p className="layer-flow-hint">data flows left to right. the shaded zones are residual connections. hover the "+" to learn more. click the colored blocks to see inside.</p>

            <AnimatePresence mode="wait">
                {activeSubBlock && (
                    <motion.div key={activeSubBlock}
                        initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}>
                        <div className="arch-panel">
                            <div className="arch-header">
                                <span className="arch-title" style={{ color: activeSubBlock === 'attn' ? '#AFA9EC' : '#EF9F27' }}>
                                    <span className="arch-title-dot" style={{ background: activeSubBlock === 'attn' ? '#7F77DD' : '#BA7517' }} />
                                    {activeSubBlock === 'attn' ? `${attnLabel} architecture` : `${ffnLabel} architecture`}
                                </span>
                                <button className="arch-close" onClick={() => { setActiveSubBlock(null); setActiveTab('main') }}>close</button>
                            </div>
                            {activeSubBlock === 'attn' && <AttentionArch model={model} fam={fam} T={T} />}
                            {activeSubBlock === 'ffn' && <FFNArch model={model} fam={fam} T={T} />}
                        </div>

                        <div className="sub-detail">
                            <div className="sub-detail-header">
                                <span className="sub-detail-title" style={{ color: activeSubBlock === 'attn' ? '#AFA9EC' : '#EF9F27' }}>
                                    step-through
                                </span>
                                <div className="sub-detail-tabs">
                                    <button className={`sub-tab ${activeTab === 'main' ? 'sub-tab-on' : ''}`} onClick={() => setActiveTab('main')}>
                                        {activeSubBlock === 'attn' ? 'attention' : 'FFN'}
                                    </button>
                                    <button className={`sub-tab ${activeTab === 'matmul' ? 'sub-tab-on' : ''}`} onClick={() => setActiveTab('matmul')}>
                                        matmul
                                    </button>
                                </div>
                            </div>
                            <AnimatePresence mode="wait">
                                <motion.div key={activeTab}
                                    initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
                                    exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.2 }}>
                                    {activeTab === 'main' && activeSubBlock === 'attn' && <AttentionDetail T={T} model={model} family={fam} tokens={tokens} />}
                                    {activeTab === 'main' && activeSubBlock === 'ffn' && <FFNDetail T={T} model={model} family={fam} />}
                                    {activeTab === 'matmul' && <MatmulViz />}
                                </motion.div>
                            </AnimatePresence>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    )
}