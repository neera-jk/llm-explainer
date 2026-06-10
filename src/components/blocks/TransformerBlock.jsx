import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
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

function AttentionArch({ model, T }) {
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
                    <span className="arch-node-sub">project [{model.d_model}x{model.d_model}]</span>
                </div>
                <ArchBrace />
                <div className="arch-fan">
                    <div className="arch-fan-item arch-fan-item--q">Q <span className="arch-fan-dim">[{T}x{model.head_dim}]</span></div>
                    <div className="arch-fan-item arch-fan-item--k">K <span className="arch-fan-dim">[{T}x{model.head_dim}]</span></div>
                    <div className="arch-fan-item arch-fan-item--v">V <span className="arch-fan-dim">[{T}x{model.head_dim}]</span></div>
                </div>
                <ArchArrow color="lf-pulse--purple" />
                <div className="arch-node arch-node--amber">
                    <span className="arch-node-title">Q x K^T</span>
                    <span className="arch-node-sub">scores [{T}x{T}]</span>
                </div>
                <ArchArrow color="lf-pulse--purple" />
                <div className="arch-node">
                    <span className="arch-node-title">scale + softmax</span>
                    <span className="arch-node-sub">div sqrt({model.head_dim}), weights</span>
                </div>
                <ArchArrow color="lf-pulse--purple" />
                <div className="arch-node arch-node--amber">
                    <span className="arch-node-title">wt x V</span>
                    <span className="arch-node-sub">context [{T}x{model.head_dim}]</span>
                </div>
                <ArchArrow color="lf-pulse--purple" />
                <div className="arch-node">
                    <span className="arch-node-title">concat {model.n_heads} heads</span>
                    <span className="arch-node-sub">[{T}x{model.d_model}]</span>
                </div>
                <ArchArrow color="lf-pulse--purple" />
                <div className="arch-node arch-node--purple">
                    <span className="arch-node-title">W_O</span>
                    <span className="arch-node-sub">output [{T}x{model.d_model}]</span>
                </div>
            </div>
            <div className="arch-pills">
                <span className="arch-pill arch-pill--green">K and V cached for inference</span>
                <span className="arch-pill arch-pill--purple">x{model.n_heads} heads in parallel</span>
                <span className="arch-pill arch-pill--amber">matmul operations</span>
            </div>
            <p className="arch-hint">animated pulses show data flowing left to right through attention</p>
        </div>
    )
}

function FFNArch({ model, T }) {
    const ratio = model.d_ff / model.d_model
    return (
        <div className="arch-body">
            <div className="arch-row">
                <div className="arch-node arch-node--purple" style={{ height: 52 }}>
                    <span className="arch-node-title">input</span>
                    <span className="arch-node-sub">[{T}x{model.d_model}]</span>
                </div>
                <ArchArrow color="lf-pulse--amber" />
                <div className="arch-node" style={{ height: 52 }}>
                    <span className="arch-node-title">W_up</span>
                    <span className="arch-node-sub">[{model.d_model}x{model.d_ff}]</span>
                </div>
                <ArchArrow color="lf-pulse--amber" />
                <div className="arch-node arch-node--amber arch-node--tall">
                    <span className="arch-node-title">hidden</span>
                    <span className="arch-node-sub">[{T}x{model.d_ff}]</span>
                    <span className="arch-expand-note">{ratio}x wider</span>
                </div>
                <ArchArrow color="lf-pulse--amber" />
                <div className="arch-node" style={{ height: 80 }}>
                    <span className="arch-node-title">{model.ffn_activation.split(' ')[0]}</span>
                    <span className="arch-node-sub">activation</span>
                    <span className="arch-node-sub">elementwise</span>
                </div>
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
            <div className="arch-ffn-note">
                <span>{model.d_model}d &#8594; <span style={{ color: '#EF9F27' }}>{model.d_ff}d</span> &#8594; {model.d_model}d</span>
                <span>the tall "hidden" box shows the {ratio}x expansion</span>
            </div>
            <p className="arch-hint">the bottleneck forces the model to learn compressed, useful representations</p>
        </div>
    )
}

export default function TransformerBlock({ model, tokens, layerIndex }) {
    const [activeSubBlock, setActiveSubBlock] = useState(null)
    const [activeTab, setActiveTab] = useState('main')
    const T = tokens.length

    function handleSubClick(block) {
        if (activeSubBlock === block) {
            setActiveSubBlock(null)
            setActiveTab('main')
        } else {
            setActiveSubBlock(block)
            setActiveTab('main')
        }
    }

    return (
        <div className="layer-detail">
            <p className="layer-detail-intro">
                <strong>Layer {layerIndex}</strong> of {model.n_layers}. Data flows left to right.
                First, the input is normalized using <strong>{model.norm_type}</strong> (this stabilizes the
                numbers so they do not grow too large or too small). Then <strong>self-attention</strong> lets
                every token look at every other token to gather context. The original input is added
                back via a <strong>residual connection</strong> (the "+" circle and dashed skip arc), which
                helps the model train by giving gradients a shortcut. The same pattern repeats:
                normalize, then the <strong>feed-forward network</strong> processes each token independently,
                and another residual connection adds the input back. The output has the
                same shape as the input: <strong>[{T} x {model.d_model}]</strong>.
            </p>

            {/* Horizontal layer flow */}
            <div className="layer-flow-scroll">
                <div className="layer-flow-row">
                    <div className="lf-node lf-node--io">input<sub>[{T}x{model.d_model}]</sub></div>
                    <FlowArrow color="lf-pulse--purple" />
                    <div className="lf-node">{model.norm_type.split(' ')[0]}<sub>normalize</sub></div>
                    <FlowArrow color="lf-pulse--purple" />

                    <div className="lf-skip-area">
                        <svg className="lf-skip-svg" viewBox="0 0 200 34" height="34" preserveAspectRatio="none">
                            <path d="M0,34 C0,8 8,0 30,0 L170,0 C192,0 200,8 200,34" className="lf-skip-path" />
                            <text x="100" y="14" textAnchor="middle" className="lf-skip-text">skip</text>
                        </svg>
                        <div
                            className={`lf-node lf-node--attn ${activeSubBlock === 'attn' ? 'lf-node--active' : ''}`}
                            onClick={() => handleSubClick('attn')}
                        >
                            self-attention<sub>{model.n_heads} heads x {model.head_dim}d</sub>
                        </div>
                    </div>

                    <PlainArrow />
                    <div className="lf-add">+</div>
                    <FlowArrow color="lf-pulse--amber" />
                    <div className="lf-node">{model.norm_type.split(' ')[0]}<sub>normalize</sub></div>
                    <FlowArrow color="lf-pulse--amber" />

                    <div className="lf-skip-area">
                        <svg className="lf-skip-svg" viewBox="0 0 200 34" height="34" preserveAspectRatio="none">
                            <path d="M0,34 C0,8 8,0 30,0 L170,0 C192,0 200,8 200,34" className="lf-skip-path" />
                            <text x="100" y="14" textAnchor="middle" className="lf-skip-text">skip</text>
                        </svg>
                        <div
                            className={`lf-node lf-node--ffn ${activeSubBlock === 'ffn' ? 'lf-node--active' : ''}`}
                            onClick={() => handleSubClick('ffn')}
                        >
                            FFN<sub>{model.d_model} to {model.d_ff} to {model.d_model}</sub>
                        </div>
                    </div>

                    <PlainArrow />
                    <div className="lf-add">+</div>
                    <PlainArrow />
                    <div className="lf-node lf-node--io">output<sub>[{T}x{model.d_model}]</sub></div>
                </div>
            </div>
            <p className="layer-flow-hint">data flows left to right. dashed arcs are residual skip connections. click the colored blocks to see inside.</p>

            {/* Architecture diagrams */}
            <AnimatePresence mode="wait">
                {activeSubBlock && (
                    <motion.div
                        key={activeSubBlock}
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
                    >
                        <div className="arch-panel">
                            <div className="arch-header">
                                <span className="arch-title" style={{ color: activeSubBlock === 'attn' ? '#AFA9EC' : '#EF9F27' }}>
                                    <span className="arch-title-dot" style={{ background: activeSubBlock === 'attn' ? '#7F77DD' : '#BA7517' }} />
                                    {activeSubBlock === 'attn' ? 'self-attention architecture' : 'FFN architecture'}
                                </span>
                                <button className="arch-close" onClick={() => { setActiveSubBlock(null); setActiveTab('main'); }}>close</button>
                            </div>

                            {activeSubBlock === 'attn' && <AttentionArch model={model} T={T} />}
                            {activeSubBlock === 'ffn' && <FFNArch model={model} T={T} />}
                        </div>

                        {/* Step-through / animation detail */}
                        <div className="sub-detail">
                            <div className="sub-detail-header">
                                <span className="sub-detail-title" style={{ color: activeSubBlock === 'attn' ? '#AFA9EC' : '#EF9F27' }}>
                                    {activeSubBlock === 'attn' ? 'step-through' : 'animation'}
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
                                <motion.div
                                    key={activeTab}
                                    initial={{ opacity: 0, y: 8 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    exit={{ opacity: 0, y: -8 }}
                                    transition={{ duration: 0.2 }}
                                >
                                    {activeTab === 'main' && activeSubBlock === 'attn' && <AttentionDetail T={T} model={model} />}
                                    {activeTab === 'main' && activeSubBlock === 'ffn' && <FFNDetail T={T} model={model} />}
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