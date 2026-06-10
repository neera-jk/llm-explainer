import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import AttentionDetail from './AttentionDetail'
import FFNDetail from './FFNDetail'
import MatmulViz from './MatmulViz'

function DiagramArrow() {
    return (
        <div className="diagram-arrow">
            <svg width="48" height="16" viewBox="0 0 48 16">
                <line x1="0" y1="8" x2="38" y2="8" stroke="rgba(255,255,255,0.12)" strokeWidth="1.5" />
                <polygon points="36,3 48,8 36,13" fill="rgba(255,255,255,0.12)" />
            </svg>
        </div>
    )
}

export default function TransformerBlock({ model, tokens, layerIndex }) {
    const [activeSubBlock, setActiveSubBlock] = useState(null)
    const [activeTab, setActiveTab] = useState('main')
    const T = tokens.length

    function handleSubBlockClick(block) {
        if (activeSubBlock === block) {
            setActiveSubBlock(null)
            setActiveTab('main')
        } else {
            setActiveSubBlock(block)
            setActiveTab('main')
        }
    }

    const headCount = Math.min(model.n_heads, 6)

    return (
        <div className="layer-detail">
            <p className="layer-detail-intro">
                <strong>Layer {layerIndex}</strong> of {model.n_layers}. Every transformer layer runs two operations in sequence:
                <strong> self-attention</strong> (lets tokens look at each other) then a
                <strong> feed-forward network</strong> (processes each token independently).
                Input and output are always <strong>[{T} x {model.d_model}]</strong>.
                A residual connection adds the input back after each operation.
            </p>

            {/* Two-block diagram */}
            <div className="layer-diagram">
                {/* Self-Attention block */}
                <motion.div
                    className={`layer-diagram-block layer-diagram-block--attn ${activeSubBlock === 'attn' ? 'layer-diagram-block-active' : ''}`}
                    onClick={() => handleSubBlockClick('attn')}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                >
                    <span className="ldb-title" style={{ color: '#AFA9EC' }}>Self-Attention</span>
                    <span className="ldb-sub">{model.attn_type}</span>

                    {/* Stacked heads */}
                    <div className="attn-heads-stack">
                        {Array.from({ length: headCount }, (_, i) => (
                            <motion.div
                                key={i}
                                className="attn-head-bar"
                                initial={{ scaleX: 0 }}
                                animate={{ scaleX: 1 }}
                                transition={{ delay: i * 0.06, duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
                                style={{ transformOrigin: 'left', opacity: 1 - i * 0.12 }}
                            />
                        ))}
                        <span className="attn-heads-label">{model.n_heads} heads x {model.head_dim}d each</span>
                    </div>

                    <span className="ldb-shape" style={{ color: '#AFA9EC' }}>[{T}x{model.d_model}] in/out</span>
                </motion.div>

                <DiagramArrow />

                {/* FFN block */}
                <motion.div
                    className={`layer-diagram-block layer-diagram-block--ffn ${activeSubBlock === 'ffn' ? 'layer-diagram-block-active' : ''}`}
                    onClick={() => handleSubBlockClick('ffn')}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                >
                    <span className="ldb-title" style={{ color: '#EF9F27' }}>Feed-Forward Network</span>
                    <span className="ldb-sub">{model.ffn_activation}</span>

                    {/* Expand/contract visualization */}
                    <svg width="120" height="40" viewBox="0 0 120 40" style={{ margin: '4px 0' }}>
                        <rect x="0" y="10" width="28" height="20" rx="3" fill="rgba(186,117,23,0.15)" stroke="rgba(186,117,23,0.4)" strokeWidth="0.8" />
                        <text x="14" y="23" fontSize="7" fill="#EF9F27" textAnchor="middle" fontFamily="monospace">{model.d_model}</text>
                        <path d="M30 20 L42 20" stroke="rgba(186,117,23,0.3)" strokeWidth="1" />
                        <rect x="44" y="2" width="32" height="36" rx="3" fill="rgba(186,117,23,0.2)" stroke="rgba(186,117,23,0.5)" strokeWidth="0.8" />
                        <text x="60" y="23" fontSize="7" fill="#EF9F27" textAnchor="middle" fontFamily="monospace">{model.d_ff}</text>
                        <path d="M78 20 L90 20" stroke="rgba(186,117,23,0.3)" strokeWidth="1" />
                        <rect x="92" y="10" width="28" height="20" rx="3" fill="rgba(186,117,23,0.15)" stroke="rgba(186,117,23,0.4)" strokeWidth="0.8" />
                        <text x="106" y="23" fontSize="7" fill="#EF9F27" textAnchor="middle" fontFamily="monospace">{model.d_model}</text>
                    </svg>

                    <span className="ldb-shape" style={{ color: '#EF9F27' }}>{model.d_model} &rarr; {model.d_ff} &rarr; {model.d_model}</span>
                </motion.div>
            </div>

            <p className="residual-label">+ residual connection after each block</p>
            <p className="click-hint">click either block to see its full architecture</p>

            {/* Drill-down detail */}
            <AnimatePresence mode="wait">
                {activeSubBlock && (
                    <motion.div
                        key={activeSubBlock}
                        className="sub-detail"
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
                    >
                        <div className="sub-detail-header">
                            <span
                                className="sub-detail-title"
                                style={{ color: activeSubBlock === 'attn' ? '#AFA9EC' : '#EF9F27' }}
                            >
                                {activeSubBlock === 'attn' ? 'Self-Attention' : 'Feed-Forward Network'}
                            </span>
                            <div className="sub-detail-tabs">
                                <button
                                    className={`sub-tab ${activeTab === 'main' ? 'sub-tab-on' : ''}`}
                                    onClick={() => setActiveTab('main')}
                                >
                                    {activeSubBlock === 'attn' ? 'attention' : 'FFN'}
                                </button>
                                <button
                                    className={`sub-tab ${activeTab === 'matmul' ? 'sub-tab-on' : ''}`}
                                    onClick={() => setActiveTab('matmul')}
                                >
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
                                {activeTab === 'main' && activeSubBlock === 'attn' && (
                                    <AttentionDetail T={T} model={model} />
                                )}
                                {activeTab === 'main' && activeSubBlock === 'ffn' && (
                                    <FFNDetail T={T} model={model} />
                                )}
                                {activeTab === 'matmul' && (
                                    <MatmulViz />
                                )}
                            </motion.div>
                        </AnimatePresence>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    )
}