import { motion } from 'framer-motion'
import { getFamily, getFFNLabel } from '../data/models'

function getShape(blockId, tokens, model) {
    const T = tokens.length
    return {
        tokenizer: `[${T}]`,
        embedding: `[${T}×${model.d_model}]`,
        unembed: `[${T}×${model.vocab_size.toLocaleString()}]`,
        softmax: `[${model.vocab_size.toLocaleString()}]`,
    }[blockId]
}

const COLORS = {
    tokenizer: { dot: '#888780', pulse: 'flow-pulse--gray' },
    embedding: { dot: '#7F77DD', pulse: 'flow-pulse--purple' },
    unembed: { dot: '#b85428', pulse: 'flow-pulse--coral' },
    softmax: { dot: '#1a8a65', pulse: 'flow-pulse--green' },
}

function TokenizerPreview({ tokens }) {
    return (
        <svg viewBox="0 0 110 48" width="110" height="48">
            {tokens.slice(0, 4).map((t, i) => (
                <g key={i} transform={`translate(${i * 26 + 2}, 0)`}>
                    <rect x="0" y="2" width="22" height="16" rx="3" fill="#2a2a32" />
                    <text x="11" y="13" fontSize="7" fill="#73726c" textAnchor="middle" fontFamily="monospace">{t.token}</text>
                    <line x1="11" y1="19" x2="11" y2="27" stroke="#444441" strokeWidth="0.8" strokeDasharray="2,1" />
                    <rect x="0" y="28" width="22" height="15" rx="2" fill="#2a2a32" />
                    <text x="11" y="38" fontSize="6.5" fill="#888780" textAnchor="middle" fontFamily="monospace">{t.id}</text>
                </g>
            ))}
        </svg>
    )
}

function EmbeddingPreview({ tokens }) {
    const heights = [30, 42, 22, 36]
    return (
        <svg viewBox="0 0 110 48" width="110" height="48">
            {tokens.slice(0, 4).map((_, i) => (
                <g key={i} transform={`translate(${i * 26 + 4}, 0)`}>
                    <rect x="0" y={46 - heights[i]} width="20" height={heights[i]} rx="2"
                        fill="rgba(127,119,221,0.15)" stroke="rgba(127,119,221,0.35)" strokeWidth="0.5" />
                    <rect x="0" y={46 - heights[i]} width="20" height="8" rx="2" fill="#7F77DD" />
                </g>
            ))}
        </svg>
    )
}

function UnembedPreview({ model }) {
    const vocabLabel = model.vocab_size >= 1000
        ? `${Math.round(model.vocab_size / 1000)}K`
        : `${model.vocab_size}`
    return (
        <svg viewBox="0 0 110 48" width="110" height="48">
            <rect x="2" y="12" width="26" height="24" rx="3" fill="rgba(184,84,40,0.12)" stroke="rgba(184,84,40,0.35)" strokeWidth="0.5" />
            <text x="15" y="26" fontSize="7.5" fill="#b85428" textAnchor="middle" fontFamily="monospace">{model.d_model}</text>
            <line x1="30" y1="24" x2="44" y2="24" stroke="#b85428" strokeWidth="1" />
            <polygon points="42,21 48,24 42,27" fill="#b85428" />
            <rect x="50" y="5" width="52" height="38" rx="3" fill="rgba(184,84,40,0.08)" stroke="#b85428" strokeWidth="0.6" />
            <text x="76" y="22" fontSize="7" fill="#b85428" textAnchor="middle" fontFamily="monospace">{vocabLabel}</text>
            <text x="76" y="32" fontSize="6" fill="rgba(184,84,40,0.5)" textAnchor="middle" fontFamily="monospace">logits</text>
        </svg>
    )
}

function SoftmaxPreview() {
    const bars = [
        { w: 84, o: 1 }, { w: 62, o: 0.7 }, { w: 40, o: 0.45 },
        { w: 26, o: 0.28 }, { w: 12, o: 0.15 },
    ]
    return (
        <svg viewBox="0 0 110 48" width="110" height="48">
            {bars.map((b, i) => (
                <rect key={i} x="0" y={i * 9 + 1} width={b.w} height="7" rx="2"
                    fill="#1a8a65" opacity={b.o} />
            ))}
            <text x="90" y="9" fontSize="6" fill="#1a8a65" fontFamily="monospace">38%</text>
        </svg>
    )
}

function FlowConn({ color }) {
    return (
        <div className="flow-conn">
            <svg width="36" height="12" viewBox="0 0 36 12">
                <line x1="0" y1="6" x2="28" y2="6" className="flow-track" />
                <line x1="0" y1="6" x2="28" y2="6" className={`flow-pulse ${color}`} />
                <polygon points="26,2 36,6 26,10" className="flow-head" />
            </svg>
        </div>
    )
}

function PBlock({ id, name, sub, tokens, model, isActive, onClick, index }) {
    const shape = getShape(id, tokens, model)
    const c = COLORS[id]
    return (
        <motion.button
            className={`pblock pblock--${id} ${isActive ? 'pblock-active' : ''}`}
            onClick={onClick}
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: index * 0.06, ease: [0.16, 1, 0.3, 1] }}
        >
            <span className="pblock-dot" style={{ background: c.dot }} />
            <div className="pblock-preview">
                {id === 'tokenizer' && <TokenizerPreview tokens={tokens} />}
                {id === 'embedding' && <EmbeddingPreview tokens={tokens} />}
                {id === 'unembed' && <UnembedPreview model={model} />}
                {id === 'softmax' && <SoftmaxPreview />}
            </div>
            <div className="pblock-info">
                <span className="pblock-name">{name}</span>
                <span className="pblock-sub">{sub}</span>
            </div>
            <div className="pblock-footer">
                <span className="pblock-shape">{shape}</span>
            </div>
        </motion.button>
    )
}

function LayerBlock({ index, isActive, onClick, delay }) {
    return (
        <motion.button
            className={`layer-block ${isActive ? 'layer-block-active' : ''}`}
            onClick={onClick}
            initial={{ opacity: 0, scale: 0.85 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3, delay, ease: [0.16, 1, 0.3, 1] }}
        >
            <span className="layer-block-num">{index}</span>
            <span className="layer-block-label">layer</span>
        </motion.button>
    )
}

function LayerFlow() {
    return (
        <div className="layer-flow">
            <div className="layer-flow-dot" />
        </div>
    )
}

export default function Pipeline({ model, tokens, onBlockClick, onLayerClick, activeBlock, activeLayer }) {
    const T = tokens.length
    const layerShape = `[${T}×${model.d_model}]`
    const fam = getFamily(model)
    const ffnLabel = getFFNLabel(model, fam)

    const showFirst = Math.min(3, model.n_layers)
    const lastLayer = model.n_layers - 1
    const needsEllipsis = model.n_layers > 5

    const visibleLayers = []
    for (let i = 0; i < showFirst; i++) visibleLayers.push(i)
    if (needsEllipsis) {
        visibleLayers.push('...')
        visibleLayers.push(lastLayer)
    } else {
        for (let i = showFirst; i < model.n_layers; i++) visibleLayers.push(i)
    }

    return (
        <div className="pipeline-section">
            <div className="pipeline-scroll">
                <div className="pipeline">
                    <PBlock id="tokenizer" name="Tokenizer" sub="text to integer IDs"
                        tokens={tokens} model={model}
                        isActive={activeBlock === 'tokenizer'}
                        onClick={() => onBlockClick('tokenizer')} index={0} />
                    <FlowConn color="flow-pulse--gray" />

                    <PBlock id="embedding" name="Embedding" sub={`IDs to ${model.d_model}-dim vectors`}
                        tokens={tokens} model={model}
                        isActive={activeBlock === 'embedding'}
                        onClick={() => onBlockClick('embedding')} index={1} />
                    <FlowConn color="flow-pulse--purple" />

                    <div className="layer-group">
                        {visibleLayers.map((item, vi) => {
                            if (item === '...') {
                                return (
                                    <div key="ell" style={{ display: 'flex', alignItems: 'center' }}>
                                        <LayerFlow />
                                        <span className="layer-ellipsis">···</span>
                                        <LayerFlow />
                                    </div>
                                )
                            }
                            const i = item
                            return (
                                <div key={i} style={{ display: 'flex', alignItems: 'center' }}>
                                    <LayerBlock
                                        index={i}
                                        isActive={activeLayer === i}
                                        onClick={() => onLayerClick(i)}
                                        delay={0.12 + vi * 0.04}
                                    />
                                    {vi < visibleLayers.length - 1 && visibleLayers[vi + 1] !== '...' && typeof item === 'number' && <LayerFlow />}
                                </div>
                            )
                        })}
                        <motion.span className="layer-group-label"
                            initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }}>
                            {model.n_layers} transformer layers  ·  {layerShape} each  ·  {fam.norm}, {fam.attn}, {ffnLabel}
                        </motion.span>
                    </div>

                    <FlowConn color="flow-pulse--amber" />

                    <PBlock id="unembed" name="Unembed" sub={`${model.d_model}d to vocab logits`}
                        tokens={tokens} model={model}
                        isActive={activeBlock === 'unembed'}
                        onClick={() => onBlockClick('unembed')} index={model.n_layers + 2} />
                    <FlowConn color="flow-pulse--coral" />

                    <PBlock id="softmax" name="Softmax" sub="logits to probabilities"
                        tokens={tokens} model={model}
                        isActive={activeBlock === 'softmax'}
                        onClick={() => onBlockClick('softmax')} index={model.n_layers + 3} />
                </div>
            </div>
            <p className="pipeline-hint">click any block to explore its internals</p>
        </div>
    )
}