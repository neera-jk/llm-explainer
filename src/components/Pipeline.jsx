import { motion } from 'framer-motion'

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
    unembed: { dot: '#D85A30', pulse: 'flow-pulse--coral' },
    softmax: { dot: '#1D9E75', pulse: 'flow-pulse--green' },
}

function TokenizerPreview({ tokens }) {
    return (
        <svg viewBox="0 0 100 42" width="100" height="42">
            {tokens.slice(0, 4).map((t, i) => (
                <g key={i} transform={`translate(${i * 24 + 2}, 0)`}>
                    <rect x="0" y="2" width="20" height="14" rx="3" fill="#2a2a32" />
                    <text x="10" y="12" fontSize="6.5" fill="#73726c" textAnchor="middle" fontFamily="monospace">{t.token}</text>
                    <line x1="10" y1="17" x2="10" y2="24" stroke="#444441" strokeWidth="0.8" strokeDasharray="2,1" />
                    <rect x="0" y="25" width="20" height="13" rx="2" fill="#2a2a32" />
                    <text x="10" y="34" fontSize="6" fill="#888780" textAnchor="middle" fontFamily="monospace">{t.id}</text>
                </g>
            ))}
        </svg>
    )
}

function EmbeddingPreview({ tokens }) {
    const heights = [28, 38, 20, 34]
    return (
        <svg viewBox="0 0 100 42" width="100" height="42">
            {tokens.slice(0, 4).map((_, i) => (
                <g key={i} transform={`translate(${i * 24 + 4}, 0)`}>
                    <rect x="0" y={40 - heights[i]} width="18" height={heights[i]} rx="2"
                        fill="rgba(127,119,221,0.15)" stroke="rgba(127,119,221,0.35)" strokeWidth="0.5" />
                    <rect x="0" y={40 - heights[i]} width="18" height="7" rx="2" fill="#7F77DD" />
                </g>
            ))}
        </svg>
    )
}

function UnembedPreview() {
    return (
        <svg viewBox="0 0 100 42" width="100" height="42">
            <rect x="2" y="10" width="24" height="22" rx="3" fill="rgba(216,90,48,0.12)" stroke="rgba(216,90,48,0.35)" strokeWidth="0.5" />
            <text x="14" y="24" fontSize="7" fill="#D85A30" textAnchor="middle" fontFamily="monospace">768</text>
            <line x1="28" y1="21" x2="42" y2="21" stroke="#D85A30" strokeWidth="1" />
            <polygon points="40,18 46,21 40,24" fill="#D85A30" />
            <rect x="48" y="4" width="46" height="34" rx="3" fill="rgba(216,90,48,0.08)" stroke="#D85A30" strokeWidth="0.6" />
            <text x="71" y="19" fontSize="6.5" fill="#D85A30" textAnchor="middle" fontFamily="monospace">50,257</text>
            <text x="71" y="28" fontSize="5.5" fill="rgba(216,90,48,0.5)" textAnchor="middle" fontFamily="monospace">logits</text>
        </svg>
    )
}

function SoftmaxPreview() {
    const bars = [
        { w: 80, o: 1 }, { w: 58, o: 0.7 }, { w: 36, o: 0.45 },
        { w: 22, o: 0.28 }, { w: 10, o: 0.15 },
    ]
    return (
        <svg viewBox="0 0 100 42" width="100" height="42">
            {bars.map((b, i) => (
                <rect key={i} x="0" y={i * 8 + 1} width={b.w} height="6" rx="2"
                    fill="#1D9E75" opacity={b.o} />
            ))}
            <text x="84" y="8" fontSize="5.5" fill="#1D9E75" fontFamily="monospace">38%</text>
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
                {id === 'unembed' && <UnembedPreview />}
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

    // Collapse: show first 3, ellipsis, last layer
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
                            {model.n_layers} transformer layers  ·  {layerShape} each
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