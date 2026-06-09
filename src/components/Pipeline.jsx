import { useEffect, useRef } from 'react'

function getShape(blockId, tokens, model) {
    const T = tokens.length
    return {
        tokenizer: `[${T}]`,
        embedding: `[${T}×${model.d_model}]`,
        transformer: `[${T}×${model.d_model}]`,
        unembed: `[${T}×${model.vocab_size.toLocaleString()}]`,
        softmax: `[${model.vocab_size.toLocaleString()}]`,
    }[blockId]
}

const ACCENTS = {
    tokenizer: { bar: '#888780', shape: '#888780', border: 'rgba(136,135,128,0.4)' },
    embedding: { bar: '#7F77DD', shape: '#AFA9EC', border: 'rgba(127,119,221,0.5)' },
    transformer: { bar: '#BA7517', shape: '#EF9F27', border: 'rgba(186,117,23,0.5)' },
    unembed: { bar: '#D85A30', shape: '#F0997B', border: 'rgba(216,90,48,0.5)' },
    softmax: { bar: '#1D9E75', shape: '#5DCAA5', border: 'rgba(29,158,117,0.5)' },
}

function TokenizerPreview({ tokens }) {
    return (
        <svg className="preview-svg" viewBox="0 0 108 52" aria-hidden="true">
            {tokens.slice(0, 4).map((t, i) => (
                <g key={i} transform={`translate(${i * 26 + 2}, 0)`}>
                    <rect x="0" y="4" width="22" height="16" rx="3" fill="#2c2c2a" />
                    <text x="11" y="15" fontSize="7" fill="#73726c" textAnchor="middle" fontFamily="monospace">{t.token}</text>
                    <line x1="11" y1="20" x2="11" y2="28" stroke="#444441" strokeWidth="1" strokeDasharray="2,1" />
                    <rect x="0" y="29" width="22" height="16" rx="2" fill="#3a3a38" />
                    <text x="11" y="41" fontSize="6.5" fill="#888780" textAnchor="middle" fontFamily="monospace">{t.id}</text>
                </g>
            ))}
        </svg>
    )
}

function EmbeddingPreview({ tokens }) {
    const heights = [30, 42, 24, 36]
    return (
        <svg className="preview-svg" viewBox="0 0 108 52" aria-hidden="true">
            {tokens.slice(0, 4).map((_, i) => (
                <g key={i} transform={`translate(${i * 26 + 4}, 0)`}>
                    <rect x="0" y={48 - heights[i]} width="20" height={heights[i]} rx="2" fill="rgba(127,119,221,0.2)" stroke="rgba(127,119,221,0.4)" strokeWidth="0.5" />
                    <rect x="0" y={48 - heights[i]} width="20" height="9" rx="2" fill="#7F77DD" />
                    <text x="10" y="50" fontSize="6" fill="#7F77DD" textAnchor="middle" fontFamily="monospace">768d</text>
                </g>
            ))}
        </svg>
    )
}

function TransformerPreview({ model }) {
    return (
        <svg className="preview-svg preview-svg--wide" viewBox="0 0 120 62" aria-hidden="true">
            {/* shadow layers — offset to show stack depth */}
            <rect x="12" y="40" width="100" height="13" rx="3" fill="rgba(186,117,23,0.06)" stroke="rgba(186,117,23,0.12)" strokeWidth="0.5" />
            <rect x="7" y="27" width="106" height="13" rx="3" fill="rgba(186,117,23,0.1)" stroke="rgba(186,117,23,0.2)" strokeWidth="0.5" />
            <rect x="3" y="14" width="110" height="13" rx="3" fill="rgba(186,117,23,0.15)" stroke="rgba(186,117,23,0.28)" strokeWidth="0.5" />
            {/* front active layer */}
            <rect x="0" y="2" width="116" height="13" rx="3" fill="rgba(186,117,23,0.28)" stroke="#BA7517" strokeWidth="1" />
            <text x="58" y="11.5" fontSize="7" fill="#EF9F27" textAnchor="middle" fontFamily="sans-serif" fontWeight="500">attn + FFN · layer 1 of {model.n_layers}</text>
            <text x="58" y="58" fontSize="5.5" fill="#555552" textAnchor="middle" fontFamily="sans-serif">· · · {model.n_layers - 1} more layers below · · ·</text>
        </svg>
    )
}

function UnembedPreview() {
    return (
        <svg className="preview-svg" viewBox="0 0 108 52" aria-hidden="true">
            <rect x="2" y="14" width="28" height="24" rx="3" fill="rgba(216,90,48,0.15)" stroke="rgba(216,90,48,0.4)" strokeWidth="0.5" />
            <text x="16" y="28" fontSize="7.5" fill="#F0997B" textAnchor="middle" fontFamily="monospace">768d</text>
            <path d="M31 26 L50 26" stroke="#D85A30" strokeWidth="1.5" />
            <path d="M46 22 L52 26 L46 30" fill="#D85A30" />
            <rect x="53" y="6" width="52" height="40" rx="3" fill="rgba(216,90,48,0.12)" stroke="#D85A30" strokeWidth="1" />
            <text x="79" y="21" fontSize="7" fill="#F0997B" textAnchor="middle" fontFamily="monospace">50,257</text>
            <text x="79" y="31" fontSize="6.5" fill="#993C1D" textAnchor="middle" fontFamily="monospace">logits</text>
            <text x="79" y="41" fontSize="6" fill="#993C1D" textAnchor="middle" fontFamily="monospace">vocab</text>
        </svg>
    )
}

function SoftmaxPreview() {
    const bars = [
        { width: 88, color: '#1D9E75', label: '38%' },
        { width: 66, color: '#0F6E56' },
        { width: 44, color: '#085041' },
        { width: 28, color: '#085041' },
        { width: 14, color: '#04342C' },
    ]
    return (
        <svg className="preview-svg" viewBox="0 0 108 52" aria-hidden="true">
            {bars.map((bar, i) => (
                <g key={i} transform={`translate(0, ${i * 9 + 2})`}>
                    <rect x="0" y="0" width={bar.width} height="7" rx="2" fill={bar.color} />
                    {bar.label && (
                        <text x={bar.width + 3} y="7" fontSize="6.5" fill="#1D9E75" fontFamily="monospace">{bar.label}</text>
                    )}
                </g>
            ))}
        </svg>
    )
}

function FlowConnector({ label }) {
    const canvasRef = useRef(null)

    useEffect(() => {
        const cv = canvasRef.current
        if (!cv) return
        const ctx = cv.getContext('2d')
        const W = cv.offsetWidth || 80
        const H = 90
        const cy = H / 2 - 20

        const trackIdle = 'rgba(255,255,255,0.07)'
        const trackLive = '#7F77DD'
        const pillBg = 'rgba(127,119,221,0.15)'
        const pillStroke = 'rgba(127,119,221,0.5)'
        const pillText = '#AFA9EC'

        let progress = -0.1
        let rafId = null
        let timerId = null
        let mounted = true

        function drawIdle() {
            ctx.clearRect(0, 0, W, H)
            ctx.setLineDash([4, 4])
            ctx.beginPath(); ctx.moveTo(4, cy); ctx.lineTo(W - 4, cy)
            ctx.strokeStyle = trackIdle; ctx.lineWidth = 1.5; ctx.stroke()
            ctx.setLineDash([])
            ctx.beginPath()
            ctx.moveTo(W - 10, cy - 4); ctx.lineTo(W - 2, cy); ctx.lineTo(W - 10, cy + 4)
            ctx.strokeStyle = trackIdle; ctx.lineWidth = 1; ctx.stroke()
        }

        function ease(t) { return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t }

        function drawFrame() {
            if (!mounted) return
            progress += 0.007
            if (progress > 1.1) {
                progress = -0.1
                drawIdle()
                timerId = setTimeout(() => { if (mounted) rafId = requestAnimationFrame(drawFrame) }, 1400)
                return
            }
            if (progress < 0) { rafId = requestAnimationFrame(drawFrame); return }

            const e = ease(Math.min(Math.max(progress, 0), 1))
            ctx.clearRect(0, 0, W, H)
            ctx.setLineDash([])
            ctx.beginPath(); ctx.moveTo(4, cy); ctx.lineTo(W - 4, cy)
            ctx.strokeStyle = trackLive; ctx.lineWidth = 1.5; ctx.stroke()
            ctx.beginPath()
            ctx.moveTo(W - 10, cy - 4); ctx.lineTo(W - 2, cy); ctx.lineTo(W - 10, cy + 4)
            ctx.fillStyle = trackLive; ctx.strokeStyle = trackLive; ctx.lineWidth = 1; ctx.fill()

            ctx.font = '500 9px -apple-system, monospace'
            const pw = Math.max(ctx.measureText(label).width + 14, 44)
            const bH = 20
            const bx = -pw + e * (W + pw)
            ctx.beginPath(); ctx.roundRect(bx, cy - bH / 2, pw, bH, 4)
            ctx.fillStyle = pillBg; ctx.fill()
            ctx.strokeStyle = pillStroke; ctx.lineWidth = 1; ctx.stroke()
            ctx.fillStyle = pillText; ctx.textAlign = 'center'
            ctx.fillText(label, bx + pw / 2, cy + 3.5)
            rafId = requestAnimationFrame(drawFrame)
        }

        drawIdle()
        timerId = setTimeout(
            () => { if (mounted) rafId = requestAnimationFrame(drawFrame) },
            Math.random() * 800 + 200
        )
        return () => { mounted = false; cancelAnimationFrame(rafId); clearTimeout(timerId) }
    }, [label])

    return <canvas ref={canvasRef} className="flow-canvas" width={80} height={90} />
}

const BLOCKS = [
    { id: 'tokenizer', name: 'Tokenizer', sub: 'text → integer IDs', step: '01' },
    { id: 'embedding', name: 'Embedding', sub: 'IDs → 768-dim vectors', step: '02' },
    { id: 'transformer', name: '', sub: 'attention + FFN each', step: '03', wide: true },
    { id: 'unembed', name: 'Unembed', sub: '768d → vocab logits', step: '04' },
    { id: 'softmax', name: 'Softmax', sub: 'logits → probabilities', step: '05' },
]

export default function Pipeline({ model, tokens, onBlockClick, activeBlock }) {
    return (
        <div className="pipeline-wrap">
            <div className="pipeline">
                {BLOCKS.map((block, i) => {
                    const shape = getShape(block.id, tokens, model)
                    const label = block.id === 'transformer' ? `×${model.n_layers} Layers` : block.name
                    const accent = ACCENTS[block.id]
                    const isActive = activeBlock === block.id

                    return (
                        <div key={block.id} className="pipeline-item">
                            <div className="block-col">
                                <button
                                    className={[
                                        'block',
                                        block.wide ? 'block-accent' : '',
                                        isActive ? 'block-active' : '',
                                        `block--${block.id}`,
                                    ].filter(Boolean).join(' ')}
                                    style={{
                                        borderLeftColor: accent.bar,
                                        ...(isActive && {
                                            borderColor: accent.border,
                                            borderLeftColor: accent.bar,
                                            boxShadow: `0 0 0 1px ${accent.border}, 0 12px 32px rgba(0,0,0,0.5)`,
                                        }),
                                    }}
                                    onClick={() => onBlockClick(block.id)}
                                >
                                    {/* step number uses accent color */}
                                    <span className="block-step" style={{ color: accent.bar }}>{block.step}</span>

                                    {/* mini SVG preview */}
                                    <div className={`block-preview block-preview--${block.id}`}>
                                        {block.id === 'tokenizer' && <TokenizerPreview tokens={tokens} />}
                                        {block.id === 'embedding' && <EmbeddingPreview tokens={tokens} />}
                                        {block.id === 'transformer' && <TransformerPreview model={model} />}
                                        {block.id === 'unembed' && <UnembedPreview />}
                                        {block.id === 'softmax' && <SoftmaxPreview />}
                                    </div>

                                    <div className="block-info">
                                        <span className="block-name">{label}</span>
                                        <span className="block-sub">{block.sub}</span>
                                    </div>

                                    <div className="block-footer">
                                        <span className="block-shape" style={{ color: accent.shape }}>{shape}</span>
                                    </div>
                                </button>
                            </div>

                            {i < BLOCKS.length - 1 && (
                                <FlowConnector label={shape} />
                            )}
                        </div>
                    )
                })}
            </div>
            <p className="pipeline-hint">click any block to explore its internals</p>
        </div>
    )
}