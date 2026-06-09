import { useEffect, useRef } from 'react'

/** Returns the output tensor shape label for each pipeline block */
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

// ─── Block preview SVGs ───────────────────────────────────────────────────────
// Each SVG gives a visual hint of what the block does.
// Colors match the CSS palette — these are decorative diagrams.

/** Tokenizer — words mapping down to integer ID pills */
function TokenizerPreview({ tokens }) {
    return (
        <svg className="preview-svg" viewBox="0 0 108 48" aria-hidden="true">
            {tokens.slice(0, 4).map((t, i) => (
                <g key={i} transform={`translate(${i * 26 + 2}, 0)`}>
                    <rect x="0" y="6" width="22" height="14" rx="3" fill="#D3D1C7" />
                    <text x="11" y="16" fontSize="7" fill="#444441" textAnchor="middle" fontFamily="monospace">{t.token}</text>
                    <line x1="11" y1="20" x2="11" y2="28" stroke="#B4B2A9" strokeWidth="1" strokeDasharray="2,1" />
                    <rect x="0" y="29" width="22" height="13" rx="2" fill="#AFA9EC" />
                    <text x="11" y="39" fontSize="6.5" fill="#26215C" textAnchor="middle" fontFamily="monospace">{t.id}</text>
                </g>
            ))}
        </svg>
    )
}

/** Embedding — one vertical bar per token representing its d_model-dim vector */
function EmbeddingPreview({ tokens }) {
    const heights = [28, 38, 22, 34, 30]
    return (
        <svg className="preview-svg" viewBox="0 0 108 48" aria-hidden="true">
            {tokens.slice(0, 4).map((_, i) => (
                <g key={i} transform={`translate(${i * 26 + 4}, 0)`}>
                    <rect x="0" y={44 - heights[i]} width="20" height={heights[i]} rx="2" fill="#CECBF6" stroke="#AFA9EC" strokeWidth="0.5" />
                    <rect x="0" y={44 - heights[i]} width="20" height="8" rx="2" fill="#7F77DD" />
                    <text x="10" y="46" fontSize="6" fill="#534AB7" textAnchor="middle" fontFamily="monospace">768d</text>
                </g>
            ))}
        </svg>
    )
}

/** Transformer — stacked layer cards communicating depth */
function TransformerPreview({ model }) {
    return (
        <svg className="preview-svg preview-svg--wide" viewBox="0 0 116 48" aria-hidden="true">
            <rect x="2" y="32" width="112" height="13" rx="3" fill="#CECBF6" stroke="#AFA9EC" strokeWidth="0.5" />
            <rect x="5" y="19" width="106" height="13" rx="3" fill="#CECBF6" stroke="#AFA9EC" strokeWidth="0.5" />
            <rect x="8" y="6" width="100" height="13" rx="3" fill="#AFA9EC" stroke="#7F77DD" strokeWidth="1" />
            <text x="58" y="15.5" fontSize="7" fill="#26215C" textAnchor="middle" fontFamily="sans-serif" fontWeight="500">attn + FFN</text>
            <text x="58" y="28" fontSize="6.5" fill="#534AB7" textAnchor="middle" fontFamily="sans-serif">layer 1 of {model.n_layers}</text>
            <text x="58" y="42" fontSize="6" fill="#888780" textAnchor="middle" fontFamily="sans-serif">· · · {model.n_layers - 1} more · · ·</text>
        </svg>
    )
}

/** Unembed — small 768d vector projecting to a wide vocab logits rectangle */
function UnembedPreview() {
    return (
        <svg className="preview-svg" viewBox="0 0 108 48" aria-hidden="true">
            <rect x="2" y="12" width="28" height="24" rx="3" fill="#F5C4B3" stroke="#F0997B" strokeWidth="0.5" />
            <text x="16" y="24" fontSize="7" fill="#712B13" textAnchor="middle" fontFamily="monospace">768d</text>
            <path d="M31 24 L51 24" stroke="#F0997B" strokeWidth="1.5" />
            <path d="M47 20 L53 24 L47 28" fill="#F0997B" />
            <rect x="53" y="4" width="52" height="40" rx="3" fill="#F5C4B3" stroke="#D85A30" strokeWidth="1" />
            <text x="79" y="18" fontSize="6.5" fill="#712B13" textAnchor="middle" fontFamily="monospace">50,257</text>
            <text x="79" y="28" fontSize="6" fill="#993C1D" textAnchor="middle" fontFamily="monospace">logits</text>
            <text x="79" y="38" fontSize="6" fill="#993C1D" textAnchor="middle" fontFamily="monospace">vocab</text>
        </svg>
    )
}

/** Softmax — probability bar chart with top token highlighted */
function SoftmaxPreview() {
    const bars = [
        { width: 90, color: '#1D9E75', label: '38%' },
        { width: 68, color: '#5DCAA5' },
        { width: 44, color: '#9FE1CB' },
        { width: 28, color: '#9FE1CB' },
        { width: 16, color: '#C0E8D5' },
        { width: 8, color: '#D4EEE3' },
    ]
    return (
        <svg className="preview-svg" viewBox="0 0 108 48" aria-hidden="true">
            {bars.map((bar, i) => (
                <g key={i} transform={`translate(0, ${i * 7 + 2})`}>
                    <rect x="0" y="0" width={bar.width} height="5" rx="2" fill={bar.color} />
                    {bar.label && (
                        <text x={bar.width + 3} y="5" fontSize="6" fill="#085041" fontFamily="monospace">{bar.label}</text>
                    )}
                </g>
            ))}
        </svg>
    )
}

// ─── Animated connector ───────────────────────────────────────────────────────

/**
 * Canvas connector that animates the tensor shape label sliding from
 * one block to the next — represents the full tensor moving at once.
 */
function FlowConnector({ label }) {
    const canvasRef = useRef(null)

    useEffect(() => {
        const cv = canvasRef.current
        if (!cv) return
        const ctx = cv.getContext('2d')
        const W = 80
        const H = 90
        const cy = H / 2 - 19
        const dark = matchMedia('(prefers-color-scheme:dark)').matches

        const trackIdle = dark ? 'rgba(175,169,236,0.4)' : 'rgba(127,119,221,0.3)'
        const trackLive = '#7F77DD'
        const pillBg = dark ? '#26215C' : '#EEEDFE'
        const pillStroke = '#7F77DD'
        const pillText = dark ? '#CECBF6' : '#3C3489'

        let progress = -0.1
        let rafId = null
        let timerId = null
        let mounted = true

        function drawIdle() {
            ctx.clearRect(0, 0, W, H)
            ctx.setLineDash([4, 4])
            ctx.beginPath()
            ctx.moveTo(4, cy)
            ctx.lineTo(W - 4, cy)
            ctx.strokeStyle = trackIdle
            ctx.lineWidth = 1.5
            ctx.stroke()
            ctx.setLineDash([])
            ctx.beginPath()
            ctx.moveTo(W - 10, cy - 4)
            ctx.lineTo(W - 2, cy)
            ctx.lineTo(W - 10, cy + 4)
            ctx.strokeStyle = trackIdle
            ctx.lineWidth = 1
            ctx.stroke()
        }

        function easeInOut(t) {
            return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t
        }

        function drawFrame() {
            if (!mounted) return
            progress += 0.007

            if (progress > 1.1) {
                progress = -0.1
                drawIdle()
                timerId = setTimeout(() => {
                    if (mounted) rafId = requestAnimationFrame(drawFrame)
                }, 1400)
                return
            }

            if (progress < 0) {
                rafId = requestAnimationFrame(drawFrame)
                return
            }

            const e = easeInOut(Math.min(Math.max(progress, 0), 1))
            ctx.clearRect(0, 0, W, H)

            // solid track while tensor is in flight
            ctx.setLineDash([])
            ctx.beginPath()
            ctx.moveTo(4, cy)
            ctx.lineTo(W - 4, cy)
            ctx.strokeStyle = trackLive
            ctx.lineWidth = 1.5
            ctx.stroke()

            ctx.beginPath()
            ctx.moveTo(W - 10, cy - 4)
            ctx.lineTo(W - 2, cy)
            ctx.lineTo(W - 10, cy + 4)
            ctx.fillStyle = trackLive
            ctx.strokeStyle = trackLive
            ctx.lineWidth = 1
            ctx.fill()

            // size the pill to fit the label text
            ctx.font = '500 9px -apple-system, monospace'
            const pw = Math.max(ctx.measureText(label).width + 14, 44)
            const bH = 20
            const bx = -pw + e * (W + pw)

            ctx.beginPath()
            ctx.roundRect(bx, cy - bH / 2, pw, bH, 4)
            ctx.fillStyle = pillBg
            ctx.fill()
            ctx.strokeStyle = pillStroke
            ctx.lineWidth = 1
            ctx.stroke()

            ctx.fillStyle = pillText
            ctx.textAlign = 'center'
            ctx.fillText(label, bx + pw / 2, cy + 3.5)

            rafId = requestAnimationFrame(drawFrame)
        }

        drawIdle()
        // stagger start times so connectors don't all animate in sync
        timerId = setTimeout(
            () => { if (mounted) rafId = requestAnimationFrame(drawFrame) },
            Math.random() * 800 + 200
        )

        return () => {
            mounted = false
            cancelAnimationFrame(rafId)
            clearTimeout(timerId)
        }
    }, [label])

    return <canvas ref={canvasRef} className="flow-canvas" width={80} height={90} />
}

// ─── Block config ─────────────────────────────────────────────────────────────

const BLOCKS = [
    { id: 'tokenizer', name: 'Tokenizer', sub: 'text → integer IDs', accent: false },
    { id: 'embedding', name: 'Embedding', sub: 'IDs → 768-dim vectors', accent: false },
    { id: 'transformer', name: '', sub: 'attention + FFN each', accent: true },
    { id: 'unembed', name: 'Unembed', sub: '768d → vocab logits', accent: false },
    { id: 'softmax', name: 'Softmax', sub: 'logits → probabilities', accent: false },
]

// ─── Main export ──────────────────────────────────────────────────────────────

/**
 * Horizontal pipeline of blocks representing the GPT-2 forward pass.
 * Each block shows a mini SVG preview and an animated flow connector.
 */
export default function Pipeline({ model, tokens, onBlockClick, activeBlock }) {
    return (
        <div className="pipeline">
            {BLOCKS.map((block, i) => {
                const shape = getShape(block.id, tokens, model)
                const label = block.id === 'transformer'
                    ? `×${model.n_layers} Layers`
                    : block.name

                return (
                    <div key={block.id} className="pipeline-item">
                        <div className="block-col">
                            <button
                                className={[
                                    'block',
                                    block.accent ? 'block-accent' : '',
                                    activeBlock === block.id ? 'block-active' : '',
                                ].filter(Boolean).join(' ')}
                                onClick={() => onBlockClick(block.id)}
                            >
                                {/* mini diagram — visual hint of what this block does */}
                                <div className={`block-preview block-preview--${block.id}`}>
                                    {block.id === 'tokenizer' && <TokenizerPreview tokens={tokens} />}
                                    {block.id === 'embedding' && <EmbeddingPreview tokens={tokens} />}
                                    {block.id === 'transformer' && <TransformerPreview model={model} />}
                                    {block.id === 'unembed' && <UnembedPreview />}
                                    {block.id === 'softmax' && <SoftmaxPreview />}
                                </div>

                                {/* block name and one-line description */}
                                <div className="block-info">
                                    <span className="block-name">{label}</span>
                                    <span className="block-sub">{block.sub}</span>
                                </div>

                                {/* output tensor shape in the footer */}
                                <div className={`block-footer ${block.accent ? 'block-footer--accent' : ''}`}>
                                    <span className="block-shape">{shape}</span>
                                </div>
                            </button>
                        </div>

                        {/* animated tensor connector — not shown after the last block */}
                        {i < BLOCKS.length - 1 && (
                            <FlowConnector label={shape} />
                        )}
                    </div>
                )
            })}
        </div>
    )
}