import { useState } from 'react'
import { makeEmbeddingVector } from '../../data/models'
import Tooltip from '../Tooltip'

const SHOW_DIMS = 32
const TOP_DIMS = 10

function valueToColor(v) {
    const a = Math.abs(v)
    return v > 0 ? `rgba(127,119,221,${0.12 + a * 0.88})` : `rgba(216,90,48,${0.12 + a * 0.88})`
}

export default function EmbeddingBlock({ model, tokens, accent }) {
    const a = accent || { color: '#7F77DD', light: 'rgba(127,119,221,0.1)', border: 'rgba(127,119,221,0.3)', text: '#AFA9EC' }
    const [selectedToken, setSelectedToken] = useState(0)
    const [revealed, setRevealed] = useState(SHOW_DIMS)
    const [animating, setAnimating] = useState(false)

    const vecs = tokens.map(t => makeEmbeddingVector(t.id, model.d_model))
    const selectedVec = vecs[selectedToken]
    const topDims = selectedVec.map((v, i) => ({ v, i })).sort((x, y) => Math.abs(y.v) - Math.abs(x.v)).slice(0, TOP_DIMS)

    function handleAnimate() {
        if (animating) return
        setAnimating(true); setRevealed(0)
        let r = 0
        const interval = setInterval(() => {
            r += 2; setRevealed(r)
            if (r >= SHOW_DIMS) { clearInterval(interval); setAnimating(false) }
        }, 55)
    }

    return (
        <div className="block-content">
            <p className="block-explain">
                Each token ID is looked up in an <strong>embedding matrix [{model.vocab_size.toLocaleString()} × {model.d_model}]</strong> to
                produce a {model.d_model}-dim vector. Similar words end up close together.
                A <strong>positional embedding</strong> is added to encode position in the sequence.
            </p>
            <div className="stats-row">
                {[
                    { val: model.d_model, label: 'dims per token' },
                    { val: `${(model.vocab_size * model.d_model / 1e6).toFixed(1)}M`, label: 'embedding params' },
                    { val: `[${tokens.length}×${model.d_model}]`, label: 'output shape' },
                ].map(s => (
                    <div key={s.label} className="stat-card" style={{ background: a.light, borderColor: a.border }}>
                        <span className="stat-val" style={{ color: a.text }}>{s.val}</span>
                        <Tooltip term={s.label}><span className="stat-label">{s.label}</span></Tooltip>
                    </div>
                ))}
            </div>
            <div className="heatmap-legend">
                <span className="legend-item"><span className="legend-swatch" style={{ background: 'rgba(127,119,221,0.9)' }} />positive</span>
                <span className="legend-item"><span className="legend-swatch" style={{ background: 'rgba(216,90,48,0.9)' }} />negative</span>
                <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>intensity = magnitude</span>
                <button className="anim-btn" style={{ marginLeft: 'auto', background: a.light, borderColor: a.border, color: a.text }} onClick={handleAnimate} disabled={animating}>
                    {animating ? 'animating…' : '▶ animate'}
                </button>
            </div>
            <h3 className="section-heading" style={{ color: a.color }}>
                vector heatmap <span className="section-note">first {SHOW_DIMS} of {model.d_model} dims · click a row</span>
            </h3>
            <div className="heatmap">
                {tokens.map((t, ti) => (
                    <div key={ti} className={`heatmap-row ${selectedToken === ti ? 'heatmap-row-selected' : ''}`} onClick={() => setSelectedToken(ti)}>
                        <span className="heatmap-label">{t.token}</span>
                        <div className="heatmap-cells">
                            {vecs[ti].slice(0, SHOW_DIMS).map((v, di) => (
                                <div key={di} className="heatmap-cell" style={{ background: valueToColor(v), opacity: ti === selectedToken && di >= revealed ? 0 : 1, transition: 'opacity 0.08s ease' }} title={`dim ${di}: ${v.toFixed(3)}`} />
                            ))}
                        </div>
                    </div>
                ))}
            </div>
            <h3 className="section-heading" style={{ color: a.color }}>
                top {TOP_DIMS} dims for <span style={{ color: a.text }}>"{tokens[selectedToken].token}"</span>
            </h3>
            <div className="bar-chart">
                {topDims.map(({ v, i }) => (
                    <div key={i} className="bar-row">
                        <span className="bar-label">dim {i}</span>
                        <div className="bar-track"><div className="bar-fill" style={{ width: `${Math.abs(v) * 100}%`, background: v > 0 ? '#7F77DD' : '#D85A30' }} /></div>
                        <span className="bar-val">{v.toFixed(3)}</span>
                    </div>
                ))}
            </div>
            <p className="footnote">Values learned during training — "cat" and "dog" end up similar because they appeared in similar contexts millions of times.</p>
        </div>
    )
}