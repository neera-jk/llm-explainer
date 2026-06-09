import { useState } from 'react'
import { makeEmbeddingVector } from '../../data/models'

const SHOW_DIMS = 32  // heatmap columns
const TOP_DIMS = 10  // bar chart rows

/** Maps a vector value to a heatmap color — purple positive, coral negative */
function valueToColor(v) {
    const a = Math.abs(v)
    return v > 0
        ? `rgba(127,119,221,${0.12 + a * 0.88})`
        : `rgba(216,90,48,${0.12 + a * 0.88})`
}

/**
 * Embedding block detail.
 * Shows each token's d_model-dim vector as a color-coded heatmap.
 * The animate button reveals heatmap cells left-to-right.
 * Clicking a token row updates the bar chart to show its top dims.
 */
export default function EmbeddingBlock({ model, tokens }) {
    const [selectedToken, setSelectedToken] = useState(0)
    const [revealed, setRevealed] = useState(SHOW_DIMS)
    const [animating, setAnimating] = useState(false)

    const vecs = tokens.map(t => makeEmbeddingVector(t.id, model.d_model))
    const selectedVec = vecs[selectedToken]

    const topDims = selectedVec
        .map((v, i) => ({ v, i }))
        .sort((a, b) => Math.abs(b.v) - Math.abs(a.v))
        .slice(0, TOP_DIMS)

    function handleAnimate() {
        if (animating) return
        setAnimating(true)
        setRevealed(0)
        let r = 0
        const interval = setInterval(() => {
            r += 2
            setRevealed(r)
            if (r >= SHOW_DIMS) { clearInterval(interval); setAnimating(false) }
        }, 55)
    }

    return (
        <div className="block-content">
            <p className="block-explain">
                Each token ID is looked up in an <strong>embedding matrix
                    [{model.vocab_size.toLocaleString()} × {model.d_model}]</strong> to produce
                a {model.d_model}-dim vector. This is the token's coordinates in meaning space —
                similar words end up close together. A <strong>positional embedding</strong> is
                added to encode where in the sequence the token sits.
            </p>

            <div className="stats-row">
                <div className="stat-card">
                    <span className="stat-val">{model.d_model}</span>
                    <span className="stat-label">dims per token</span>
                </div>
                <div className="stat-card">
                    <span className="stat-val">{(model.vocab_size * model.d_model / 1e6).toFixed(1)}M</span>
                    <span className="stat-label">embedding params</span>
                </div>
                <div className="stat-card">
                    <span className="stat-val">[{tokens.length}×{model.d_model}]</span>
                    <span className="stat-label">output shape</span>
                </div>
            </div>

            <div className="heatmap-legend">
                <span className="legend-item">
                    <span className="legend-swatch" style={{ background: 'rgba(127,119,221,0.9)' }} />
                    positive
                </span>
                <span className="legend-item">
                    <span className="legend-swatch" style={{ background: 'rgba(216,90,48,0.9)' }} />
                    negative
                </span>
                <span style={{ fontSize: 11, color: '#888780' }}>intensity = magnitude</span>
                <button
                    className="anim-btn"
                    style={{ marginLeft: 'auto' }}
                    onClick={handleAnimate}
                    disabled={animating}
                >
                    {animating ? 'animating…' : '▶ animate'}
                </button>
            </div>

            <h3 className="section-heading">
                vector heatmap
                <span className="section-note">first {SHOW_DIMS} of {model.d_model} dims · click a row to inspect</span>
            </h3>
            <div className="heatmap">
                {tokens.map((t, ti) => (
                    <div
                        key={ti}
                        className={`heatmap-row ${selectedToken === ti ? 'heatmap-row-selected' : ''}`}
                        onClick={() => setSelectedToken(ti)}
                    >
                        <span className="heatmap-label">{t.token}</span>
                        <div className="heatmap-cells">
                            {vecs[ti].slice(0, SHOW_DIMS).map((v, di) => (
                                <div
                                    key={di}
                                    className="heatmap-cell"
                                    style={{
                                        background: valueToColor(v),
                                        opacity: ti === selectedToken && di >= revealed ? 0 : 1,
                                        transition: 'opacity 0.08s ease',
                                    }}
                                    title={`dim ${di}: ${v.toFixed(3)}`}
                                />
                            ))}
                        </div>
                    </div>
                ))}
            </div>

            <h3 className="section-heading">
                top {TOP_DIMS} dims for
                <span className="accent"> "{tokens[selectedToken].token}"</span>
            </h3>
            <div className="bar-chart">
                {topDims.map(({ v, i }) => (
                    <div key={i} className="bar-row">
                        <span className="bar-label">dim {i}</span>
                        <div className="bar-track">
                            <div
                                className="bar-fill"
                                style={{ width: `${Math.abs(v) * 100}%`, background: v > 0 ? '#7F77DD' : '#D85A30' }}
                            />
                        </div>
                        <span className="bar-val">{v.toFixed(3)}</span>
                    </div>
                ))}
            </div>

            <p className="footnote">
                These values are learned during training. After training, "cat" and "dog" will
                have similar vectors because they appeared in similar contexts millions of times.
            </p>
        </div>
    )
}