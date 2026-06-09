import { useState } from 'react'
import { NEXT_TOKEN_PREDICTIONS } from '../../data/prompts'
import { softmax } from '../../data/models'

export default function SoftmaxBlock({ model, tokens, promptKey, accent }) {
    const a = accent || { color: '#1D9E75', light: 'rgba(29,158,117,0.1)', border: 'rgba(29,158,117,0.3)', text: '#5DCAA5' }
    const [temperature, setTemperature] = useState(1.0)
    const [topK, setTopK] = useState(5)

    const predictions = NEXT_TOKEN_PREDICTIONS[promptKey] || []
    const probs = softmax(predictions.map(p => p.logit), temperature)
    const ranked = predictions.map((p, i) => ({ ...p, prob: probs[i] })).sort((x, y) => y.prob - x.prob)
    const topKSet = new Set(ranked.slice(0, topK).map(p => p.token))

    function tempNote() {
        if (temperature < 0.7) return 'Very low — model almost always picks the top token.'
        if (temperature <= 1.3) return 'Normal range — good balance of coherence and variety.'
        if (temperature <= 2.0) return 'High — more creative and surprising, less coherent.'
        return 'Very high — output is nearly random at this point.'
    }

    return (
        <div className="block-content">
            <p className="block-explain">
                Softmax converts raw logits into <strong>probabilities that sum to 1.0</strong>.
                Temperature scales the logits — lower means sharper, higher means flatter.
                Top-K limits sampling to the K most likely tokens.
            </p>
            <div className="stats-row">
                {[
                    { val: `${(ranked[0]?.prob * 100).toFixed(1)}%`, label: 'top token prob' },
                    { val: ranked[0]?.token, label: 'most likely next' },
                    { val: temperature.toFixed(1), label: 'temperature' },
                ].map(s => (
                    <div key={s.label} className="stat-card" style={{ background: a.light, borderColor: a.border }}>
                        <span className="stat-val" style={{ color: a.text }}>{s.val}</span>
                        <span className="stat-label">{s.label}</span>
                    </div>
                ))}
            </div>
            <div className="controls">
                <div className="control-row">
                    <label className="control-label">temperature</label>
                    <input type="range" min="1" max="30" step="1" value={Math.round(temperature * 10)} onChange={e => setTemperature(parseInt(e.target.value) / 10)} className="slider" style={{ accentColor: a.color }} />
                    <span className="control-val">{temperature.toFixed(1)}</span>
                </div>
                <div className="control-row">
                    <label className="control-label">top-k</label>
                    <input type="range" min="1" max={predictions.length} step="1" value={topK} onChange={e => setTopK(parseInt(e.target.value))} className="slider" style={{ accentColor: a.color }} />
                    <span className="control-val">{topK}</span>
                </div>
            </div>
            <div className="temp-note" style={{ borderColor: a.border, color: a.text }}>{tempNote()}</div>
            <h3 className="section-heading" style={{ color: a.color }}>
                probability distribution <span className="section-note">greyed out = excluded by top-{topK}</span>
            </h3>
            <div className="bar-chart">
                {ranked.map(({ token, prob, logit }) => (
                    <div key={token} className={`bar-row ${!topKSet.has(token) ? 'bar-row-dim' : ''}`}>
                        <span className="bar-label" style={{ width: 64 }}>{token}</span>
                        <div className="bar-track">
                            <div className="bar-fill" style={{ width: `${prob * 100}%`, background: topKSet.has(token) ? a.color : 'rgba(255,255,255,0.08)', transition: 'width 0.25s ease' }} />
                        </div>
                        <span className="bar-val">{(prob * 100).toFixed(1)}%</span>
                        <span className="bar-logit">({logit.toFixed(1)})</span>
                    </div>
                ))}
            </div>
            <p className="footnote">At temperature=1.0 the model samples — it doesn't always pick the top token. Temperature→0 always picks greedily.</p>
        </div>
    )
}