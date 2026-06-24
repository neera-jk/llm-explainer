import { useState } from 'react'
import { softmax } from '../../data/models'
import Tooltip from '../Tooltip'

export default function SoftmaxBlock({ model, family, tokens, predictions, accent }) {
    const a = accent || { color: '#1a8a65', light: 'rgba(26,138,101,0.1)', border: 'rgba(26,138,101,0.3)', text: '#4db896' }
    const [temperature, setTemperature] = useState(1.0)
    const [topK, setTopK] = useState(5)

    const preds = predictions || []
    const probs = softmax(preds.map(p => p.logit), temperature)
    const ranked = preds.map((p, i) => ({ ...p, prob: probs[i] })).sort((x, y) => y.prob - x.prob)
    const topKSet = new Set(ranked.slice(0, topK).map(p => p.token))
    const topToken = ranked[0]

    function tempNote() {
        if (temperature < 0.7) return 'Very low: the model will almost always pick the single most likely token. Output becomes very predictable and repetitive.'
        if (temperature <= 1.3) return 'Normal range: the model balances coherence with some variety. This is where most applications operate.'
        if (temperature <= 2.0) return 'High: the model becomes more creative and surprising, but also less coherent. Good for brainstorming, risky for factual tasks.'
        return 'Very high: probabilities are nearly uniform across all tokens. Output is essentially random at this point.'
    }

    function tempZone() {
        if (temperature < 0.4) return 'greedy'
        if (temperature <= 1.3) return 'normal'
        if (temperature <= 2.0) return 'creative'
        return 'random'
    }

    const tempPercent = Math.min(((temperature - 0.1) / 2.9) * 100, 100)

    return (
        <div className="block-content">
            <p className="block-explain">
                The previous step produced {model.vocab_size.toLocaleString()} raw scores (logits), one for every token
                in {model.label}'s vocabulary. But those scores can be any number and they do not add up to
                anything meaningful. Softmax fixes this by converting them into <strong>probabilities
                    that always sum to 1.0</strong> (or 100%).
                {topToken ? ` A logit of ${topToken.logit.toFixed(1)} for "${topToken.token}" becomes a ${(topToken.prob * 100).toFixed(0)}% probability.` : ''}
                {' '}Two controls let you adjust how the model picks its next token: <strong>temperature</strong> scales
                the logits before softmax (lower = more confident, higher = more random),
                and <strong>top-k</strong> limits sampling to only the K most likely tokens.
            </p>

            <div className="stats-row">
                {[
                    { val: topToken ? `${(topToken.prob * 100).toFixed(1)}%` : '-', label: 'top token prob' },
                    { val: topToken?.token || '-', label: 'most likely next' },
                    { val: temperature.toFixed(1), label: 'temperature' },
                ].map(s => (
                    <div key={s.label} className="stat-card" style={{ background: a.light, borderColor: a.border }}>
                        <span className="stat-val" style={{ color: a.text }}>{s.val}</span>
                        <Tooltip term={s.label}><span className="stat-label">{s.label}</span></Tooltip>
                    </div>
                ))}
            </div>

            <div className="controls">
                <div className="control-row">
                    <label className="control-label">temperature</label>
                    <div className="pill-slider">
                        <div className="pill-track">
                            <input type="range" min="1" max="30" step="1"
                                value={Math.round(temperature * 10)}
                                onChange={e => setTemperature(parseInt(e.target.value) / 10)}
                                className="pill-input" />
                            <div className="pill-fill" style={{
                                width: `${Math.max(tempPercent, 8)}%`,
                                background: 'rgba(26,138,101,0.15)',
                            }}>
                                <span className="pill-thumb">{temperature.toFixed(1)}</span>
                            </div>
                        </div>
                        <div className="pill-zones">
                            <span className={`pill-zone ${tempZone() === 'greedy' ? 'pill-zone-active' : ''}`}>greedy</span>
                            <span className={`pill-zone ${tempZone() === 'normal' ? 'pill-zone-active' : ''}`}>normal</span>
                            <span className={`pill-zone ${tempZone() === 'creative' ? 'pill-zone-active' : ''}`}>creative</span>
                            <span className={`pill-zone ${tempZone() === 'random' ? 'pill-zone-active' : ''}`}>random</span>
                        </div>
                    </div>
                </div>

                <div className="control-row">
                    <label className="control-label">top-k</label>
                    <div className="step-slider">
                        <div className="step-track">
                            <input type="range" min="1" max={preds.length || 1} step="1"
                                value={topK}
                                onChange={e => setTopK(parseInt(e.target.value))}
                                className="step-input" />
                            {Array.from({ length: preds.length }, (_, i) => (
                                <div key={i}
                                    className={`step-seg ${i < topK ? 'step-seg-on' : ''}`}
                                    style={i < topK ? { background: a.color, borderColor: 'transparent' } : {}}
                                />
                            ))}
                        </div>
                        <div className="step-labels">
                            <span className="step-label">1</span>
                            <span className="step-label">{Math.ceil(preds.length / 2)}</span>
                            <span className="step-label">{preds.length}</span>
                        </div>
                    </div>
                    <span className="control-val">{topK}</span>
                </div>

                <p className="controls-hint">drag the sliders to see how temperature and top-k affect the probabilities below</p>
            </div>

            <div className="temp-note" style={{ borderColor: a.border, color: a.text }}>{tempNote()}</div>

            <h3 className="section-heading" style={{ color: a.color }}>
                probability distribution <span className="section-note">greyed out tokens are excluded by top-{topK}</span>
            </h3>
            <div className="bar-chart">
                {ranked.map(({ token, prob, logit }) => (
                    <div key={token} className={`bar-row ${!topKSet.has(token) ? 'bar-row-dim' : ''}`}>
                        <span className="bar-label" style={{ width: 64 }}>{token}</span>
                        <div className="bar-track">
                            <div className="bar-fill" style={{
                                width: `${prob * 100}%`,
                                background: topKSet.has(token) ? a.color : 'rgba(255,255,255,0.06)',
                                transition: 'width 0.3s cubic-bezier(0.16, 1, 0.3, 1)',
                            }} />
                        </div>
                        <span className="bar-val">{(prob * 100).toFixed(1)}%</span>
                        <span className="bar-logit">({logit.toFixed(1)})</span>
                    </div>
                ))}
            </div>

            <p className="footnote">
                At temperature 1.0, {model.label} randomly samples from this distribution, so it will not
                always pick the top token. Lowering temperature toward 0 makes the model always pick the
                most likely token (greedy decoding). Raising it makes the distribution flatter, giving
                less likely tokens a better chance of being selected.
            </p>
        </div>
    )
}