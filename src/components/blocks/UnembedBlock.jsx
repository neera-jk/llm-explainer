import { NEXT_TOKEN_PREDICTIONS } from '../../data/prompts'
import Tooltip from '../Tooltip'

export default function UnembedBlock({ model, tokens, promptKey, accent }) {
    const a = accent || { color: '#D85A30', light: 'rgba(216,90,48,0.1)', border: 'rgba(216,90,48,0.3)', text: '#F0997B' }
    const predictions = NEXT_TOKEN_PREDICTIONS[promptKey] || []
    const T = tokens.length
    const maxLogit = Math.max(...predictions.map(p => p.logit))

    return (
        <div className="block-content">
            <p className="block-explain">
                After the final transformer layer we have <strong>[{T} × {model.d_model}]</strong>.
                A linear layer multiplies the <strong>last token's vector</strong> by
                <strong> [{model.d_model} × {model.vocab_size.toLocaleString()}]</strong> to
                produce a raw score — a <strong>logit</strong> — for every vocab token.
                Higher logit = more likely next word.
            </p>
            <div className="stats-row">
                {[
                    { val: model.vocab_size.toLocaleString(), label: 'output logits' },
                    { val: `${(model.d_model * model.vocab_size / 1e6).toFixed(0)}M`, label: 'unembed params' },
                    { val: `[1×${model.vocab_size.toLocaleString()}]`, label: 'output per position' },
                ].map(s => (
                    <div key={s.label} className="stat-card" style={{ background: a.light, borderColor: a.border }}>
                        <span className="stat-val" style={{ color: a.text }}>{s.val}</span>
                        <Tooltip term={s.label}><span className="stat-label">{s.label}</span></Tooltip>
                    </div>
                ))}
            </div>
            <h3 className="section-heading" style={{ color: a.color }}>
                top logits after "{tokens[T - 1].token}"
                <span className="section-note">raw scores, not probabilities yet</span>
            </h3>
            <div className="bar-chart">
                {predictions.map(({ token, logit }) => (
                    <div key={token} className="bar-row">
                        <span className="bar-label" style={{ width: 64 }}>{token}</span>
                        <div className="bar-track"><div className="bar-fill" style={{ width: `${(logit / maxLogit) * 100}%`, background: a.color }} /></div>
                        <span className="bar-val">{logit.toFixed(1)}</span>
                    </div>
                ))}
            </div>
            <p className="footnote">These are logits, not probabilities. Softmax in the next block converts them to actual probabilities that sum to 1.</p>
        </div>
    )
}