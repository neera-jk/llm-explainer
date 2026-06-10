import { NEXT_TOKEN_PREDICTIONS } from '../../data/prompts'
import Tooltip from '../Tooltip'

export default function UnembedBlock({ model, tokens, promptKey, accent }) {
    const a = accent || { color: '#b85428', light: 'rgba(184,84,40,0.1)', border: 'rgba(184,84,40,0.3)', text: '#d4896b' }
    const predictions = NEXT_TOKEN_PREDICTIONS[promptKey] || []
    const T = tokens.length
    const maxLogit = Math.max(...predictions.map(p => p.logit))

    return (
        <div className="block-content">
            <p className="block-explain">
                After passing through all the transformer layers, we have a <strong>[{T} x {model.d_model}]</strong> tensor.
                The model now needs to convert its internal representation back into a prediction over
                actual words. It does this by multiplying the <strong>last token's vector</strong> by
                a large matrix <strong>[{model.d_model} x {model.vocab_size.toLocaleString()}]</strong>.
                This produces one number (called a <strong>logit</strong>) for every token in the vocabulary.
                A logit is just a raw score: the higher the number, the more likely the model thinks
                that token should come next.
            </p>

            <div className="stats-row">
                {[
                    { val: model.vocab_size.toLocaleString(), label: 'output logits' },
                    { val: `${(model.d_model * model.vocab_size / 1e6).toFixed(0)}M`, label: 'unembed params' },
                    { val: `[1x${model.vocab_size.toLocaleString()}]`, label: 'output per position' },
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
                        <div className="bar-track">
                            <div className="bar-fill" style={{ width: `${(logit / maxLogit) * 100}%`, background: a.color }} />
                        </div>
                        <span className="bar-val">{logit.toFixed(1)}</span>
                    </div>
                ))}
            </div>

            <p className="footnote">
                Important: these logits are not probabilities. They can be any number, and they do not
                add up to 1. The next step (Softmax) will convert them into actual probabilities. Right now
                "mat" has the highest logit of 4.2, which means the model thinks "mat" is the most likely
                next word after "{tokens[T - 1].token}".
            </p>
        </div>
    )
}