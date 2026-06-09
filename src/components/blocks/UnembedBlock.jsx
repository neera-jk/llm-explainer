import { NEXT_TOKEN_PREDICTIONS } from '../../data/prompts'

/**
 * Unembed block detail.
 * Shows how the final transformer output is projected to a logit
 * score for every token in the vocabulary.
 */
export default function UnembedBlock({ model, tokens, promptKey }) {
    const predictions = NEXT_TOKEN_PREDICTIONS[promptKey] || []
    const T = tokens.length
    const maxLogit = Math.max(...predictions.map(p => p.logit))

    return (
        <div className="block-content">
            <p className="block-explain">
                After the final transformer layer we have <strong>[{T} × {model.d_model}]</strong>.
                A linear layer multiplies the <strong>last token's vector</strong> by
                <strong> [{model.d_model} × {model.vocab_size.toLocaleString()}]</strong> to
                produce a raw score — a <strong>logit</strong> — for every token in the vocabulary.
                Higher logit means more likely next word.
            </p>

            <div className="stats-row">
                <div className="stat-card">
                    <span className="stat-val">{model.vocab_size.toLocaleString()}</span>
                    <span className="stat-label">output logits</span>
                </div>
                <div className="stat-card">
                    <span className="stat-val">{(model.d_model * model.vocab_size / 1e6).toFixed(0)}M</span>
                    <span className="stat-label">unembed params</span>
                </div>
                <div className="stat-card">
                    <span className="stat-val">[1×{model.vocab_size.toLocaleString()}]</span>
                    <span className="stat-label">output per position</span>
                </div>
            </div>

            <h3 className="section-heading">the matmul</h3>
            <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start', marginBottom: '1rem' }}>
                <span className="layer-num-badge">1</span>
                <div>
                    <div style={{ fontSize: 13, fontWeight: 500, color: '#3d3d3a', marginBottom: 6 }}>
                        project last token position to vocab size
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap', marginBottom: 6 }}>
                        <span className="shape-pill purple">[1×{model.d_model}]</span>
                        <span className="shape-arrow">×</span>
                        <span className="shape-pill gray">W_unembed [{model.d_model}×{model.vocab_size.toLocaleString()}]</span>
                        <span className="shape-arrow">=</span>
                        <span className="shape-pill coral">[1×{model.vocab_size.toLocaleString()}]</span>
                    </div>
                    <p style={{ fontSize: 12, color: '#73726c', lineHeight: 1.6 }}>
                        Only position {T - 1} (the last token) predicts the next word.
                        The result is {model.vocab_size.toLocaleString()} raw scores — one per vocab token.
                    </p>
                </div>
            </div>

            <h3 className="section-heading">
                top logits after "{tokens[T - 1].token}"
                <span className="section-note">raw scores, not probabilities yet</span>
            </h3>
            <div className="bar-chart">
                {predictions.map(({ token, logit }) => (
                    <div key={token} className="bar-row">
                        <span className="bar-label" style={{ width: 64 }}>{token}</span>
                        <div className="bar-track">
                            <div className="bar-fill" style={{ width: `${(logit / maxLogit) * 100}%`, background: '#D85A30' }} />
                        </div>
                        <span className="bar-val">{logit.toFixed(1)}</span>
                    </div>
                ))}
            </div>

            <p className="footnote">
                These are logits, not probabilities. Softmax in the next block converts them
                to actual probabilities that sum to 1.
            </p>
        </div>
    )
}