import Tooltip from '../Tooltip'

export default function TokenizerBlock({ model, family, tokens, accent }) {
    const a = accent || { color: '#888780', light: 'rgba(136,135,128,0.1)', border: 'rgba(136,135,128,0.25)', text: '#9c9a92' }

    const isGPT2 = family && family.key === 'gpt2'

    return (
        <div className="block-content">
            <p className="block-explain">
                The tokenizer is the very first step. It breaks your raw text into <strong>tokens</strong>,
                which are chunks roughly the size of a word or a syllable. Each token is then mapped to a
                unique integer ID from a fixed vocabulary of <strong>{model.vocab_size.toLocaleString()} tokens</strong>.
                The model never sees your actual text. It only works with these integer IDs from this point forward.
            </p>

            <div className="stats-row">
                {[
                    { val: model.vocab_size.toLocaleString(), label: 'vocab size' },
                    { val: tokens.length, label: 'tokens in prompt' },
                    { val: 'BPE', label: 'BPE' },
                ].map(s => (
                    <div key={s.label} className="stat-card" style={{ background: a.light, borderColor: a.border }}>
                        <span className="stat-val" style={{ color: a.text }}>{s.val}</span>
                        <Tooltip term={s.label}><span className="stat-label">{s.label}</span></Tooltip>
                    </div>
                ))}
            </div>

            <h3 className="section-heading" style={{ color: a.color }}>token to ID mapping</h3>
            <table className="data-table">
                <thead><tr><th>position</th><th>token</th><th>ID</th></tr></thead>
                <tbody>
                    {tokens.map((t, i) => (
                        <tr key={i}>
                            <td className="muted">{i}</td>
                            <td>
                                <span className="token-pill" style={{ background: a.light, border: `1px solid ${a.border}`, color: a.text }}>
                                    {t.token}
                                </span>
                            </td>
                            <td className="mono" style={{ color: a.text }}>{t.id}</td>
                        </tr>
                    ))}
                </tbody>
            </table>

            <h3 className="section-heading" style={{ color: a.color }}>as a 1D integer vector</h3>
            <div className="code-block" style={{ color: a.text }}>
                [{tokens.map(t => t.id).join(', ')}]
                <span className="code-dim"> shape [{tokens.length}]</span>
            </div>

            <p className="footnote">
                {isGPT2
                    ? `${model.label} uses Byte Pair Encoding (BPE) with a vocabulary of ${model.vocab_size.toLocaleString()} tokens, trained primarily on English web text. It starts with individual characters and keeps merging the most frequent adjacent pairs into new tokens. Common words like "the" become a single token, while rare or long words get split into smaller pieces.`
                    : `${model.label} uses Byte Pair Encoding (BPE) with a much larger vocabulary of ${model.vocab_size.toLocaleString()} tokens, trained on multilingual data. The larger vocabulary means common words in many languages (not just English) get their own token instead of being split into many small pieces. This makes the model significantly more efficient at non-English text, at the cost of a larger embedding matrix.`
                }
            </p>
        </div>
    )
}