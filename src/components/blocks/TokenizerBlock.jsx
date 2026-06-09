export default function TokenizerBlock({ model, tokens, accent }) {
    const a = accent || { color: '#888780', light: 'rgba(136,135,128,0.1)', border: 'rgba(136,135,128,0.25)', text: '#9c9a92' }

    return (
        <div className="block-content">
            <p className="block-explain">
                The tokenizer breaks raw text into <strong>tokens</strong> — chunks roughly the
                size of a word or syllable. Each token maps to a unique integer ID from a fixed
                vocabulary of <strong>{model.vocab_size.toLocaleString()} tokens</strong>.
                The model never sees raw text — only these integer IDs.
            </p>
            <div className="stats-row">
                {[
                    { val: model.vocab_size.toLocaleString(), label: 'vocab size' },
                    { val: tokens.length, label: 'tokens in prompt' },
                    { val: 'BPE', label: 'tokenization method' },
                ].map(s => (
                    <div key={s.label} className="stat-card" style={{ background: a.light, borderColor: a.border }}>
                        <span className="stat-val" style={{ color: a.text }}>{s.val}</span>
                        <span className="stat-label">{s.label}</span>
                    </div>
                ))}
            </div>
            <h3 className="section-heading" style={{ color: a.color }}>token → ID mapping</h3>
            <table className="data-table">
                <thead><tr><th>position</th><th>token</th><th>ID</th></tr></thead>
                <tbody>
                    {tokens.map((t, i) => (
                        <tr key={i}>
                            <td className="muted">{i}</td>
                            <td><span className="token-pill" style={{ background: a.light, border: `0.5px solid ${a.border}`, color: a.text }}>{t.token}</span></td>
                            <td className="mono" style={{ color: a.text }}>{t.id}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
            <h3 className="section-heading" style={{ color: a.color }}>as a 1D integer vector</h3>
            <div className="code-block" style={{ color: a.text }}>
                [{tokens.map(t => t.id).join(', ')}]
                <span className="code-dim"> — shape [{tokens.length}]</span>
            </div>
            <p className="footnote">
                GPT-2 uses Byte Pair Encoding (BPE) — starts with individual characters and
                merges the most frequent pairs. Common words like "the" get one token.
                Rare words split into subwords.
            </p>
        </div>
    )
}