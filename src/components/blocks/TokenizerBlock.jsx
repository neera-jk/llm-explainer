/**
 * Tokenizer block detail.
 * Shows how raw text is split into tokens and mapped to integer IDs.
 */
export default function TokenizerBlock({ model, tokens }) {
    return (
        <div className="block-content">
            <p className="block-explain">
                The tokenizer breaks raw text into <strong>tokens</strong> — chunks roughly the
                size of a word or syllable. Each token maps to a unique integer ID from a fixed
                vocabulary of <strong>{model.vocab_size.toLocaleString()} tokens</strong>.
                The model never sees raw text — only these integer IDs.
            </p>

            <div className="stats-row">
                <div className="stat-card">
                    <span className="stat-val">{model.vocab_size.toLocaleString()}</span>
                    <span className="stat-label">vocab size</span>
                </div>
                <div className="stat-card">
                    <span className="stat-val">{tokens.length}</span>
                    <span className="stat-label">tokens in prompt</span>
                </div>
                <div className="stat-card">
                    <span className="stat-val">BPE</span>
                    <span className="stat-label">tokenization method</span>
                </div>
            </div>

            <h3 className="section-heading">token → ID mapping</h3>
            <table className="data-table">
                <thead>
                    <tr>
                        <th>position</th>
                        <th>token</th>
                        <th>ID</th>
                    </tr>
                </thead>
                <tbody>
                    {tokens.map((t, i) => (
                        <tr key={i}>
                            <td className="muted">{i}</td>
                            <td><span className="token-pill">{t.token}</span></td>
                            <td className="mono accent">{t.id}</td>
                        </tr>
                    ))}
                </tbody>
            </table>

            <h3 className="section-heading">as a 1D integer vector</h3>
            <div className="code-block">
                [{tokens.map(t => t.id).join(', ')}]
                <span className="code-dim"> — shape [{tokens.length}]</span>
            </div>

            <p className="footnote">
                Byte Pair Encoding (BPE) starts with individual characters and repeatedly merges
                the most frequent adjacent pairs. Common words like "the" get a single token.
                Rare words split into subwords — "unbelievable" might become "un" + "believ" + "able".
            </p>
        </div>
    )
}