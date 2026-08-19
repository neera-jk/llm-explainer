import Tooltip from '../Tooltip'

const TOKENIZER_INFO = {
    gpt2: {
        method: 'BPE',
        footnote: `GPT-2 uses Byte Pair Encoding (BPE) with a vocabulary of 50,257 tokens, trained primarily on English web text. It starts with individual characters and keeps merging the most frequent adjacent pairs into new tokens. Common words like "the" become a single token, while rare or long words get split into smaller pieces.`,
    },
    llama: {
        method: 'BPE (SentencePiece)',
        footnote: `Llama 3 uses a 128K vocab BPE tokenizer (SentencePiece style) trained on multilingual data. The larger vocabulary means common words across many languages get their own token instead of being split, making the model much more efficient on non-English text.`,
    },
    mixtral: {
        method: 'BPE (SentencePiece)',
        footnote: `Mixtral uses a 32K vocab SentencePiece BPE tokenizer, the same lineage as Mistral. The smaller vocabulary keeps the embedding matrix compact but splits rare and non-English words into more pieces than the larger tokenizers.`,
    },
    deepseek: {
        method: 'BPE (custom)',
        footnote: `DeepSeek V3 uses a custom 129K vocab BPE tokenizer tuned for a mix of English, Chinese, and code. The large multilingual vocabulary lets it represent CJK text and programming tokens efficiently in a single token each.`,
    },
    qwen: {
        method: 'BPE (tiktoken-based)',
        footnote: `Qwen 3 uses a 152K vocab tiktoken-based BPE tokenizer with strong multilingual and code coverage. The very large vocabulary reduces how often words in Chinese, English, and other languages get split into multiple tokens.`,
    },
    llama4: {
        method: 'BPE (tiktoken-based)',
        footnote: `Llama 4 uses a 202K vocab tiktoken-based BPE tokenizer, the largest here. The huge vocabulary keeps token counts low across many languages and code, which pairs well with its very long context windows.`,
    },
}

export default function TokenizerBlock({ model, family, tokens, accent }) {
    const a = accent || { color: '#888780', light: 'rgba(136,135,128,0.1)', border: 'rgba(136,135,128,0.25)', text: '#9c9a92' }

    const fam = family || { key: 'gpt2' }
    const info = TOKENIZER_INFO[fam.key] || TOKENIZER_INFO.gpt2

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
                    { val: info.method, label: 'BPE' },
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
                {info.footnote}
            </p>
        </div>
    )
}