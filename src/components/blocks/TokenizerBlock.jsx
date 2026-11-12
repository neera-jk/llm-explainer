import Tooltip from '../Tooltip'

// What each kind of tokenizer does, in plain words. The kind and the vocabulary
// size come from the model's own tokenizer file, so a new model needs no new text.
const TOKENIZER_KINDS = {
    'byte-level': {
        method: 'byte-level BPE',
        footnote: (name, vocab) => `${name} uses byte-level BPE (Byte Pair Encoding) with a vocabulary of ${vocab} tokens. Text is first turned into raw bytes, so any character in any language can be written down. Then the most frequent neighboring pieces are merged into bigger tokens. Common words like "the" become a single token, while rare or long words get split into smaller pieces. A space is stored as part of the word that follows it.`,
    },
    sentencepiece: {
        method: 'SentencePiece BPE',
        footnote: (name, vocab) => `${name} uses SentencePiece-style BPE with a vocabulary of ${vocab} tokens. Like other BPE tokenizers, it merges the most frequent pieces of text into bigger tokens, so common words become one token and rare words get split. It marks each space with a visible "▁" glued to the start of the next word, so "▁cat" (a word on its own) and "cat" (inside a longer word) are different tokens.`,
    },
    tiktoken: {
        method: 'tiktoken BPE',
        footnote: (name, vocab) => `${name} uses a tiktoken-format BPE tokenizer with a vocabulary of ${vocab} tokens. tiktoken is a fast tokenizer format made popular by OpenAI. Like byte-level BPE, it works on raw bytes and merges frequent pieces, so common words are one token and rare words are split into several.`,
    },
}
const UNREAD_TOKENIZER = {
    method: 'BPE',
    footnote: (name, vocab) => `${name} has a vocabulary of ${vocab} tokens. Its tokenizer file could not be read, so the IDs for this prompt are not shown.`,
}

export default function TokenizerBlock({ model, tokens, accent }) {
    const a = accent || { color: '#888780', light: 'rgba(136,135,128,0.1)', border: 'rgba(136,135,128,0.25)', text: '#9c9a92' }

    const info = TOKENIZER_KINDS[model.tokenizer?.kind] || UNREAD_TOKENIZER

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
                            <td className="mono" style={{ color: a.text }}>{t.id ?? '?'}</td>
                        </tr>
                    ))}
                </tbody>
            </table>

            <h3 className="section-heading" style={{ color: a.color }}>as a 1D integer vector</h3>
            <div className="code-block" style={{ color: a.text }}>
                [{tokens.map(t => t.id ?? '?').join(', ')}]
                <span className="code-dim"> shape [{tokens.length}]</span>
            </div>

            <p className="footnote">
                {info.footnote(model.label, model.vocab_size.toLocaleString())}
            </p>
        </div>
    )
}