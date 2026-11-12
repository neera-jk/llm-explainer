// Turn the app's example prompts into the token IDs a model's own tokenizer
// gives them. Reads tokenizer.json (or tiktoken.model) directly instead of
// installing a tokenizer library. That is exact as long as every prompt word is
// a single token; when one isn't, it is reported, never guessed.
const HF = 'https://huggingface.co'

async function fetchFile(repo, revision, file, token) {
    const res = await fetch(`${HF}/${repo}/resolve/${revision}/${file}`, { headers: token ? { Authorization: `Bearer ${token}` } : {} })
    if (res.status === 404) return null
    if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${file} in ${repo}`)
    return res.text()
}

// Tokenizers mark the space before a word differently: byte-level BPE (GPT-2,
// Llama 3, Qwen) writes it as "Ġ" and leaves the first word bare, while
// SentencePiece-style BPE (Mistral) writes "▁" and usually marks the first word too.
function spaceRule(tok) {
    const parts = [tok.normalizer, ...(tok.normalizer?.normalizers || []), tok.pre_tokenizer, ...(tok.pre_tokenizer?.pretokenizers || [])]
        .filter(Boolean)
    const metaspace = parts.find(p => p.type === 'Metaspace')
    if (metaspace) {
        const scheme = metaspace.prepend_scheme ?? (metaspace.add_prefix_space === false ? 'never' : 'always')
        return { kind: 'sentencepiece', marker: metaspace.replacement || '▁', markFirst: scheme !== 'never' }
    }
    if (parts.some(p => p.type === 'Prepend' && p.prepend === '▁')) return { kind: 'sentencepiece', marker: '▁', markFirst: true }
    const byteLevel = parts.find(p => p.type === 'ByteLevel')
    if (byteLevel) return { kind: 'byte-level', marker: 'Ġ', markFirst: !!byteLevel.add_prefix_space }
    return null
}

// tiktoken.model lists every token as "<base64 bytes> <id>", one per line
function tiktokenIds(text) {
    const ids = new Map()
    for (const line of text.split('\n')) {
        const [bytes, id] = line.trim().split(' ')
        if (bytes && id) ids.set(bytes, Number(id))
    }
    return piece => ids.get(Buffer.from(piece, 'utf8').toString('base64'))
}

export async function loadTokenizer(repo, revision, token = process.env.HF_TOKEN) {
    const json = await fetchFile(repo, revision, 'tokenizer.json', token)
    if (json) {
        const tok = JSON.parse(json)
        const rule = spaceRule(tok)
        const vocab = tok.model?.vocab
        if (!rule || !vocab || Array.isArray(vocab)) throw new Error(`${repo} uses a tokenizer.json format the reader does not know`)
        return {
            kind: rule.kind,
            piece: (word, i) => (i === 0 && !rule.markFirst ? word : rule.marker + word),
            id: piece => vocab[piece],
        }
    }
    const tiktoken = await fetchFile(repo, revision, 'tiktoken.model', token)
    if (tiktoken) return { kind: 'tiktoken', piece: (word, i) => (i === 0 ? word : ` ${word}`), id: tiktokenIds(tiktoken) }
    throw new Error(`${repo} has no tokenizer.json or tiktoken.model to read`)
}

export function tokenizePrompts(tokenizer, prompts) {
    return Object.fromEntries(prompts.map(prompt => [prompt, prompt.split(' ').map((word, i) => {
        const id = tokenizer.id(tokenizer.piece(word, i))
        if (id === undefined) throw new Error(`"${word}" is not a single token, so a full tokenizer would be needed`)
        return { token: word, id }
    })]))
}
