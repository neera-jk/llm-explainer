import { useRef, useEffect, useState } from 'react'

export const GLOSSARY = {
    layers: 'Transformer blocks stacked on top of each other. Each layer refines the model\'s understanding. More layers means deeper reasoning, but also slower and more memory.',
    dims: 'Short for dimensions. Each token is represented as a list of numbers (a vector). This is how many numbers are in that list. Think of it as the "resolution" of the model\'s understanding.',
    heads: 'Attention heads run in parallel inside each layer. Each head learns to focus on different relationships between words (grammar, meaning, position, etc). Their results get combined at the end.',
    vocab: 'The total number of unique tokens the model recognizes. This includes common words, parts of words, punctuation, and special tokens. Anything not in the vocabulary gets split into smaller pieces.',
    'head dim': 'The size of the vectors each attention head works with. Calculated by dividing the model dimension by the number of heads. Smaller heads are cheaper to compute.',
    'FFN expansion': 'How much wider the feed-forward network gets compared to the model dimension. If the model is 768-wide and expansion is 4x, the hidden layer is 3072-wide. This extra room lets the model learn richer patterns.',
    'embedding params': 'The total count of learnable numbers in the embedding table. Calculated as vocab size multiplied by the model dimension. These numbers are adjusted during training.',
    'output shape': 'The dimensions of the data after this step. For example, [4x768] means 4 tokens, each represented by 768 numbers.',
    'vocab size': 'How many unique tokens the model knows. Built during training using byte-pair encoding, which merges common character pairs into single tokens.',
    'tokens in prompt': 'The number of pieces the tokenizer split your input text into. Common words are usually one token. Rare or long words get split into multiple tokens.',
    'BPE': 'Byte Pair Encoding. A method that builds a vocabulary by starting with individual characters and repeatedly merging the most common adjacent pairs into new tokens.',
    'output logits': 'A raw score for every token in the vocabulary. These are not probabilities yet. A higher score means the model thinks that token is more likely to come next. Softmax converts these into actual probabilities.',
    'unembed params': 'The count of learnable numbers in the unembedding matrix. This matrix converts the model\'s internal representation back into predictions over the vocabulary.',
    'output per position': 'The output shape for a single token position. The model produces one score for every token in the vocabulary, so this is [1 x vocab_size].',
    'top token prob': 'The probability assigned to the single most likely next token, after softmax converts logits into probabilities. Higher means the model is more confident.',
    'most likely next': 'The token with the highest probability. This is what the model would output if using greedy decoding (always picking the top choice).',
    'temperature': 'A knob that controls randomness. Low values (like 0.1) make the model nearly always pick the top token. High values (like 2.0) spread probability more evenly, making output more random and creative.',
    'top-k': 'Limits sampling to only the K most likely tokens. Everything else gets zero probability. This prevents the model from picking extremely unlikely tokens that would make the output incoherent.',
    'dims per token': 'Each token is turned into a list of this many numbers. These numbers together encode the meaning, position, and context of the token.',
    'attn heads': 'Attention heads that run in parallel inside each layer. Each one learns different word relationships. Their outputs are combined into a single result.',
}

export default function Tooltip({ term, children }) {
    const definition = GLOSSARY[term.toLowerCase()] || GLOSSARY[term]
    const ref = useRef(null)
    const [edge, setEdge] = useState('')

    useEffect(() => {
        if (!ref.current) return
        function check() {
            const rect = ref.current.getBoundingClientRect()
            const vw = window.innerWidth
            if (rect.right + 110 > vw) {
                setEdge('tip-wrap--right')
            } else if (rect.left - 110 < 0) {
                setEdge('tip-wrap--left')
            } else {
                setEdge('')
            }
        }
        check()
        window.addEventListener('resize', check)
        return () => window.removeEventListener('resize', check)
    }, [])

    if (!definition) return children || term

    return (
        <span className={`tip-wrap ${edge}`} tabIndex={0} ref={ref}>
            {children || term}
            <span className="tip-box">{definition}</span>
        </span>
    )
}