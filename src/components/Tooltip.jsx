import { useRef, useEffect, useState } from 'react'

export const GLOSSARY = {
    layers: 'The number of transformer blocks stacked on top of each other. More layers = deeper reasoning, but slower and more memory.',
    dims: 'The size of each token\'s vector representation (d_model). Every token is a list of this many numbers as it flows through the model.',
    heads: 'Attention heads. Each head learns a different pattern of which tokens to focus on. They run in parallel, then their results are combined.',
    vocab: 'The total number of unique tokens the model knows. Includes words, subwords, and special characters.',
    'head dim': 'The size of Q, K, V vectors inside each attention head. Usually d_model divided by n_heads.',
    'FFN expansion': 'How much wider the feed-forward hidden layer is compared to d_model. A 4x expansion means 768 becomes 3072.',
    'embedding params': 'Number of learnable numbers in the embedding matrix. vocab_size times d_model.',
    'output shape': 'The tensor dimensions after this step. [tokens x dims] means each token has a vector of that many dimensions.',
    'vocab size': 'Total unique tokens in the model\'s dictionary, built during training using byte-pair encoding.',
    'tokens in prompt': 'How many tokens the tokenizer split your input text into.',
    'BPE': 'Byte Pair Encoding. Starts with single characters and repeatedly merges the most common adjacent pairs into new tokens.',
    'output logits': 'One raw score for every token in the vocabulary. Higher score = model thinks that token is more likely to come next.',
    'unembed params': 'Learnable numbers in the unembedding matrix. d_model times vocab_size.',
    'output per position': 'The shape of the output for a single token position after the unembedding projection.',
    'top token prob': 'The probability of the single most likely next token after softmax.',
    'most likely next': 'The token with the highest probability after softmax. What the model would pick with greedy decoding.',
    'temperature': 'Controls randomness. Low (0.1) = almost always picks the top token. High (2.0) = very random output.',
    'top-k': 'Only sample from the K most likely tokens. The rest get zero probability. Keeps output from going off the rails.',
    'dims per token': 'Each token is represented as a vector with this many numbers. These numbers encode the token\'s meaning.',
    'attn heads': 'Attention heads that run in parallel inside each layer. Each learns different token relationships.',
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