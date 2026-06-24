export const PROMPTS = {
    'The cat sat on': {
        gpt2: [
            { token: 'The', id: 464 },
            { token: 'cat', id: 3797 },
            { token: 'sat', id: 3332 },
            { token: 'on', id: 319 },
        ],
        llama: [
            { token: 'The', id: 791 },
            { token: 'cat', id: 8415 },
            { token: 'sat', id: 7731 },
            { token: 'on', id: 389 },
        ],
        moe: [
            { token: 'The', id: 791 },
            { token: 'cat', id: 8415 },
            { token: 'sat', id: 7731 },
            { token: 'on', id: 389 },
        ],
    },
    'He used to live under': {
        gpt2: [
            { token: 'He', id: 1544 },
            { token: 'used', id: 973 },
            { token: 'to', id: 284 },
            { token: 'live', id: 2107 },
            { token: 'under', id: 739 },
        ],
        llama: [
            { token: 'He', id: 940 },
            { token: 'used', id: 1304 },
            { token: 'to', id: 311 },
            { token: 'live', id: 3974 },
            { token: 'under', id: 1234 },
        ],
        moe: [
            { token: 'He', id: 940 },
            { token: 'used', id: 1304 },
            { token: 'to', id: 311 },
            { token: 'live', id: 3974 },
            { token: 'under', id: 1234 },
        ],
    },
    'Once upon a time': {
        gpt2: [
            { token: 'Once', id: 4971 },
            { token: 'upon', id: 2402 },
            { token: 'a', id: 257 },
            { token: 'time', id: 640 },
        ],
        llama: [
            { token: 'Once', id: 12805 },
            { token: 'upon', id: 2501 },
            { token: 'a', id: 264 },
            { token: 'time', id: 892 },
        ],
        moe: [
            { token: 'Once', id: 12805 },
            { token: 'upon', id: 2501 },
            { token: 'a', id: 264 },
            { token: 'time', id: 892 },
        ],
    },
}

export const PROMPT_KEYS = Object.keys(PROMPTS)
export const DEFAULT_PROMPT = 'The cat sat on'

export const NEXT_TOKEN_PREDICTIONS = {
    'The cat sat on': {
        gpt2: [
            { token: 'mat', logit: 4.2 },
            { token: 'floor', logit: 3.8 },
            { token: 'table', logit: 3.1 },
            { token: 'ground', logit: 2.9 },
            { token: 'the', logit: 2.1 },
            { token: 'rug', logit: 1.8 },
            { token: 'carpet', logit: 1.2 },
            { token: 'grass', logit: 0.9 },
        ],
        llama: [
            { token: 'the', logit: 5.1 },
            { token: 'mat', logit: 3.9 },
            { token: 'a', logit: 3.4 },
            { token: 'his', logit: 2.8 },
            { token: 'her', logit: 2.3 },
            { token: 'my', logit: 1.9 },
            { token: 'top', logit: 1.4 },
            { token: 'its', logit: 0.8 },
        ],
        moe: [
            { token: 'the', logit: 5.3 },
            { token: 'mat', logit: 3.7 },
            { token: 'a', logit: 3.5 },
            { token: 'his', logit: 2.6 },
            { token: 'her', logit: 2.4 },
            { token: 'my', logit: 1.7 },
            { token: 'top', logit: 1.3 },
            { token: 'its', logit: 0.7 },
        ],
    },
    'He used to live under': {
        gpt2: [
            { token: 'the', logit: 4.8 },
            { token: 'a', logit: 3.9 },
            { token: 'his', logit: 3.2 },
            { token: 'an', logit: 2.4 },
            { token: 'that', logit: 1.9 },
            { token: 'this', logit: 1.4 },
            { token: 'one', logit: 1.1 },
            { token: 'my', logit: 0.8 },
        ],
        llama: [
            { token: 'the', logit: 5.2 },
            { token: 'a', logit: 4.1 },
            { token: 'his', logit: 3.0 },
            { token: 'an', logit: 2.6 },
            { token: 'one', logit: 1.8 },
            { token: 'strict', logit: 1.3 },
            { token: 'that', logit: 1.0 },
            { token: 'constant', logit: 0.6 },
        ],
        moe: [
            { token: 'the', logit: 5.4 },
            { token: 'a', logit: 4.0 },
            { token: 'his', logit: 3.1 },
            { token: 'an', logit: 2.5 },
            { token: 'one', logit: 1.9 },
            { token: 'strict', logit: 1.2 },
            { token: 'that', logit: 1.0 },
            { token: 'constant', logit: 0.5 },
        ],
    },
    'Once upon a time': {
        gpt2: [
            { token: 'there', logit: 5.1 },
            { token: ',', logit: 4.2 },
            { token: 'in', logit: 3.3 },
            { token: 'when', logit: 2.8 },
            { token: 'long', logit: 2.1 },
            { token: 'a', logit: 1.7 },
            { token: 'not', logit: 1.2 },
        ],
        llama: [
            { token: ',', logit: 5.6 },
            { token: 'there', logit: 4.8 },
            { token: 'in', logit: 3.1 },
            { token: 'when', logit: 2.5 },
            { token: 'long', logit: 2.0 },
            { token: 'not', logit: 1.3 },
            { token: 'the', logit: 0.9 },
        ],
        moe: [
            { token: ',', logit: 5.8 },
            { token: 'there', logit: 4.6 },
            { token: 'in', logit: 3.2 },
            { token: 'when', logit: 2.4 },
            { token: 'long', logit: 1.9 },
            { token: 'not', logit: 1.4 },
            { token: 'the', logit: 0.8 },
        ],
    },
}