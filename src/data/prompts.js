export const PROMPTS = {
    'The cat sat on': [
        { token: 'The', id: 464 },
        { token: 'cat', id: 3797 },
        { token: 'sat', id: 3332 },
        { token: 'on', id: 319 },
    ],
    'He used to live under': [
        { token: 'He', id: 1544 },
        { token: 'used', id: 973 },
        { token: 'to', id: 284 },
        { token: 'live', id: 2107 },
        { token: 'under', id: 739 },
    ],
    'Once upon a time': [
        { token: 'Once', id: 4971 },
        { token: 'upon', id: 2402 },
        { token: 'a', id: 257 },
        { token: 'time', id: 640 },
    ],
}

export const PROMPT_KEYS = Object.keys(PROMPTS)
export const DEFAULT_PROMPT = 'The cat sat on'

export const NEXT_TOKEN_PREDICTIONS = {
    'The cat sat on': [
        { token: 'mat', logit: 4.2 },
        { token: 'floor', logit: 3.8 },
        { token: 'table', logit: 3.1 },
        { token: 'ground', logit: 2.9 },
        { token: 'the', logit: 2.1 },
        { token: 'rug', logit: 1.8 },
        { token: 'carpet', logit: 1.2 },
        { token: 'grass', logit: 0.9 },
    ],
    'He used to live under': [
        { token: 'the', logit: 4.8 },
        { token: 'a', logit: 3.9 },
        { token: 'his', logit: 3.2 },
        { token: 'an', logit: 2.4 },
        { token: 'that', logit: 1.9 },
        { token: 'this', logit: 1.4 },
        { token: 'one', logit: 1.1 },
        { token: 'my', logit: 0.8 },
    ],
    'Once upon a time': [
        { token: 'there', logit: 5.1 },
        { token: ',', logit: 4.2 },
        { token: 'in', logit: 3.3 },
        { token: 'when', logit: 2.8 },
        { token: 'long', logit: 2.1 },
        { token: 'a', logit: 1.7 },
        { token: 'not', logit: 1.2 },
    ],
}