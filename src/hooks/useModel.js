import { useState } from 'react'
import { MODEL_REGISTRY, DEFAULT_MODEL, getFamily } from '../data/models'
import { PROMPTS, DEFAULT_PROMPT } from '../data/prompts'

export function useModel() {
    const [modelKey, setModelKey] = useState(DEFAULT_MODEL)
    const [promptKey, setPromptKey] = useState(DEFAULT_PROMPT)

    const model = MODEL_REGISTRY[modelKey]
    const family = getFamily(model)

    return {
        modelKey,
        setModelKey,
        model,
        family,
        promptKey,
        setPromptKey,
        tokens: PROMPTS[promptKey],
    }
}