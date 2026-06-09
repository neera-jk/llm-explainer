import { useState } from 'react'
import { MODEL_REGISTRY, DEFAULT_MODEL } from '../data/models'
import { PROMPTS, DEFAULT_PROMPT } from '../data/prompts'

/**
 * Central state for selected model and prompt.
 * All components that need model/token data consume this hook.
 */
export function useModel() {
    const [modelKey, setModelKey] = useState(DEFAULT_MODEL)
    const [promptKey, setPromptKey] = useState(DEFAULT_PROMPT)

    return {
        modelKey,
        setModelKey,
        model: MODEL_REGISTRY[modelKey],
        promptKey,
        setPromptKey,
        tokens: PROMPTS[promptKey],
    }
}