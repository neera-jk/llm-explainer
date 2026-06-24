import { useState } from 'react'
import { MODEL_REGISTRY, DEFAULT_MODEL, getFamily } from '../data/models'
import { PROMPTS, NEXT_TOKEN_PREDICTIONS, DEFAULT_PROMPT } from '../data/prompts'

export function useModel() {
    const [modelKey, setModelKey] = useState(DEFAULT_MODEL)
    const [promptKey, setPromptKey] = useState(DEFAULT_PROMPT)

    const model = MODEL_REGISTRY[modelKey]
    const family = getFamily(model)

    const promptData = PROMPTS[promptKey]
    const tokens = promptData[model.family] || promptData.gpt2

    const predData = NEXT_TOKEN_PREDICTIONS[promptKey]
    const predictions = predData ? (predData[model.family] || predData.gpt2) : []

    return {
        modelKey,
        setModelKey,
        model,
        family,
        promptKey,
        setPromptKey,
        tokens,
        predictions,
    }
}