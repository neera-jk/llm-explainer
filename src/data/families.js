// A family is only the dropdown group (and prompt tokenizer). Architecture
// features come from each model's own config. Kept apart from models.js so
// Node scripts can import it without Vite.
export const MODEL_FAMILIES = {
    gpt2: { key: 'gpt2', label: 'GPT-2 family' },
    llama: { key: 'llama', label: 'Llama 3 family' },
    mixtral: { key: 'mixtral', label: 'Mixtral family' },
    deepseek: { key: 'deepseek', label: 'DeepSeek family' },
    qwen: { key: 'qwen', label: 'Qwen 3 family' },
    llama4: { key: 'llama4', label: 'Llama 4 family' },
    mistral: { key: 'mistral', label: 'Mistral family' },
}
