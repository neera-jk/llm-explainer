// Reading and writing the app's model files, shared by the importer and the watcher.
import { readdir, readFile, writeFile } from 'node:fs/promises'

export const MODELS_DIR = new URL('../../src/data/models/', import.meta.url)
export const FAMILIES_FILE = new URL('../../src/data/families.js', import.meta.url)
// written by hand; everything else in a model file is regenerated from HF
export const HAND_FIELDS = ['key', 'label', 'family', 'params']

export async function loadModels() {
    const files = (await readdir(MODELS_DIR)).filter(f => f.endsWith('.json')).sort()
    return Promise.all(files.map(async file => {
        const url = new URL(file, MODELS_DIR)
        return { file, url, model: JSON.parse(await readFile(url, 'utf8')) }
    }))
}

export async function writeModel(model, dir = MODELS_DIR) {
    await writeFile(new URL(`${model.key}.json`, dir), JSON.stringify(model, null, 4) + '\n')
}

function flatten(obj, prefix = '') {
    return Object.entries(obj).flatMap(([k, v]) =>
        v && typeof v === 'object' && !Array.isArray(v)
            ? flatten(v, `${prefix}${k}.`)
            : [[`${prefix}${k}`, JSON.stringify(v)]])
}

// a new upstream commit on its own isn't a difference, so source is skipped
export function generatedFields(model) {
    const rest = { ...model }
    for (const f of ['source', ...HAND_FIELDS]) delete rest[f]
    return Object.fromEntries(flatten(rest))
}

// two repos with the same fingerprint are the same design (e.g. Base and Instruct)
export function fingerprint(model) {
    return JSON.stringify(Object.entries(generatedFields(model)).sort())
}

export function slug(text) {
    return text.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '')
}

// "Qwen/Qwen3-14B-Base" -> "Qwen3 14B"; variant words are dropped wherever they
// appear, since Base and Instruct share a design. The reviewer can still rename it.
export function labelFromRepo(repo) {
    return repo.split('/')[1].replace(/[-_](base|instruct|chat|it|hf)(?=[-_]|$)/gi, '').replace(/[-_]+/g, ' ')
}

export function formatParams(total) {
    if (!total) return '?'
    const [value, unit] = total >= 1e12 ? [total / 1e12, 'T'] : total >= 1e9 ? [total / 1e9, 'B'] : [total / 1e6, 'M']
    return `${value >= 100 ? Math.round(value) : +value.toFixed(1)}${unit}`
}

// Which existing dropdown group a company's architecture belongs to. Anything
// else gets a new group named after the company.
const FAMILY_BY_ORG = {
    'openai-community': { gpt2: 'gpt2' },
    'meta-llama': { llama: 'llama', llama4: 'llama4' },
    mistralai: { mixtral: 'mixtral' },
    'deepseek-ai': { deepseek_v2: 'deepseek', deepseek_v3: 'deepseek' },
    moonshotai: { kimi_k2: 'deepseek' },
    Qwen: { qwen3: 'qwen', qwen3_moe: 'qwen' },
}
const ORG_NAMES = {
    'openai-community': 'GPT-2', 'meta-llama': 'Llama', mistralai: 'Mistral', 'deepseek-ai': 'DeepSeek',
    moonshotai: 'Kimi', Qwen: 'Qwen', google: 'Google', microsoft: 'Microsoft', openai: 'OpenAI',
    'zai-org': 'GLM', baidu: 'ERNIE', MiniMaxAI: 'MiniMax', XiaomiMiMo: 'MiMo', 'meituan-longcat': 'LongCat',
    nvidia: 'NVIDIA', tencent: 'Tencent', 'ibm-granite': 'Granite', internlm: 'InternLM', 'ByteDance-Seed': 'ByteDance Seed',
    LiquidAI: 'Liquid', inclusionAI: 'Ant Ling', sarvamai: 'Sarvam', tiiuae: 'Falcon', 'LGAI-EXAONE': 'EXAONE',
    upstage: 'Solar', 'stepfun-ai': 'Step', 'ServiceNow-AI': 'Apriel', 'krutrim-ai-labs': 'Krutrim', ai4bharat: 'AI4Bharat',
    allenai: 'OLMo', HuggingFaceTB: 'SmolLM', 'swiss-ai': 'Apertus', CohereLabs: 'Cohere',
}

export function classifyFamily(modelType, repo, families) {
    const org = repo.split('/')[0]
    const known = FAMILY_BY_ORG[org]?.[modelType]
    if (known) return { key: known, label: families[known].label, isNew: false }
    // trending models can come from any org, so keep its name to plain characters
    const name = (ORG_NAMES[org] || org).replace(/[^\w .-]/g, '')
    const key = name.toLowerCase().replace(/[^a-z0-9]/g, '')
    if (Object.hasOwn(families, key)) return { key, label: families[key].label, isNew: false }
    return { key, label: `${name} family`, isNew: true }
}

// families.js is hand-formatted, so add one line before the object's closing brace
export async function addFamily({ key, label }, file = FAMILIES_FILE) {
    const text = await readFile(file, 'utf8')
    const end = text.lastIndexOf('\n}')
    if (end < 0 || !/export const MODEL_FAMILIES = \{/.test(text)) throw new Error('families.js is not in the expected shape')
    const quote = s => `'${s.replace(/\\/g, '\\\\').replace(/'/g, "\\'")}'`
    const prop = /^[a-z_$][\w$]*$/i.test(key) ? key : quote(key)
    const line = `\n    ${prop}: { key: ${quote(key)}, label: ${quote(label)} },`
    await writeFile(file, text.slice(0, end) + line + text.slice(end))
}
