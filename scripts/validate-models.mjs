#!/usr/bin/env node
// Check every model file before the app (or a PR) uses it.
// usage: node scripts/validate-models.mjs [dir]   (default: src/data/models)
import { readdir, readFile } from 'node:fs/promises'
import { resolve, basename } from 'node:path'
import { fileURLToPath } from 'node:url'
import { MODEL_FAMILIES } from '../src/data/families.js'
import { unknownFeatures } from '../src/data/features.js'
import { PROMPTS, NEXT_TOKEN_PREDICTIONS } from '../src/data/prompts.js'

const isPosInt = v => Number.isInteger(v) && v > 0
const isNonNegInt = v => Number.isInteger(v) && v >= 0
const isText = v => typeof v === 'string' && v.trim() !== ''
const show = v => JSON.stringify(v)

function checkModel(m, file) {
    const errors = []
    const warnings = []
    const need = (ok, msg) => { if (!ok) errors.push(msg) }

    need(m.key === basename(file, '.json'), `key ${show(m.key)} should match the file name`)
    for (const f of ['label', 'family', 'params']) need(isText(m[f]), `${f} must be a non-empty string`)
    need(isText(m.source?.repo), 'source.repo must name the Hugging Face repo')
    if (isText(m.family)) {
        need(Object.hasOwn(MODEL_FAMILIES, m.family),
            `family ${show(m.family)} is not in src/data/families.js, so the model won't appear in the dropdown`)
    }

    for (const f of ['d_model', 'n_layers', 'n_heads', 'n_kv_heads', 'head_dim', 'vocab_size', 'context_length']) {
        need(isPosInt(m[f]), `${f} must be a positive whole number (got ${show(m[f])})`)
    }
    if (isPosInt(m.n_heads) && isPosInt(m.n_kv_heads)) {
        need(m.n_heads % m.n_kv_heads === 0, `n_heads (${m.n_heads}) must be a multiple of n_kv_heads (${m.n_kv_heads})`)
    }

    const f = m.features
    if (!f || typeof f !== 'object') {
        errors.push('features is missing')
    } else {
        for (const k of ['attn', 'norm', 'ffn', 'posEncoding']) need(isText(f[k]), `features.${k} must be a non-empty string`)
        for (const k of ['hasBias', 'tiedEmbeddings', 'qkNorm']) need(typeof f[k] === 'boolean', `features.${k} must be true or false`)
        // the attention name has to agree with the head counts the diagrams draw
        const kv = m.n_kv_heads
        const heads = m.n_heads
        if (f.attn === 'MHA') need(kv === heads, `MHA needs n_kv_heads == n_heads (got ${kv} of ${heads})`)
        if (f.attn === 'GQA') need(kv > 1 && kv < heads, `GQA needs 1 < n_kv_heads < n_heads (got ${kv} of ${heads})`)
        if (f.attn === 'MQA') need(kv === 1, `MQA needs n_kv_heads == 1 (got ${kv})`)
        need((f.attn === 'MLA') === Boolean(m.mla),
            f.attn === 'MLA' ? 'MLA models need an mla block' : 'only MLA models should have an mla block')
        for (const u of unknownFeatures(f).filter(u => isText(u.value))) {
            warnings.push(`features.${u.field} ${show(u.value)} has no detailed explanation yet, so the app shows a generic one`)
        }
    }
    if (m.mla) {
        for (const k of ['kv_lora_rank', 'qk_nope_head_dim', 'qk_rope_head_dim', 'v_head_dim']) {
            need(isPosInt(m.mla[k]), `mla.${k} must be a positive whole number`)
        }
        need(m.head_dim === m.mla.qk_nope_head_dim + m.mla.qk_rope_head_dim,
            `MLA head_dim (${m.head_dim}) should be qk_nope_head_dim + qk_rope_head_dim`)
    }

    need(isNonNegInt(m.n_experts), `n_experts must be 0 or more (got ${show(m.n_experts)})`)
    if (m.n_experts > 1) {
        need(isPosInt(m.d_ff_expert), 'MoE models need d_ff_expert (the width of one expert)')
        need(isPosInt(m.n_active_experts) && m.n_active_experts <= m.n_experts,
            `n_active_experts must be between 1 and n_experts (got ${show(m.n_active_experts)} of ${m.n_experts})`)
        need(isNonNegInt(m.n_shared_experts), 'n_shared_experts must be 0 or more')
        need(m.d_ff === null || isPosInt(m.d_ff), 'd_ff must be a positive whole number, or null when there are no dense layers')
    } else {
        need(isPosInt(m.d_ff), 'dense models need d_ff (the FFN width)')
    }
    need(isNonNegInt(m.n_dense_layers) && (!isPosInt(m.n_layers) || m.n_dense_layers < m.n_layers),
        `n_dense_layers must be between 0 and n_layers - 1 (got ${show(m.n_dense_layers)})`)
    need(isNonNegInt(m.sliding_window), `sliding_window must be 0 or more (got ${show(m.sliding_window)})`)
    need(typeof m.alternating_moe === 'boolean', 'alternating_moe must be true or false')

    if (isText(m.family)) {
        const noTokens = Object.keys(PROMPTS).filter(p => !PROMPTS[p][m.family])
        const noPreds = Object.keys(NEXT_TOKEN_PREDICTIONS).filter(p => !NEXT_TOKEN_PREDICTIONS[p][m.family])
        if (noTokens.length || noPreds.length) {
            warnings.push(`src/data/prompts.js has no tokens or predictions for family ${show(m.family)}, so GPT-2's are shown`)
        }
    }
    return { errors, warnings }
}

const dir = resolve(process.argv[2] || fileURLToPath(new URL('../src/data/models/', import.meta.url)))
const files = (await readdir(dir)).filter(f => f.endsWith('.json')).sort()
let errorCount = 0
let warningCount = 0

for (const file of files) {
    let result
    try {
        const model = JSON.parse(await readFile(resolve(dir, file), 'utf8'))
        result = model && typeof model === 'object' && !Array.isArray(model)
            ? checkModel(model, file)
            : { errors: ['file must contain a JSON object'], warnings: [] }
    } catch (err) {
        result = { errors: [`not valid JSON: ${err.message}`], warnings: [] }
    }
    const { errors, warnings } = result
    errorCount += errors.length
    warningCount += warnings.length
    console.log(`${errors.length ? '✗' : warnings.length ? '!' : '✓'} ${file}`)
    for (const e of errors) console.log(`    error:   ${e}`)
    for (const w of warnings) console.log(`    warning: ${w}`)
}

if (!files.length) {
    errorCount++
    console.log(`✗ no model files found in ${dir}`)
}
console.log(`\n${files.length} model files: ${errorCount} errors, ${warningCount} warnings`)
process.exitCode = errorCount ? 1 : 0
