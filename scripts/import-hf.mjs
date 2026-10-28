#!/usr/bin/env node
// Usage:
//   node scripts/import-hf.mjs <org/repo> [--mirror <org/repo>]   print the normalized config
//   node scripts/import-hf.mjs --check                            diff src/data/models.js against HF
import { readFile } from 'node:fs/promises'
import { fetchConfig, normalize, isMoE } from './lib/hf.mjs'
import { MODEL_REGISTRY, MODEL_FAMILIES } from '../src/data/models.js'

const DIM_FIELDS = [
    'd_model', 'n_layers', 'n_heads', 'n_kv_heads', 'head_dim', 'vocab_size', 'context_length',
    'd_ff', 'n_experts', 'n_active_experts', 'n_shared_experts', 'n_dense_layers',
    'sliding_window', 'alternating_moe',
]
const FEATURE_FIELDS = ['attn', 'norm', 'ffn', 'posEncoding', 'hasBias', 'tiedEmbeddings', 'qkNorm']

async function importRepo(repo, mirror) {
    const { config, source } = await fetchConfig(repo, { mirror })
    return { source, ...normalize(config) }
}

function diffEntry(entry, imported) {
    const family = MODEL_FAMILIES[entry.family] || {}
    const diffs = []
    for (const field of DIM_FIELDS) {
        if (!(field in entry)) continue
        // the UI draws d_ff as the per-expert width for MoE models
        const actual = field === 'd_ff' && isMoE(imported) ? imported.d_ff_expert : imported[field]
        if (entry[field] !== actual) diffs.push({ field, ours: entry[field], hf: actual })
    }
    for (const field of FEATURE_FIELDS) {
        if (family[field] !== imported.features[field]) {
            diffs.push({ field: `family.${field}`, ours: family[field], hf: imported.features[field] })
        }
    }
    return diffs
}

async function check() {
    const sources = JSON.parse(await readFile(new URL('./hf-sources.json', import.meta.url), 'utf8'))
    const keys = Object.keys(MODEL_REGISTRY)
    const results = await Promise.all(keys.map(async key => {
        const src = sources[key]
        if (!src) return { key, error: 'no entry in scripts/hf-sources.json' }
        try {
            return { key, imported: await importRepo(src.repo, src.mirror) }
        } catch (err) {
            return { key, error: err.message }
        }
    }))

    let failed = 0
    for (const { key, imported, error } of results) {
        if (error) {
            failed++
            console.log(`✗ ${key}: ${error}`)
            continue
        }
        const diffs = diffEntry(MODEL_REGISTRY[key], imported)
        const via = imported.source.config_from ? ` (config via ${imported.source.config_from})` : ''
        if (!diffs.length) {
            console.log(`✓ ${key}  ${imported.source.repo}${via}`)
            continue
        }
        failed++
        console.log(`✗ ${key}  ${imported.source.repo}${via}`)
        for (const d of diffs) console.log(`    ${d.field.padEnd(22)} ours: ${String(d.ours).padEnd(10)} hf: ${d.hf}`)
    }
    console.log(`\n${keys.length - failed}/${keys.length} models match their HF config`)
    process.exitCode = failed ? 1 : 0
}

const args = process.argv.slice(2)
if (args[0] === '--check') {
    await check()
} else if (args[0] && !args[0].startsWith('--')) {
    const mirrorIdx = args.indexOf('--mirror')
    const mirror = mirrorIdx >= 0 ? args[mirrorIdx + 1] : undefined
    try {
        console.log(JSON.stringify(await importRepo(args[0], mirror), null, 4))
    } catch (err) {
        console.error(`✗ ${args[0]}: ${err.message}`)
        process.exitCode = 1
    }
} else {
    console.error('usage: node scripts/import-hf.mjs <org/repo> [--mirror <org/repo>] | --check')
    process.exitCode = 2
}
