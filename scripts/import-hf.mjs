#!/usr/bin/env node
// Usage:
//   node scripts/import-hf.mjs <org/repo> [--mirror <org/repo>]   print the normalized config
//   node scripts/import-hf.mjs --check                            compare src/data/models/*.json with HF
//   node scripts/import-hf.mjs --write                            same, and update the files that differ
import { readdir, readFile, writeFile } from 'node:fs/promises'
import { fetchConfig, normalize } from './lib/hf.mjs'

const MODELS_DIR = new URL('../src/data/models/', import.meta.url)
// written by hand; everything else in a model file is regenerated from HF
const HAND_FIELDS = ['key', 'label', 'family', 'params']

async function importRepo(repo, mirror) {
    const { config, source } = await fetchConfig(repo, { mirror })
    return { source: { repo, ...(mirror && { mirror }), ...source }, ...normalize(config) }
}

function flatten(obj, prefix = '') {
    return Object.entries(obj).flatMap(([k, v]) =>
        v && typeof v === 'object' && !Array.isArray(v)
            ? flatten(v, `${prefix}${k}.`)
            : [[`${prefix}${k}`, JSON.stringify(v)]])
}

// a new upstream commit on its own isn't a difference, so source is skipped
function generatedFields(model) {
    const rest = { ...model }
    for (const f of ['source', ...HAND_FIELDS]) delete rest[f]
    return Object.fromEntries(flatten(rest))
}

function diffModel(current, fresh) {
    const ours = generatedFields(current)
    const hf = generatedFields(fresh)
    return [...new Set([...Object.keys(ours), ...Object.keys(hf)])]
        .filter(field => ours[field] !== hf[field])
        .map(field => ({ field, ours: ours[field], hf: hf[field] }))
}

async function sync({ write }) {
    const files = (await readdir(MODELS_DIR)).filter(f => f.endsWith('.json')).sort()
    const results = await Promise.all(files.map(async file => {
        const url = new URL(file, MODELS_DIR)
        const current = JSON.parse(await readFile(url, 'utf8'))
        try {
            const fresh = await importRepo(current.source.repo, current.source.mirror)
            return { file, url, current, fresh, diffs: diffModel(current, fresh) }
        } catch (err) {
            return { file, error: err.message }
        }
    }))

    let errors = 0
    let differing = 0
    for (const { file, url, current, fresh, diffs, error } of results) {
        if (error) {
            errors++
            console.log(`✗ ${file}: ${error}`)
            continue
        }
        if (!diffs.length) {
            console.log(`✓ ${file}`)
            continue
        }
        differing++
        console.log(`✗ ${file}  ${current.source.repo}`)
        for (const d of diffs) console.log(`    ${d.field.padEnd(26)} ours: ${String(d.ours).padEnd(10)} hf: ${d.hf}`)
        if (write) {
            const hand = Object.fromEntries(HAND_FIELDS.map(f => [f, current[f]]))
            await writeFile(url, JSON.stringify({ ...hand, ...fresh }, null, 4) + '\n')
            console.log('    updated')
        }
    }
    const ok = files.length - errors - differing
    console.log(`\n${ok}/${files.length} model files match their HF config`)
    process.exitCode = errors || (differing && !write) ? 1 : 0
}

const args = process.argv.slice(2)
if (args[0] === '--check' || args[0] === '--write') {
    await sync({ write: args[0] === '--write' })
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
    console.error('usage: node scripts/import-hf.mjs <org/repo> [--mirror <org/repo>] | --check | --write')
    process.exitCode = 2
}
