#!/usr/bin/env node
// Usage:
//   node scripts/import-hf.mjs <org/repo> [--mirror <org/repo>]   print the normalized config
//   node scripts/import-hf.mjs --check                            compare src/data/models/*.json with HF
//   node scripts/import-hf.mjs --write                            same, and update the files that differ
import { writeFile } from 'node:fs/promises'
import { importModel } from './lib/hf.mjs'
import { HAND_FIELDS, loadModels, generatedFields } from './lib/catalog.mjs'

function diffModel(current, fresh) {
    const ours = generatedFields(current)
    const hf = generatedFields(fresh)
    return [...new Set([...Object.keys(ours), ...Object.keys(hf)])]
        .filter(field => ours[field] !== hf[field])
        .map(field => ({ field, ours: ours[field], hf: hf[field] }))
}

async function sync({ write }) {
    const models = await loadModels()
    const results = await Promise.all(models.map(async ({ file, url, model: current }) => {
        try {
            const fresh = await importModel(current.source.repo, current.source.mirror)
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
    const ok = models.length - errors - differing
    console.log(`\n${ok}/${models.length} model files match their HF config`)
    process.exitCode = errors || (differing && !write) ? 1 : 0
}

const args = process.argv.slice(2)
if (args[0] === '--check' || args[0] === '--write') {
    await sync({ write: args[0] === '--write' })
} else if (args[0] && !args[0].startsWith('--')) {
    const mirrorIdx = args.indexOf('--mirror')
    const mirror = mirrorIdx >= 0 ? args[mirrorIdx + 1] : undefined
    try {
        console.log(JSON.stringify(await importModel(args[0], mirror), null, 4))
    } catch (err) {
        console.error(`✗ ${args[0]}: ${err.message}`)
        process.exitCode = 1
    }
} else {
    console.error('usage: node scripts/import-hf.mjs <org/repo> [--mirror <org/repo>] | --check | --write')
    process.exitCode = 2
}
