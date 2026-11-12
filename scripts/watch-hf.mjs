#!/usr/bin/env node
// Look for new models on Hugging Face and propose them for the app.
// usage:
//   node scripts/watch-hf.mjs [--since YYYY-MM-DD] [--out <dir>]   report only; --out also writes the proposed model files there
//   node scripts/watch-hf.mjs --open-prs [--since YYYY-MM-DD]      also open one pull request per model (GitHub Actions only)
import { readFile, mkdir } from 'node:fs/promises'
import { execFileSync } from 'node:child_process'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { listModelsSince, listTrending, getModelInfo, importModel, hasAdapter, UnknownArchitectureError } from './lib/hf.mjs'
import { loadModels, writeModel, fingerprint, slug, labelFromRepo, formatParams, classifyFamily, addFamily } from './lib/catalog.mjs'
import { MODEL_FAMILIES } from '../src/data/families.js'

const settings = JSON.parse(await readFile(new URL('./watch-list.json', import.meta.url), 'utf8'))
const TEXT_PIPELINES = ['text-generation', 'image-text-to-text']
const ISSUE_TITLE = 'hf-watcher: new architectures waiting for an adapter'
const LINEAGE_WORDS = { quantized: 'compressed copy', adapter: 'add-on', merge: 'blend' }
// compressed copies that HF hasn't tagged as quantized still say so in their name
const COMPRESSED_NAME = /(^|[-_.])(gguf|gptq|awq|(nv|mx)?fp[48]|int[48]|w\d+a\d+|bnb|[48]bit|mlx|onnx|exl[23]|qat)([-_.]|$)/i
const SHOW_REPOS = 5
const ignoredTypes = new Set(settings.ignoreModelTypes || [])

const argValue = name => {
    const i = process.argv.indexOf(name)
    return i >= 0 ? process.argv[i + 1] : undefined
}
const openPRs = process.argv.includes('--open-prs')
const outDir = argValue('--out')
const since = argValue('--since') ? new Date(argValue('--since')) : new Date(Date.now() - settings.lookbackDays * 864e5)
if (Number.isNaN(since.getTime())) {
    console.error('--since must be a date like 2026-10-01')
    process.exit(2)
}
// opening pull requests pushes to the repo, so only the GitHub workflow may do it
if (openPRs && process.env.GITHUB_ACTIONS !== 'true') {
    console.error('--open-prs only runs inside GitHub Actions; run without it to see the report')
    process.exit(2)
}

const watched = new Set(settings.orgs.map(o => o.toLowerCase()))
const orgOf = repo => repo.split('/')[0]
const today = new Date().toISOString().slice(0, 10)

// HF tags a derived repo with base_model:<kind>:<repo it came from>
function lineage(tags = []) {
    for (const tag of tags) {
        const [prefix, kind, ...rest] = tag.split(':')
        if (prefix === 'base_model' && rest.length) return { kind, base: rest.join(':') }
    }
    return null
}

// follow fine-tunes back to the original design, as long as it stays with a watched company
async function findOriginal(repo, info, depth = 0) {
    const from = lineage(info.tags)
    if (!from) return { repo, info }
    if (from.kind !== 'finetune') return { skip: `${LINEAGE_WORDS[from.kind] || from.kind} of ${from.base}` }
    if (!watched.has(orgOf(from.base).toLowerCase())) return { skip: `fine-tune of ${from.base}, which is not from a watched company` }
    if (depth >= 3) return { skip: 'fine-tune chain is too long to follow' }
    try {
        return await findOriginal(from.base, await getModelInfo(from.base), depth + 1)
    } catch (err) {
        return { skip: `could not look up ${from.base}: ${err.message}` }
    }
}

// model types are free text in someone else's config, so keep them to plain characters
// before they go into reports, issues or pull requests
const safe = text => String(text).replace(/[^\w./-]/g, '_').slice(0, 100)

function noteUnknown(unknown, modelType, repo) {
    const type = safe(modelType)
    if (!unknown.has(type)) unknown.set(type, new Set())
    unknown.get(type).add(safe(repo))
}

// Recent uploads from the watched companies, then popular trending models from
// anyone. Both go through the same filters below.
async function gatherUploads(result) {
    const uploads = new Map()
    for (const org of settings.orgs) {
        try {
            for (const item of await listModelsSince(org, since)) uploads.set(item.id, { item, origin: 'watch list' })
        } catch (err) {
            result.errors.push(`${org}: ${err.message}`)
        }
    }
    if (!settings.trending?.enabled) return [...uploads.values()]
    for (const pipeline of TEXT_PIPELINES) {
        try {
            for (const item of await listTrending(pipeline)) {
                if ((item.likes || 0) < settings.trending.minLikes || uploads.has(item.id)) continue
                uploads.set(item.id, { item, origin: 'trending' })
                result.trending++
            }
        } catch (err) {
            result.errors.push(`trending ${pipeline} models: ${err.message}`)
        }
    }
    return [...uploads.values()]
}

async function discover() {
    const existing = await loadModels()
    const knownRepos = new Set(existing.flatMap(({ model }) => [model.source.repo, model.source.mirror]).filter(Boolean))
    const knownKeys = new Set(existing.map(({ model }) => model.key))
    const designs = new Map(existing.map(({ model }) => [fingerprint(model), model.key]))
    const result = { scanned: 0, trending: 0, ignored: 0, proposals: [], unknown: new Map(), skipped: [], errors: [] }
    const handled = new Set()

    for (const { item, origin } of await gatherUploads(result)) {
        if (!TEXT_PIPELINES.includes(item.pipeline_tag)) continue
        if (ignoredTypes.has(item.config?.model_type)) { result.ignored++; continue }
        result.scanned++
        const skip = why => result.skipped.push({ repo: item.id, why })
        if (COMPRESSED_NAME.test(item.id.split('/')[1])) { skip('compressed copy (from its name)'); continue }

        const original = await findOriginal(item.id, item)
        if (original.skip) {
            skip(original.skip)
            continue
        }
        const { repo, info } = original
        const via = repo === item.id ? null : item.id
        if (handled.has(repo)) continue
        handled.add(repo)

        if (knownRepos.has(repo)) { skip(`${repo} is already in the app`); continue }
        if (info.gated && !process.env.HF_TOKEN) { skip(`${repo} is gated; needs an HF_TOKEN to read`); continue }
        const modelType = info.config?.model_type
        if (!modelType) { skip(`${repo} has no config.json to read`); continue }
        if (ignoredTypes.has(modelType)) { result.ignored++; continue }
        if (!hasAdapter(modelType)) { noteUnknown(result.unknown, modelType, repo); continue }

        let imported
        try {
            imported = await importModel(repo)
        } catch (err) {
            if (err instanceof UnknownArchitectureError) noteUnknown(result.unknown, err.modelType, repo)
            else result.errors.push(`${repo}: ${err.message}`)
            continue
        }
        const sameAs = designs.get(fingerprint(imported))
        if (sameAs) { skip(`${repo} has the same design as ${sameAs}`); continue }
        const label = labelFromRepo(repo)
        const key = slug(label)
        if (knownKeys.has(key)) { skip(`${repo}: a model file named ${key}.json already exists`); continue }

        const family = classifyFamily(modelType, repo, MODEL_FAMILIES)
        const model = { key, label, family: family.key, params: formatParams(info.safetensors?.total), ...imported }
        designs.set(fingerprint(model), key)
        knownKeys.add(key)
        result.proposals.push({ model, family, via, origin, likes: item.likes })
    }
    return result
}

const shortList = (items, show = x => x) => items.length > SHOW_REPOS
    ? `${items.slice(0, SHOW_REPOS).map(show).join(', ')} and ${items.length - SHOW_REPOS} more`
    : items.map(show).join(', ')

function report(r) {
    const lines = [
        `Looked at ${r.scanned} text models: uploads since ${since.toISOString().slice(0, 10)} from ${settings.orgs.length} watched companies,`,
        `plus ${r.trending} trending models with ${settings.trending?.minLikes}+ likes from anyone.`,
        '',
        `New models to propose (${r.proposals.length}):`,
    ]
    for (const p of r.proposals) {
        lines.push(`  + [${p.origin}] ${p.model.label} (${p.model.key}.json) from ${p.model.source.repo}${p.via ? `, found through ${p.via}` : ''}; group: ${p.family.label}${p.family.isNew ? ' (new)' : ''}`)
    }
    lines.push('', `New architectures with no adapter yet (${r.unknown.size}):`)
    for (const [type, repos] of r.unknown) lines.push(`  ? ${type}: ${shortList([...repos])}`)
    if (r.ignored) lines.push('', `Ignored ${r.ignored} models whose architecture is listed in ignoreModelTypes.`)
    lines.push('', `Skipped (${r.skipped.length}):`)
    for (const s of r.skipped) lines.push(`  - ${s.repo}: ${s.why}`)
    if (r.errors.length) {
        lines.push('', `Errors (${r.errors.length}):`)
        for (const e of r.errors) lines.push(`  ! ${e}`)
    }
    return lines.join('\n')
}

const run = (cmd, args) => execFileSync(cmd, args, { encoding: 'utf8', stdio: ['pipe', 'pipe', 'inherit'] }).trim()

function validate() {
    try {
        return run('node', ['scripts/validate-models.mjs'])
    } catch (err) {
        throw new Error(`the model file failed validation:\n${err.stdout}`, { cause: err })
    }
}

function prBody(p, validation) {
    const m = p.model
    const moe = m.n_experts > 1
    const warnings = validation.split('\n').filter(l => l.includes('warning:')).map(l => `- ${l.replace(/^\s*warning:\s*/, '')}`)
    return [
        'The daily Hugging Face watcher found a model the app does not have yet.',
        '',
        `- **Source:** [${m.source.repo}](https://huggingface.co/${m.source.repo}) at commit \`${m.source.revision.slice(0, 7)}\``,
        `- **Found on:** ${p.origin === 'trending'
            ? `the Hugging Face trending list (${p.likes} likes)${watched.has(orgOf(m.source.repo).toLowerCase()) ? '' : ', from a company you don\'t watch yet'}`
            : 'your watch list'}`,
        p.via && `- **Found through:** [${p.via}](https://huggingface.co/${p.via}), a fine-tune of it`,
        `- **Design:** ${m.n_layers} layers, ${m.d_model} dims, ${m.features.attn} attention, ${m.features.ffn}${moe ? `, ${m.n_experts} experts (${m.n_active_experts} active)` : ''}`,
        `- **Dropdown group:** ${p.family.label}${p.family.isNew ? ' (new group, added to src/data/families.js)' : ''}`,
        '',
        'Before merging, check what the watcher had to guess:',
        `- [ ] Name: "${m.label}"`,
        `- [ ] Size: "${m.params}"${moe ? ' (add the active count, e.g. "30B (3B active)")' : ''}`,
        p.family.isNew && `- [ ] New group name: "${p.family.label}"`,
        ...(warnings.length ? ['', 'Validation warnings:', ...warnings] : []),
    ].filter(line => typeof line === 'string').join('\n')
}

async function openPullRequests(proposals) {
    const start = run('git', ['rev-parse', 'HEAD'])
    let opened = 0
    for (const p of proposals) {
        if (opened >= settings.maxPullRequestsPerRun) {
            console.log(`Reached the limit of ${settings.maxPullRequestsPerRun} pull requests; the rest will be proposed on a later run.`)
            break
        }
        // not under hf-watcher/: git cannot have a branch and a folder of branches with the same name
        const branch = `new-model/${p.model.key}`
        // a closed pull request means it was rejected, so never propose it again
        if (run('gh', ['pr', 'list', '--state', 'all', '--head', branch, '--json', 'number', '--jq', 'length']) !== '0') {
            console.log(`- ${p.model.key}: a pull request for it already exists (open or closed)`)
            continue
        }
        try {
            run('git', ['switch', '--quiet', '-C', branch, start])
            await writeModel(p.model)
            if (p.family.isNew) await addFamily(p.family)
            const validation = validate()
            run('git', ['add', 'src/data'])
            run('git', ['commit', '--quiet', '-m', `add ${p.model.label} model from ${p.model.source.repo}`])
            run('git', ['push', '--quiet', '--force', 'origin', branch])
            const url = run('gh', ['pr', 'create', '--base', settings.baseBranch, '--head', branch, '--title', `Add ${p.model.label}`, '--body', prBody(p, validation)])
            console.log(`+ opened ${url}`)
            opened++
        } catch (err) {
            console.log(`! ${p.model.key}: ${err.message}`)
            process.exitCode = 1
        } finally {
            run('git', ['checkout', '--quiet', '--force', start])
            run('git', ['clean', '-fdq', '--', 'src/data'])
        }
    }
}

const DATA_MARK = /<!-- hf-watcher-data (.*) -->/

// one issue lists every architecture waiting for an adapter, so nothing is reported twice
function updateArchitectureIssue(unknown) {
    const issues = JSON.parse(run('gh', ['issue', 'list', '--state', 'open', '--search', `"${ISSUE_TITLE}" in:title`, '--json', 'number,title,body']))
    const issue = issues.find(i => i.title === ISSUE_TITLE)
    let data = {}
    try {
        data = issue ? JSON.parse(issue.body.match(DATA_MARK)?.[1] || '{}') : {}
    } catch {
        console.log('The issue\'s hidden data was edited and could not be read, so it starts fresh')
    }
    let changed = false
    for (const [type, repos] of unknown) {
        data[type] ||= { firstSeen: today, repos: [] }
        for (const repo of repos) {
            if (data[type].repos.includes(repo)) continue
            data[type].repos.push(repo)
            changed = true
        }
    }
    if (!changed) return
    const rows = Object.entries(data).map(([type, d]) =>
        `| \`${type}\` | ${d.firstSeen} | ${shortList(d.repos, r => `[${r}](https://huggingface.co/${r})`)} |`)
    const body = [
        'The daily watcher found these models, but the importer has no adapter for their architecture yet, so it cannot add them by itself. Each one needs an adapter in `scripts/lib/hf.mjs` first.',
        '',
        '| Architecture (`model_type`) | First seen | Models |',
        '|---|---|---|',
        ...rows,
        '',
        `<!-- hf-watcher-data ${JSON.stringify(data)} -->`,
    ].join('\n')
    if (issue) run('gh', ['issue', 'edit', String(issue.number), '--body', body])
    else run('gh', ['issue', 'create', '--title', ISSUE_TITLE, '--body', body])
    console.log(`Updated the "${ISSUE_TITLE}" issue`)
}

const result = await discover()
console.log(report(result))
if (outDir) {
    const dir = pathToFileURL(resolve(outDir) + '/')
    await mkdir(dir, { recursive: true })
    for (const p of result.proposals) await writeModel(p.model, dir)
    console.log(`\nWrote ${result.proposals.length} proposed model files to ${resolve(outDir)}`)
}
if (openPRs) {
    await openPullRequests(result.proposals)
    updateArchitectureIssue(result.unknown)
}
if (result.errors.length) process.exitCode = 1
