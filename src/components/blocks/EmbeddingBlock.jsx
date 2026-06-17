import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { makeEmbeddingVector } from '../../data/models'
import Tooltip from '../Tooltip'

const SHOW_DIMS = 28

function valueToColor(v) {
    const a = Math.abs(v)
    return v > 0
        ? `rgba(127,119,221,${0.15 + a * 0.85})`
        : `rgba(184,84,40,${0.15 + a * 0.85})`
}

function cosineSim(a, b) {
    let dot = 0, magA = 0, magB = 0
    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i]
        magA += a[i] * a[i]
        magB += b[i] * b[i]
    }
    return dot / (Math.sqrt(magA) * Math.sqrt(magB))
}

function SimilarityMatrix({ tokens, vecs }) {
    const [revealed, setRevealed] = useState(false)

    useEffect(() => {
        const t = setTimeout(() => setRevealed(true), 300)
        return () => clearTimeout(t)
    }, [])

    const sims = tokens.map((_, i) =>
        tokens.map((_, j) => parseFloat(cosineSim(vecs[i], vecs[j]).toFixed(2)))
    )

    return (
        <div>
            <div style={{ display: 'inline-grid', gridTemplateColumns: `44px repeat(${tokens.length}, 72px)`, gap: 0 }}>
                <div />
                {tokens.map((t, i) => (
                    <motion.div key={i}
                        style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: '#AFA9EC', textAlign: 'center', padding: '6px 4px', fontWeight: 500 }}
                        initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: i * 0.08, duration: 0.3 }}>
                        {t.token}
                    </motion.div>
                ))}
                {tokens.map((t, ri) => (
                    <div key={ri} style={{ display: 'contents' }}>
                        <motion.div
                            style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: '#AFA9EC', display: 'flex', alignItems: 'center', justifyContent: 'flex-end', paddingRight: 8, fontWeight: 500 }}
                            initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: ri * 0.08, duration: 0.3 }}>
                            {t.token}
                        </motion.div>
                        {tokens.map((_, ci) => {
                            const sim = sims[ri][ci]
                            const isDiag = ri === ci
                            const intensity = isDiag ? 0.5 : sim * 0.5
                            return (
                                <motion.div key={ci}
                                    style={{
                                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                                        height: 56, borderRadius: 6, margin: 2, fontSize: 12,
                                        fontFamily: 'var(--font-mono)', fontWeight: 500,
                                        background: `rgba(127,119,221,${intensity})`,
                                        color: isDiag ? '#e8e6e0' : '#AFA9EC',
                                    }}
                                    initial={{ opacity: 0, scale: 0.8 }}
                                    animate={revealed ? { opacity: 1, scale: 1 } : { opacity: 0, scale: 0.8 }}
                                    transition={{ delay: (ri * tokens.length + ci) * 0.04, duration: 0.35, ease: [0.16, 1, 0.3, 1] }}>
                                    {sim.toFixed(2)}
                                </motion.div>
                            )
                        })}
                    </div>
                ))}
            </div>
            <p style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 10 }}>
                Brighter cells mean the two tokens are more similar. A score of 1.00 means identical (the diagonal).
                Function words like "The" and "on" tend to be closer to each other, while content words like "cat"
                and "sat" share some overlap because they often appear together.
            </p>
        </div>
    )
}

function VectorFingerprints({ tokens, vecs }) {
    const [highlighted, setHighlighted] = useState(null)

    return (
        <div>
            <div style={{ display: 'flex', gap: 20, alignItems: 'flex-end', flexWrap: 'wrap' }}>
                {tokens.map((t, ti) => (
                    <motion.div key={ti}
                        style={{
                            display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6,
                            cursor: 'pointer',
                            opacity: highlighted !== null && highlighted !== ti ? 0.25 : 1,
                            transition: 'opacity 0.3s ease',
                        }}
                        onClick={() => setHighlighted(highlighted === ti ? null : ti)}>
                        <span style={{ fontSize: 13, fontFamily: 'var(--font-mono)', color: '#AFA9EC', fontWeight: 500 }}>{t.token}</span>
                        <div style={{ display: 'flex', gap: 1, alignItems: 'flex-end', height: 100 }}>
                            {vecs[ti].slice(0, SHOW_DIMS).map((v, di) => (
                                <motion.div key={di}
                                    style={{ width: 4, borderRadius: 2, background: valueToColor(v) }}
                                    initial={{ height: 0 }}
                                    animate={{ height: Math.max(Math.abs(v) * 55, 2) }}
                                    transition={{ delay: ti * 0.12 + di * 0.015, duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
                                />
                            ))}
                        </div>
                        <span style={{ fontSize: 9, color: 'var(--text-muted)' }}>{SHOW_DIMS} of 768</span>
                    </motion.div>
                ))}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginLeft: 8, alignSelf: 'center' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <div style={{ width: 12, height: 12, borderRadius: 2, background: 'rgba(127,119,221,0.7)' }} />
                        <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>positive</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <div style={{ width: 12, height: 12, borderRadius: 2, background: 'rgba(184,84,40,0.7)' }} />
                        <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>negative</span>
                    </div>
                    <span style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>height = magnitude</span>
                </div>
            </div>
            <p style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 10, fontStyle: 'italic' }}>
                click a token to isolate its fingerprint and compare shapes
            </p>
        </div>
    )
}

function DimensionExplorer({ tokens, vecs, accent }) {
    const [selectedDim, setSelectedDim] = useState(null)
    const [selectedToken, setSelectedToken] = useState(0)

    return (
        <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 10, fontSize: 11, color: 'var(--text-muted)', flexWrap: 'wrap' }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                    <span style={{ width: 12, height: 12, borderRadius: 2, background: 'rgba(127,119,221,0.8)', display: 'inline-block' }} />
                    positive
                </span>
                <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                    <span style={{ width: 12, height: 12, borderRadius: 2, background: 'rgba(184,84,40,0.8)', display: 'inline-block' }} />
                    negative
                </span>
                <span>intensity = magnitude</span>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                {tokens.map((t, ti) => (
                    <motion.div key={ti}
                        style={{
                            display: 'flex', alignItems: 'center', gap: 0, padding: '5px 8px', borderRadius: 6,
                            cursor: 'pointer',
                            background: selectedToken === ti ? 'var(--bg-elevated)' : 'transparent',
                            outline: selectedToken === ti ? '1px solid var(--border-soft)' : 'none',
                        }}
                        onClick={() => setSelectedToken(ti)}
                        whileHover={{ background: 'var(--bg-elevated)' }}>
                        <span style={{ width: 44, fontSize: 12, fontFamily: 'var(--font-mono)', color: '#AFA9EC', textAlign: 'right', paddingRight: 10, flexShrink: 0, fontWeight: 500 }}>
                            {t.token}
                        </span>
                        <div style={{ display: 'flex', gap: 2, flex: 1 }}>
                            {vecs[ti].slice(0, SHOW_DIMS).map((v, di) => (
                                <motion.div key={di}
                                    onClick={(e) => { e.stopPropagation(); setSelectedDim(selectedDim === di ? null : di) }}
                                    style={{
                                        width: 20, height: 28, borderRadius: 3, background: valueToColor(v),
                                        cursor: 'pointer',
                                        outline: selectedDim === di ? '2px solid #AFA9EC' : 'none',
                                        outlineOffset: 1, zIndex: selectedDim === di ? 2 : 0, position: 'relative',
                                    }}
                                    initial={{ opacity: 0, scale: 0.7 }}
                                    animate={{ opacity: 1, scale: 1 }}
                                    transition={{ delay: ti * 0.06 + di * 0.008, duration: 0.3 }}
                                    title={`dim ${di}: ${v.toFixed(3)}`}
                                />
                            ))}
                        </div>
                    </motion.div>
                ))}
            </div>
            <AnimatePresence>
                {selectedDim !== null && (
                    <motion.div
                        style={{
                            display: 'flex', gap: 16, padding: 14, marginTop: 10,
                            background: 'var(--bg-elevated)', borderRadius: 8,
                            border: '1px solid var(--border-subtle)', flexWrap: 'wrap', alignItems: 'center',
                        }}
                        initial={{ opacity: 0, y: 8, height: 0 }}
                        animate={{ opacity: 1, y: 0, height: 'auto' }}
                        exit={{ opacity: 0, y: 8, height: 0 }}
                        transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}>
                        <span style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 500, width: '100%', marginBottom: 2 }}>
                            dimension {selectedDim} across all tokens:
                        </span>
                        {tokens.map((t, ti) => {
                            const v = vecs[ti][selectedDim]
                            const isPos = v > 0
                            return (
                                <motion.div key={ti} style={{ display: 'flex', alignItems: 'center', gap: 8 }}
                                    initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }}
                                    transition={{ delay: ti * 0.06, duration: 0.3 }}>
                                    <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: '#AFA9EC', fontWeight: 500, width: 32 }}>{t.token}</span>
                                    <div style={{ width: 80, height: 8, borderRadius: 4, background: 'var(--bg-overlay)', border: '1px solid var(--border-subtle)', overflow: 'hidden' }}>
                                        <motion.div
                                            style={{ height: '100%', borderRadius: 4, background: isPos ? '#7F77DD' : '#b85428' }}
                                            initial={{ width: 0 }}
                                            animate={{ width: `${Math.abs(v) * 100}%` }}
                                            transition={{ delay: ti * 0.06, duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
                                        />
                                    </div>
                                    <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', width: 48 }}>
                                        {v > 0 ? '+' : ''}{v.toFixed(3)}
                                    </span>
                                </motion.div>
                            )
                        })}
                    </motion.div>
                )}
            </AnimatePresence>
            <p style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 8, fontStyle: 'italic' }}>
                click any column to compare that dimension across all tokens
            </p>
        </div>
    )
}

export default function EmbeddingBlock({ model, family, tokens, accent }) {
    const a = accent || { color: '#7F77DD', light: 'rgba(127,119,221,0.1)', border: 'rgba(127,119,221,0.3)', text: '#AFA9EC' }
    const vecs = tokens.map(t => makeEmbeddingVector(t.id, model.d_model))

    return (
        <div className="block-content">
            <p className="block-explain">
                Now that each token has an integer ID, the model needs to turn those IDs into something it can
                actually compute with. Each ID is looked up in a large table called the <strong>embedding
                    matrix [{model.vocab_size.toLocaleString()} x {model.d_model}]</strong>, which returns
                a list of {model.d_model} numbers (called a vector) for each token. These numbers encode meaning:
                words with similar meanings end up with similar vectors.
                {family && family.posEncoding === 'learned'
                    ? ' A learned positional embedding is also added here so the model knows the order of the tokens.'
                    : ' Position information is not added here. Instead, this model uses RoPE (Rotary Position Embeddings) which are applied later, inside each attention layer.'
                }
            </p>

            <div className="stats-row">
                {[
                    { val: model.d_model, label: 'dims per token' },
                    { val: `${(model.vocab_size * model.d_model / 1e6).toFixed(1)}M`, label: 'embedding params' },
                    { val: `[${tokens.length}x${model.d_model}]`, label: 'output shape' },
                ].map(s => (
                    <div key={s.label} className="stat-card" style={{ background: a.light, borderColor: a.border }}>
                        <span className="stat-val" style={{ color: a.text }}>{s.val}</span>
                        <Tooltip term={s.label}><span className="stat-label">{s.label}</span></Tooltip>
                    </div>
                ))}
            </div>

            <h3 className="section-heading" style={{ color: a.color }}>
                token similarity
                <span className="section-note">how similar each pair of token vectors are</span>
            </h3>
            <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.65, marginBottom: 14 }}>
                This is the whole point of embeddings: words with similar meanings get similar vectors.
                The grid below shows a similarity score for every pair of tokens, ranging from 0
                (completely different) to 1 (identical). This score is calculated using cosine similarity,
                which measures how much two vectors point in the same direction.
            </p>
            <SimilarityMatrix tokens={tokens} vecs={vecs} />

            <h3 className="section-heading" style={{ color: a.color, marginTop: '2rem' }}>
                vector fingerprints
                <span className="section-note">first {SHOW_DIMS} of {model.d_model} dims per token</span>
            </h3>
            <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.65, marginBottom: 14 }}>
                What does a vector actually look like? Each token's {model.d_model}-number vector is shown
                below as a vertical barcode. Each bar is one dimension (one number in the list). The height
                shows how large the value is, and the color shows whether it is positive (purple) or
                negative (orange). If two tokens have similar barcode shapes, they have similar meanings.
            </p>
            <VectorFingerprints tokens={tokens} vecs={vecs} />

            <h3 className="section-heading" style={{ color: a.color, marginTop: '2rem' }}>
                dimension explorer
                <span className="section-note">click a column to compare across tokens</span>
            </h3>
            <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.65, marginBottom: 14 }}>
                Each of the {model.d_model} dimensions captures some aspect of a token's meaning, though
                individual dimensions are not easy to interpret on their own. Click any cell in the heatmap
                below to see how that particular dimension compares across all tokens. If "The" and "on"
                have similar values for a dimension but "cat" is very different, that dimension might be
                capturing something about function words versus content words.
            </p>
            <DimensionExplorer tokens={tokens} vecs={vecs} accent={a} />

            <p className="footnote">
                All of these numbers are learned during training. The model sees billions of sentences
                and gradually adjusts the embedding vectors so that words appearing in similar contexts
                (like "cat" and "dog") end up with similar vectors. The model never "understands" language
                the way we do. It just learns useful patterns in these numbers.
            </p>
        </div>
    )
}