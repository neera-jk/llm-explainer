import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

const SCORES = [
    [2.1, 0.8, 0.3, 0.5],
    [0.6, 1.9, 1.4, 0.2],
    [0.4, 1.7, 2.3, 0.5],
    [0.9, 1.2, 0.8, 2.0],
]

const WEIGHTS = [
    [42, 22, 14, 22],
    [15, 38, 30, 17],
    [8, 35, 48, 9],
    [18, 24, 16, 42],
]

export default function AttentionDetail({ T, model, family, tokens }) {
    const [step, setStep] = useState(0)

    const isGQA = family && family.attn === 'GQA'
    const isMLA = family && family.attn === 'MLA'
    const hasRoPE = family && family.posEncoding === 'rope'
    const kvHeads = model.n_kv_heads || model.n_heads
    const groupSize = isGQA ? model.n_heads / kvHeads : 1

    const tokenNames = tokens ? tokens.map(t => t.token) : ['The', 'cat', 'sat', 'on']
    const displayTokens = tokenNames.slice(0, 4)

    const steps = [
        {
            title: 'Step 1: project into Q, K, V',
            desc: `The input [${T}x${model.d_model}] is multiplied by three separate weight matrices to create Query, Key, and Value vectors. Q is what each token is "looking for", K is what each token "advertises" about itself, and V is the actual information each token carries.`
                + (isGQA ? ` With GQA, there are ${model.n_heads} Q heads but only ${kvHeads} K/V heads. Groups of ${groupSize} Q heads share the same K and V, making the KV cache ${groupSize}x smaller.` : ` Each of the ${model.n_heads} heads gets its own Q, K, and V of dimension ${model.head_dim}.`)
                + (hasRoPE ? ' After projection, RoPE rotates the Q and K vectors based on their position in the sequence. This encodes where each token sits without needing a separate position embedding.' : ''),
        },
        {
            title: 'Step 2: compute attention scores (Q x K^T)',
            desc: `Each token's Query is compared against every Key via a dot product (multiply matching numbers and sum them up). This produces a ${T}x${T} grid of scores. A high score at row "${displayTokens[2] || 'sat'}", column "${displayTokens[1] || 'cat'}" means the token "${displayTokens[2] || 'sat'}" should pay a lot of attention to "${displayTokens[1] || 'cat'}". The scores are divided by ${Math.round(Math.sqrt(model.head_dim))} (the square root of ${model.head_dim}) to prevent them from growing too large.`,
        },
        {
            title: 'Step 3: softmax turns scores into weights',
            desc: `The raw scores are converted into probabilities using softmax, so each row sums to 100%. Now we can read this as: "${displayTokens[2] || 'sat'}" pays 48% of its attention to itself, 35% to "${displayTokens[1] || 'cat'}" (the subject it describes), 8% to "${displayTokens[0] || 'The'}", and 9% to "${displayTokens[3] || 'on'}". The model learned these patterns during training by seeing millions of sentences.`,
        },
        {
            title: 'Step 4: weights x V = context-aware output',
            desc: `Each token's output is a weighted mix of all Value vectors. So "${displayTokens[2] || 'sat'}"'s output becomes 48% of its own Value + 35% of "${displayTokens[1] || 'cat'}"'s Value + the rest from other tokens. This runs for all ${model.n_heads} Q heads${isGQA ? ` (sharing ${kvHeads} K/V heads)` : ''}. Their outputs are concatenated and projected back to ${model.d_model} dimensions.`,
        },
    ]

    function goStep(n) { setStep(Math.max(0, Math.min(n, steps.length - 1))) }

    return (
        <div className="anim-block">
            <div className="step-nav">
                <div className="step-dots">
                    {steps.map((_, i) => (
                        <div key={i} className={`step-dot ${step === i ? 'step-dot--on' : ''}`} onClick={() => goStep(i)} />
                    ))}
                </div>
                <button className="step-btn" onClick={() => goStep(step - 1)}>prev</button>
                <button className="step-btn" onClick={() => goStep(step + 1)}>next</button>
                <span className="step-label">step {step + 1} of {steps.length}</span>
            </div>

            <AnimatePresence mode="wait">
                <motion.div key={step} className="step-card"
                    initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -20 }}
                    transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}>
                    <div className="step-card-title">{steps[step].title}</div>
                    <div className="step-card-desc">{steps[step].desc}</div>

                    {step === 0 && (
                        <div className="qkv-split">
                            <motion.div className="qkv-box" style={{ background: 'rgba(127,119,221,0.06)', border: '1px solid rgba(127,119,221,0.2)' }}
                                initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.1 }}>
                                <span className="qkv-label" style={{ color: '#AFA9EC' }}>input</span>
                                <span className="qkv-dim" style={{ color: 'rgba(127,119,221,0.5)' }}>[{T}x{model.d_model}]</span>
                            </motion.div>
                            <span className="qkv-arrow">x W =</span>
                            <motion.div className="qkv-box" style={{ background: 'rgba(127,119,221,0.12)', border: '1px solid rgba(127,119,221,0.35)' }}
                                initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2, duration: 0.35 }}>
                                <span className="qkv-label" style={{ color: '#7F77DD' }}>Q</span>
                                <span className="qkv-dim" style={{ color: '#7F77DD' }}>{model.n_heads} x [{T}x{model.head_dim}]</span>
                            </motion.div>
                            <motion.div className="qkv-box" style={{ background: 'rgba(26,138,101,0.1)', border: '1px solid rgba(26,138,101,0.3)' }}
                                initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3, duration: 0.35 }}>
                                <span className="qkv-label" style={{ color: '#1a8a65' }}>K</span>
                                <span className="qkv-dim" style={{ color: '#1a8a65' }}>{kvHeads} x [{T}x{model.head_dim}]</span>
                            </motion.div>
                            <motion.div className="qkv-box" style={{ background: 'rgba(186,117,23,0.1)', border: '1px solid rgba(186,117,23,0.3)' }}
                                initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4, duration: 0.35 }}>
                                <span className="qkv-label" style={{ color: '#BA7517' }}>V</span>
                                <span className="qkv-dim" style={{ color: '#BA7517' }}>{kvHeads} x [{T}x{model.head_dim}]</span>
                            </motion.div>
                            {hasRoPE && (
                                <motion.div style={{ fontSize: 10, color: '#1a8a65', background: 'rgba(26,138,101,0.08)', border: '1px solid rgba(26,138,101,0.2)', borderRadius: 6, padding: '4px 10px', marginLeft: 4 }}
                                    initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }}>
                                    + RoPE on Q, K
                                </motion.div>
                            )}
                        </div>
                    )}

                    {step === 1 && (
                        <div style={{ textAlign: 'center' }}>
                            <div className="attn-matrix" style={{ gridTemplateColumns: `50px repeat(${displayTokens.length}, 48px)` }}>
                                <div />
                                {displayTokens.map((t, i) => <div key={i} className="attn-header">{t}</div>)}
                                {displayTokens.map((t, ri) => (
                                    <div key={ri} style={{ display: 'contents' }}>
                                        <div className="attn-header" style={{ textAlign: 'right', paddingRight: 6 }}>{t}</div>
                                        {SCORES[ri].slice(0, displayTokens.length).map((s, ci) => {
                                            const intensity = s / 2.5 * 0.5
                                            return (
                                                <motion.div key={ci} className="attn-cell"
                                                    style={{ height: 48, background: `rgba(127,119,221,${intensity})`, color: intensity > 0.3 ? '#e8e6e0' : '#AFA9EC' }}
                                                    initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }}
                                                    transition={{ delay: (ri * 4 + ci) * 0.04, duration: 0.3 }}>
                                                    {s.toFixed(1)}
                                                </motion.div>
                                            )
                                        })}
                                    </div>
                                ))}
                            </div>
                            <p className="attn-note">each row is a query token, each column is a key token. brighter cells mean higher attention.</p>
                        </div>
                    )}

                    {step === 2 && (
                        <div style={{ textAlign: 'center' }}>
                            <div className="attn-matrix" style={{ gridTemplateColumns: `50px repeat(${displayTokens.length}, 52px)` }}>
                                <div />
                                {displayTokens.map((t, i) => <div key={i} className="attn-header">{t}</div>)}
                                {displayTokens.map((t, ri) => (
                                    <div key={ri} style={{ display: 'contents' }}>
                                        <div className="attn-header" style={{ textAlign: 'right', paddingRight: 6 }}>{t}</div>
                                        {WEIGHTS[ri].slice(0, displayTokens.length).map((w, ci) => {
                                            const intensity = w / 50 * 0.55
                                            const isMax = w === Math.max(...WEIGHTS[ri].slice(0, displayTokens.length))
                                            return (
                                                <motion.div key={ci} className="attn-cell"
                                                    style={{ height: 48, background: `rgba(127,119,221,${intensity})`, color: intensity > 0.3 ? '#e8e6e0' : '#AFA9EC', fontWeight: isMax ? 600 : 400 }}
                                                    initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }}
                                                    transition={{ delay: (ri * 4 + ci) * 0.04, duration: 0.3 }}>
                                                    {w}%
                                                </motion.div>
                                            )
                                        })}
                                    </div>
                                ))}
                            </div>
                            <p className="attn-note">each row sums to 100%. the bolded cell is where each token pays the most attention.</p>
                        </div>
                    )}

                    {step === 3 && (
                        <div style={{ textAlign: 'center', padding: '8px 0' }}>
                            <svg width="380" height="100" viewBox="0 0 380 100">
                                <rect x="0" y="34" width="66" height="32" rx="4" fill="rgba(127,119,221,0.1)" stroke="rgba(127,119,221,0.2)" strokeWidth="0.5" />
                                <text x="33" y="54" fontSize="11" fill="#AFA9EC" textAnchor="middle" fontFamily="var(--font-mono)" fontWeight="500">weights</text>
                                <text x="80" y="54" fontSize="14" fill="var(--text-muted)" textAnchor="middle">x</text>
                                <rect x="94" y="34" width="36" height="32" rx="4" fill="rgba(186,117,23,0.1)" stroke="rgba(186,117,23,0.2)" strokeWidth="0.5" />
                                <text x="112" y="54" fontSize="11" fill="#BA7517" textAnchor="middle" fontFamily="var(--font-mono)" fontWeight="500">V</text>
                                <text x="148" y="54" fontSize="14" fill="var(--text-muted)" textAnchor="middle">=</text>
                                <rect x="162" y="26" width="80" height="48" rx="6" fill="rgba(127,119,221,0.12)" stroke="rgba(127,119,221,0.3)" strokeWidth="0.5" />
                                <text x="202" y="46" fontSize="11" fill="#AFA9EC" textAnchor="middle" fontFamily="var(--font-mono)" fontWeight="500">context</text>
                                <text x="202" y="62" fontSize="9" fill="rgba(127,119,221,0.5)" textAnchor="middle" fontFamily="var(--font-mono)">[{T}x{model.head_dim}]</text>
                                <text x="264" y="46" fontSize="10" fill="var(--text-muted)" textAnchor="middle">x{model.n_heads} heads</text>
                                <text x="264" y="60" fontSize="10" fill="var(--text-muted)" textAnchor="middle">concat</text>
                                <line x1="290" y1="50" x2="308" y2="50" stroke="var(--border-soft)" strokeWidth="1" />
                                <polygon points="306,46 316,50 306,54" fill="var(--border-soft)" />
                                <rect x="318" y="34" width="56" height="32" rx="4" fill="rgba(127,119,221,0.08)" stroke="rgba(127,119,221,0.25)" strokeWidth="0.5" />
                                <text x="346" y="54" fontSize="9" fill="#AFA9EC" textAnchor="middle" fontFamily="var(--font-mono)">[{T}x{model.d_model}]</text>
                            </svg>
                            <p className="attn-note">{model.n_heads} Q heads{isGQA ? `, ${kvHeads} KV heads` : ''} of {model.head_dim} dimensions each, concatenated back to {model.d_model} dimensions.</p>
                        </div>
                    )}
                </motion.div>
            </AnimatePresence>
        </div>
    )
}