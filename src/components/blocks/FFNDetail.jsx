import { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { isMoE, ffnWidth } from '../../data/models'

export default function FFNDetail({ T, model, family }) {
    const [phase, setPhase] = useState('idle')
    const [barWidths, setBarWidths] = useState([])
    const [barHeights, setBarHeights] = useState([])
    const timerRef = useRef(null)
    const BAR_COUNT = 28

    const isSwiGLU = family && family.ffn === 'SwiGLU'
    const isMoEModel = isMoE(model)
    const hasSharedExperts = (model.n_shared_experts || 0) > 0
    const isAlternating = model.alternating_moe === true
    const dFF = ffnWidth(model)
    const ratio = (dFF / model.d_model).toFixed(1)
    const activationName = isSwiGLU ? 'SwiGLU' : 'GELU'

    useEffect(() => {
        const w = []; const h = []
        for (let i = 0; i < BAR_COUNT; i++) {
            w.push(6)
            h.push(16 + Math.random() * 40)
        }
        setBarWidths(w)
        setBarHeights(h)
    }, [])

    function animate() {
        if (timerRef.current) clearTimeout(timerRef.current)

        setPhase('expanding')
        setBarWidths(prev => prev.map(() => 18))
        setBarHeights(prev => prev.map(h => h * 1.6))

        timerRef.current = setTimeout(() => {
            setPhase('contracting')
            setBarWidths(prev => prev.map(() => 6))
            setBarHeights(prev => prev.map(h => h / 1.6))

            timerRef.current = setTimeout(() => {
                setPhase('done')
            }, 600)
        }, 1200)
    }

    function reset() {
        if (timerRef.current) clearTimeout(timerRef.current)
        setPhase('idle')
        const w = []; const h = []
        for (let i = 0; i < BAR_COUNT; i++) {
            w.push(6)
            h.push(16 + Math.random() * 40)
        }
        setBarWidths(w)
        setBarHeights(h)
    }

    const labels = {
        idle: `Input: each token is a vector of ${model.d_model} numbers`,
        expanding: `Expanding: ${model.d_model} to ${dFF} numbers (${ratio}x wider)${isSwiGLU ? ' via gate and up projections in parallel' : ''}...`,
        contracting: `Applying ${activationName}${isSwiGLU ? ' gating' : ' activation'}, then compressing back to ${model.d_model}...`,
        done: `Done. Output: [${T}x${model.d_model}]. Same shape as input.`,
    }

    return (
        <div className="anim-block">
            <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.65, marginBottom: 14 }}>
                The feed-forward network processes each token independently (unlike attention, which mixes
                tokens together).
                {isSwiGLU
                    ? ` SwiGLU uses three weight matrices. W_gate and W_up both project the input to ${dFF} dimensions in parallel. The gate output goes through a swish activation, then is multiplied element-wise with the up output. This gating lets the model learn which features to keep and which to discard. Finally W_down compresses back to ${model.d_model}.`
                    : ` It first expands the vector to ${dFF} dimensions (${ratio}x wider), applies GELU activation (which smoothly suppresses negative values), then compresses back to ${model.d_model}. This expand-and-compress bottleneck forces the model to learn which features matter.`
                }
                {isMoEModel ? ` With MoE, a router first picks ${model.n_active_experts} of ${model.n_experts} expert FFNs for each token. Each expert is a separate copy of this FFN. Only the selected experts run, so cost stays manageable despite the massive total parameter count.` : ''}
                {hasSharedExperts ? ` On top of the routed experts, ${model.n_shared_experts} shared expert runs on every token no matter what, giving the layer a stable baseline that always contributes.` : ''}
                {isAlternating ? ' Not every layer is MoE: dense and MoE feed-forward layers alternate through the stack, which helps keep training stable.' : ''}
                {(model.n_dense_layers || 0) > 0 ? ` The first ${model.n_dense_layers} layers use a plain dense FFN before the MoE layers begin.` : ''}
            </p>

            <div className="anim-controls">
                <button className="anim-btn" onClick={animate}>&#9654; animate</button>
                <button className="anim-btn" onClick={reset}>reset</button>
            </div>
            <div className="anim-info">{labels[phase]}</div>

            <div className="ffn-bars">
                {barWidths.map((w, i) => {
                    const opacity = 0.3 + (barHeights[i] || 30) / 80
                    return (
                        <motion.div key={i} className="ffn-bar"
                            style={{ width: w, height: barHeights[i] || 30, background: `rgba(186,117,23,${opacity})` }}
                            initial={{ height: 0 }}
                            animate={{ height: barHeights[i] || 30, width: w }}
                            transition={{ delay: i * 0.02, duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
                        />
                    )
                })}
            </div>

            <div className="ffn-labels">
                <span>input {model.d_model}d</span>
                <span className={`ffn-mid-label ${phase === 'expanding' ? 'ffn-mid-label--visible' : ''}`}>
                    expanded {dFF}d ({ratio}x wider)
                </span>
                <span>output {model.d_model}d</span>
            </div>

            <div className="kv-note" style={{ marginTop: 16 }}>
                <strong>Why expand then compress?</strong> The wider hidden layer ({dFF} dimensions)
                gives the model room to represent complex patterns. Compressing back to {model.d_model} forces
                it to keep only the most useful information.
                {isSwiGLU
                    ? ' SwiGLU adds a learned gate that controls which features pass through, making it more expressive than a simple expand-activate-compress pattern. This is why most modern models use SwiGLU over GELU.'
                    : ' This bottleneck is one of the key ways transformers learn rich representations from raw data.'
                }
            </div>
        </div>
    )
}