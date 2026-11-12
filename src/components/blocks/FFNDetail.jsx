import { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { isMoE, ffnWidth } from '../../data/models'
import { ffnShape, isKnownFeature, widthChange } from '../../data/features'

export default function FFNDetail({ T, model, family }) {
    const [phase, setPhase] = useState('idle')
    const [barWidths, setBarWidths] = useState([])
    const [barHeights, setBarHeights] = useState([])
    const timerRef = useRef(null)
    const BAR_COUNT = 28

    const ffnName = family ? family.ffn : 'GELU'
    const shape = family ? ffnShape(family) : { gated: false, activation: 'GELU' }
    const knownFFN = isKnownFeature('ffn', ffnName)
    const isMoEModel = isMoE(model)
    const hasSharedExperts = (model.n_shared_experts || 0) > 0
    const isAlternating = model.alternating_moe === true
    const dFF = ffnWidth(model)
    const widthNote = widthChange(dFF, model.d_model)
    // fine-grained MoE experts can be narrower than the model itself
    const narrower = dFF < model.d_model
    const stretch = narrower ? { width: 3, height: 0.6 } : { width: 18, height: 1.6 }

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
        setBarWidths(prev => prev.map(() => stretch.width))
        setBarHeights(prev => prev.map(h => h * stretch.height))

        timerRef.current = setTimeout(() => {
            setPhase('contracting')
            setBarWidths(prev => prev.map(() => 6))
            setBarHeights(prev => prev.map(h => h / stretch.height))

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
        expanding: `${narrower ? 'Projecting' : 'Expanding'}: ${model.d_model} to ${dFF} numbers (${widthNote})${shape.gated ? ' via gate and up projections in parallel' : ''}...`,
        contracting: `Applying ${ffnName}${shape.gated ? ' gating' : ' activation'}, then ${narrower ? 'projecting' : 'compressing'} back to ${model.d_model}...`,
        done: `Done. Output: [${T}x${model.d_model}]. Same shape as input.`,
    }

    return (
        <div className="anim-block">
            <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.65, marginBottom: 14 }}>
                The feed-forward network processes each token independently (unlike attention, which mixes
                tokens together).
                {shape.gated
                    ? ` ${ffnName} uses three weight matrices. W_gate and W_up both project the input to ${dFF} dimensions in parallel. The gate output goes through a ${shape.activation} activation, then is multiplied element-wise with the up output. This gating lets the model learn which features to keep and which to discard. Finally W_down ${narrower ? 'projects' : 'compresses'} back to ${model.d_model}.`
                    : ffnName === 'GELU'
                        ? ` It first expands the vector to ${dFF} dimensions (${widthNote}), applies GELU activation (which smoothly suppresses negative values), then compresses back to ${model.d_model}. This expand-and-compress bottleneck forces the model to learn which features matter.`
                        : ` It first expands the vector to ${dFF} dimensions (${widthNote}), applies a ${ffnName} activation, then compresses back to ${model.d_model}.`
                }
                {!knownFFN ? ` This explainer doesn't cover ${ffnName} in detail yet, so this shows the general shape of the layer.` : ''}
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
                    {narrower ? 'projected' : 'expanded'} {dFF}d ({widthNote})
                </span>
                <span>output {model.d_model}d</span>
            </div>

            {narrower && isMoEModel ? (
                <div className="kv-note" style={{ marginTop: 16 }}>
                    <strong>Why so narrow?</strong> Each expert here is narrower than the model ({dFF} vs {model.d_model} dimensions).
                    The capacity comes from running several experts at once: {model.n_active_experts} routed
                    {hasSharedExperts ? ` + ${model.n_shared_experts} shared` : ''} experts per token, so together they
                    are {((model.n_active_experts + (model.n_shared_experts || 0)) * dFF).toLocaleString()} dimensions wide.
                </div>
            ) : (
                <div className="kv-note" style={{ marginTop: 16 }}>
                    <strong>Why expand then compress?</strong> The wider hidden layer ({dFF} dimensions)
                    gives the model room to represent complex patterns. Compressing back to {model.d_model} forces
                    it to keep only the most useful information.
                    {shape.gated
                        ? ` ${ffnName} adds a learned gate that controls which features pass through, making it more expressive than a simple expand-activate-compress pattern.${ffnName === 'SwiGLU' ? ' This is why most modern models use SwiGLU over GELU.' : ''}`
                        : ' This bottleneck is one of the key ways transformers learn rich representations from raw data.'
                    }
                </div>
            )}
        </div>
    )
}