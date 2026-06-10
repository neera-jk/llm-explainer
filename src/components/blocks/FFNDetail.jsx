import { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'

export default function FFNDetail({ T, model }) {
    const [phase, setPhase] = useState('idle')
    const [barWidths, setBarWidths] = useState([])
    const [barHeights, setBarHeights] = useState([])
    const timerRef = useRef(null)
    const ratio = model.d_ff / model.d_model
    const BAR_COUNT = 28

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
        idle: `input: ${model.d_model} dims per token`,
        expanding: `expanding: ${model.d_model} to ${model.d_ff} dims (${ratio}x wider)...`,
        contracting: `${model.ffn_activation.split(' ')[0]} activation, then contracting back to ${model.d_model}...`,
        done: `output: ${model.d_model} dims (same shape as input)`,
    }

    return (
        <div className="anim-block">
            <div className="anim-controls">
                <button className="anim-btn" onClick={animate}>&#9654; animate</button>
                <button className="anim-btn" onClick={reset}>reset</button>
            </div>
            <div className="anim-info">{labels[phase]}</div>

            <div className="ffn-bars">
                {barWidths.map((w, i) => {
                    const opacity = 0.3 + (barHeights[i] || 30) / 80
                    return (
                        <motion.div
                            key={i}
                            className="ffn-bar"
                            style={{
                                width: w,
                                height: barHeights[i] || 30,
                                background: `rgba(186,117,23,${opacity})`,
                            }}
                            initial={{ height: 0 }}
                            animate={{ height: barHeights[i] || 30, width: w }}
                            transition={{
                                delay: i * 0.02,
                                duration: 0.5,
                                ease: [0.16, 1, 0.3, 1],
                            }}
                        />
                    )
                })}
            </div>

            <div className="ffn-labels">
                <span>input {model.d_model}d</span>
                <span className={`ffn-mid-label ${phase === 'expanding' ? 'ffn-mid-label--visible' : ''}`}>
                    expanded {model.d_ff}d ({ratio}x wider)
                </span>
                <span>output {model.d_model}d</span>
            </div>

            <div className="kv-note" style={{ marginTop: 16 }}>
                <strong>Why expand {ratio}x?</strong> The wider hidden layer lets each token learn more complex transformations. The expand then compress bottleneck forces useful features to survive and irrelevant ones to be discarded.
            </div>
        </div>
    )
}