import { useState, useEffect, useRef, useCallback } from 'react'

const P = {
    purple: '#7F77DD', purpleL: '#2a2640', purpleD: '#AFA9EC',
    amber: '#BA7517', amberL: '#2e2618', amberD: '#EF9F27',
}

function rr(ctx, x, y, w, h, r, fill, stroke, sw = 1) {
    ctx.beginPath()
    ctx.moveTo(x + r, y); ctx.lineTo(x + w - r, y); ctx.arcTo(x + w, y, x + w, y + r, r)
    ctx.lineTo(x + w, y + h - r); ctx.arcTo(x + w, y + h, x + w - r, y + h, r)
    ctx.lineTo(x + r, y + h); ctx.arcTo(x, y + h, x, y + h - r, r)
    ctx.lineTo(x, y + r); ctx.arcTo(x, y, x + r, y, r); ctx.closePath()
    if (fill) { ctx.fillStyle = fill; ctx.fill() }
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = sw; ctx.stroke() }
}

function arrow(ctx, x1, y1, x2, y2, color, lw = 1.2) {
    const a = Math.atan2(y2 - y1, x2 - x1), hs = 6
    ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.fillStyle = color
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke()
    ctx.beginPath()
    ctx.moveTo(x2, y2)
    ctx.lineTo(x2 - hs * Math.cos(a - 0.4), y2 - hs * Math.sin(a - 0.4))
    ctx.lineTo(x2 - hs * Math.cos(a + 0.4), y2 - hs * Math.sin(a + 0.4))
    ctx.closePath(); ctx.fill()
}

function txt(ctx, t, x, y, size, color, align = 'center', weight = '400') {
    ctx.font = `${weight} ${size}px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif`
    ctx.fillStyle = color; ctx.textAlign = align
    ctx.fillText(t, x, y)
}

function dot(ctx, x, y, color, r = 5) {
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2)
    ctx.fillStyle = color; ctx.fill()
    ctx.beginPath(); ctx.arc(x, y, r * 0.4, 0, Math.PI * 2)
    ctx.fillStyle = 'rgba(255,255,255,0.6)'; ctx.fill()
}

function easeInOut(t) { return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t }

export default function FFNDetail({ T, model }) {
    const canvasRef = useRef(null)
    const animRef = useRef(null)
    const stateRef = useRef({ step: 0, playing: false })
    const [info, setInfo] = useState('click animate to watch data expand and contract')

    const ratio = model.d_ff / model.d_model

    const STEPS = [
        { msg: `input [${T}x${model.d_model}] enters the FFN, one ${model.d_model}-dim vector per token` },
        { msg: `W_up [${model.d_model}x${model.d_ff}] expands each token to ${model.d_ff} dims` },
        { msg: `hidden [${T}x${model.d_ff}] is ${ratio}x wider, holding richer per-token features` },
        { msg: `${model.ffn_activation} activation adds non-linearity, shape stays [${T}x${model.d_ff}]` },
        { msg: `W_down [${model.d_ff}x${model.d_model}] projects back to model dimension` },
        { msg: `output [${T}x${model.d_model}], same shape as input. residual connection added after` },
    ]

    const draw = useCallback(() => {
        const cv = canvasRef.current
        if (!cv) return
        const dpr = window.devicePixelRatio || 1
        const W = cv.offsetWidth, H = cv.offsetHeight
        if (cv.width !== W * dpr || cv.height !== H * dpr) {
            cv.width = W * dpr; cv.height = H * dpr
        }
        const ctx = cv.getContext('2d')
        ctx.save(); ctx.scale(dpr, dpr)
        ctx.clearRect(0, 0, W, H)

        const s = stateRef.current.step
        const fg = '#c2c0b6'
        const fg2 = '#5e5e5a'
        const bg2 = '#222228'
        const border = 'rgba(255,255,255,0.1)'
        const CY = 100

        const nodes = [
            { id: 'input', x: 16, w: 82, h: 50, label: 'input', shape: `[${T}x${model.d_model}]`, fill: P.purpleL, stroke: P.purple, tc: P.purpleD, minStep: 1 },
            { id: 'wup', x: 126, w: 78, h: 50, label: 'W_up', shape: `[${model.d_model}x${model.d_ff}]`, fill: P.amberL, stroke: P.amber, tc: P.amberD, minStep: 2 },
            { id: 'hidden', x: 232, w: 102, h: 86, label: 'hidden', shape: `[${T}x${model.d_ff}]`, fill: P.amberL, stroke: P.amber, tc: P.amberD, minStep: 3 },
            { id: 'act', x: 362, w: 78, h: 86, label: model.ffn_activation.split(' ')[0], shape: 'elementwise', fill: bg2, stroke: border, tc: fg, minStep: 4 },
            { id: 'wdown', x: 468, w: 78, h: 50, label: 'W_down', shape: `[${model.d_ff}x${model.d_model}]`, fill: P.amberL, stroke: P.amber, tc: P.amberD, minStep: 5 },
            { id: 'output', x: 574, w: 82, h: 50, label: 'output', shape: `[${T}x${model.d_model}]`, fill: P.purpleL, stroke: P.purple, tc: P.purpleD, minStep: 6 },
        ]

        const totalNeeded = 656
        const scale = Math.min((W - 16) / totalNeeded, 1)
        ctx.save(); ctx.translate(8, 0); ctx.scale(scale, 1)

        nodes.forEach((n, i) => {
            const active = s >= n.minStep
            const fill = active ? n.fill : bg2
            const stroke = active ? n.stroke : border
            const tc = active ? n.tc : fg2

            rr(ctx, n.x, CY - n.h / 2, n.w, n.h, 8, fill, stroke)
            txt(ctx, n.label, n.x + n.w / 2, CY - n.h / 2 + 20, 11, tc, 'center', '500')
            txt(ctx, n.shape, n.x + n.w / 2, CY - n.h / 2 + 36, 8, active ? tc : fg2)

            if (n.id === 'hidden' && active) {
                txt(ctx, `${ratio}x wider`, n.x + n.w / 2, CY + n.h / 2 - 8, 9, P.amberD)
            }

            if (i < nodes.length - 1) {
                const nx = nodes[i + 1]
                const arrowActive = s >= nx.minStep
                arrow(ctx, n.x + n.w, CY, nx.x, CY, arrowActive ? P.amber : border, arrowActive ? 1.2 : 0.6)
            }
        })

        if (s >= 3) {
            const hNode = nodes[2]
            rr(ctx, hNode.x, CY - hNode.h / 2 - 20, hNode.w, 14, 3, P.amberL, P.amber, 0.6)
            txt(ctx, `${model.d_ff} dims`, hNode.x + hNode.w / 2, CY - hNode.h / 2 - 10, 9, P.amberD)
        }

        if (stateRef.current.playing) {
            const p = stateRef.current.dotProgress || 0
            const startX = nodes[0].x + nodes[0].w / 2
            const endX = nodes[nodes.length - 1].x + nodes[nodes.length - 1].w / 2
            const ex = startX + (endX - startX) * easeInOut(p)
            dot(ctx, ex, CY, P.amber, 6)
        }

        ctx.restore(); ctx.restore()
    }, [T, model])

    useEffect(() => {
        draw()
        const handler = () => draw()
        window.addEventListener('resize', handler)
        return () => window.removeEventListener('resize', handler)
    }, [draw])

    function play() {
        if (animRef.current) cancelAnimationFrame(animRef.current)
        stateRef.current = { step: 0, playing: true, dotProgress: 0 }
        setInfo(STEPS[0].msg)
        let start = null
        const totalMs = 4800
        function frame(ts) {
            if (!start) start = ts
            const p = Math.min((ts - start) / totalMs, 1)
            const stepIdx = Math.min(Math.floor(p * STEPS.length), STEPS.length - 1)
            stateRef.current.step = stepIdx + 1
            stateRef.current.dotProgress = p
            setInfo(STEPS[stepIdx].msg)
            draw()
            if (p < 1) animRef.current = requestAnimationFrame(frame)
            else stateRef.current.playing = false
        }
        animRef.current = requestAnimationFrame(frame)
    }

    function reset() {
        if (animRef.current) cancelAnimationFrame(animRef.current)
        stateRef.current = { step: 6, playing: false, dotProgress: 1 }
        setInfo('click animate to watch data expand and contract')
        draw()
    }

    useEffect(() => { reset() }, [])

    return (
        <div className="anim-block">
            <div className="anim-controls">
                <button className="anim-btn" onClick={play}>&#9654; animate</button>
                <button className="anim-btn" onClick={reset}>reset</button>
            </div>
            <div className="anim-info">{info}</div>
            <canvas ref={canvasRef} style={{ width: '100%', height: 200, display: 'block' }} />
            <div className="kv-note">
                <strong>Why expand {ratio}x?</strong> The wider hidden layer lets each token learn more complex transformations. The expansion then compression is a bottleneck that forces useful features to survive.
            </div>
        </div>
    )
}