import { useState, useEffect, useRef, useCallback } from 'react'

const P = {
    purple: '#7F77DD', purpleL: '#2a2640', purpleD: '#AFA9EC',
    teal: '#1D9E75', tealL: '#1a2e28', tealD: '#5DCAA5',
    amber: '#BA7517', amberL: '#2e2618', amberD: '#EF9F27',
    coral: '#D85A30', coralL: '#2e2018', coralD: '#F0997B',
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

function txt(ctx, t, x, y, size = 11, color, align = 'center', weight = '400') {
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

export default function AttentionDetail({ T, model }) {
    const canvasRef = useRef(null)
    const animRef = useRef(null)
    const stateRef = useRef({ step: 0, dotProgress: 0, playing: false })
    const [info, setInfo] = useState('click animate to watch data flow through attention')

    const STEPS = [
        { msg: `input tensor [${T}x${model.d_model}] enters the attention block` },
        { msg: 'three weight matrices W_Q, W_K, W_V project the input' },
        { msg: `Q [${T}x${model.head_dim}], K [${T}x${model.head_dim}], V [${T}x${model.head_dim}] per head, ${model.n_heads} heads total` },
        { msg: 'K and V are stored in the KV cache, reused on every new token' },
        { msg: `Q x K^T gives raw attention scores [${T}x${T}] per head` },
        { msg: `divide by sqrt(${model.head_dim}) to stabilise, then softmax gives weights that sum to 1.0` },
        { msg: `weights x V gives context vector, concat ${model.n_heads} heads, W_O projects to [${T}x${model.d_model}]` },
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
        const dp = stateRef.current.dotProgress
        const fg = '#c2c0b6'
        const fg2 = '#5e5e5a'
        const bg2 = '#222228'
        const border = 'rgba(255,255,255,0.1)'

        const COL = {
            input: { x: 30, y: 130 },
            wqkv: { x: 140, y: 130 },
            q: { x: 268, y: 56 },
            k: { x: 268, y: 130 },
            v: { x: 268, y: 200 },
            scores: { x: 396, y: 93 },
            softmax: { x: 396, y: 167 },
            output: { x: 524, y: 130 },
        }
        const BW = 90, BH = 48, BHs = 44

        function active(minStep) { return s >= minStep }
        function bFill(ms, fill) { return active(ms) ? fill : bg2 }
        function bStroke(ms, stroke) { return active(ms) ? stroke : border }
        function bTc(ms, tc) { return active(ms) ? tc : fg2 }

        rr(ctx, COL.input.x, COL.input.y - BH / 2, BW, BH, 8, bFill(1, P.purpleL), bStroke(1, P.purple))
        txt(ctx, 'input', COL.input.x + BW / 2, COL.input.y - 6, 11, bTc(1, P.purpleD), 'center', '500')
        txt(ctx, `[${T}x${model.d_model}]`, COL.input.x + BW / 2, COL.input.y + 10, 10, bTc(1, P.purple))

        rr(ctx, COL.wqkv.x, COL.wqkv.y - BH / 2, BW, BH, 8, bFill(2, bg2), bStroke(2, border))
        txt(ctx, 'W_Q  W_K  W_V', COL.wqkv.x + BW / 2, COL.wqkv.y - 6, 9, bTc(2, fg), 'center', '500')
        txt(ctx, `project [${model.d_model}x${model.d_model}]`, COL.wqkv.x + BW / 2, COL.wqkv.y + 10, 8, bTc(2, fg2))

        rr(ctx, COL.q.x, COL.q.y - BHs / 2, BW, BHs, 8, bFill(3, P.purpleL), bStroke(3, P.purple))
        txt(ctx, 'Q', COL.q.x + BW / 2, COL.q.y - 4, 11, bTc(3, P.purpleD), 'center', '600')
        txt(ctx, `[${T}x${model.head_dim}]`, COL.q.x + BW / 2, COL.q.y + 10, 9, bTc(3, P.purple))

        rr(ctx, COL.k.x, COL.k.y - BHs / 2, BW, BHs, 8, bFill(3, P.tealL), bStroke(3, P.teal))
        txt(ctx, 'K', COL.k.x + BW / 2, COL.k.y - 4, 11, bTc(3, P.tealD), 'center', '600')
        txt(ctx, `[${T}x${model.head_dim}]`, COL.k.x + BW / 2, COL.k.y + 10, 9, bTc(3, P.teal))

        rr(ctx, COL.v.x, COL.v.y - BHs / 2, BW, BHs, 8, bFill(3, P.amberL), bStroke(3, P.amber))
        txt(ctx, 'V', COL.v.x + BW / 2, COL.v.y - 4, 11, bTc(3, P.amberD), 'center', '600')
        txt(ctx, `[${T}x${model.head_dim}]`, COL.v.x + BW / 2, COL.v.y + 10, 9, bTc(3, P.amber))

        if (active(4)) {
            ctx.setLineDash([4, 3])
            rr(ctx, COL.k.x - 5, COL.k.y - BHs / 2 - 28, BW + 10, (COL.v.y + BHs / 2) - (COL.k.y - BHs / 2) + 28 + 5, 8, null, P.teal, 1)
            ctx.setLineDash([])
            txt(ctx, 'KV cache', COL.k.x + BW / 2, COL.k.y - BHs / 2 - 16, 9, P.teal, 'center', '500')
        }

        rr(ctx, COL.scores.x, COL.scores.y - BHs / 2, BW, BHs, 8, bFill(5, P.amberL), bStroke(5, P.amber))
        txt(ctx, 'Q x K^T', COL.scores.x + BW / 2, COL.scores.y - 4, 11, bTc(5, P.amberD), 'center', '500')
        txt(ctx, `scores [${T}x${T}]`, COL.scores.x + BW / 2, COL.scores.y + 10, 9, bTc(5, P.amber))

        rr(ctx, COL.softmax.x, COL.softmax.y - BHs / 2, BW, BHs, 8, bFill(6, bg2), bStroke(6, border))
        txt(ctx, 'softmax', COL.softmax.x + BW / 2, COL.softmax.y - 4, 11, bTc(6, fg), 'center', '500')
        txt(ctx, `div sqrt(${model.head_dim}) then weights`, COL.softmax.x + BW / 2, COL.softmax.y + 10, 8, bTc(6, fg2))

        rr(ctx, COL.output.x, COL.output.y - BH / 2, BW, BH, 8, bFill(7, P.purpleL), bStroke(7, P.purple))
        txt(ctx, 'output', COL.output.x + BW / 2, COL.output.y - 6, 11, bTc(7, P.purpleD), 'center', '500')
        txt(ctx, `[${T}x${model.d_model}]`, COL.output.x + BW / 2, COL.output.y + 10, 10, bTc(7, P.purple))

        if (active(1)) arrow(ctx, COL.input.x + BW, COL.input.y, COL.wqkv.x, COL.wqkv.y, P.purple)

        if (active(2)) {
            const fan = COL.wqkv.x + BW + 8
            ctx.strokeStyle = P.purple; ctx.lineWidth = 1
            ctx.beginPath(); ctx.moveTo(COL.wqkv.x + BW, COL.wqkv.y)
            ctx.lineTo(fan, COL.wqkv.y); ctx.lineTo(fan, COL.q.y); ctx.lineTo(COL.q.x, COL.q.y); ctx.stroke()
            dot(ctx, COL.q.x, COL.q.y, P.purple, 3)

            ctx.strokeStyle = P.teal; ctx.beginPath()
            ctx.moveTo(COL.wqkv.x + BW, COL.wqkv.y); ctx.lineTo(COL.k.x, COL.k.y); ctx.stroke()
            dot(ctx, COL.k.x, COL.k.y, P.teal, 3)

            ctx.strokeStyle = P.amber; ctx.beginPath()
            ctx.moveTo(COL.wqkv.x + BW, COL.wqkv.y)
            ctx.lineTo(fan, COL.wqkv.y); ctx.lineTo(fan, COL.v.y); ctx.lineTo(COL.v.x, COL.v.y); ctx.stroke()
            dot(ctx, COL.v.x, COL.v.y, P.amber, 3)
        }

        if (active(5)) {
            ctx.strokeStyle = P.purple; ctx.lineWidth = 1
            ctx.beginPath(); ctx.moveTo(COL.q.x + BW, COL.q.y); ctx.lineTo(COL.scores.x, COL.scores.y); ctx.stroke()
            ctx.strokeStyle = P.teal; ctx.beginPath()
            ctx.moveTo(COL.k.x + BW, COL.k.y); ctx.lineTo(COL.scores.x, COL.scores.y + BHs / 2 - 4); ctx.stroke()
        }
        if (active(5)) arrow(ctx, COL.scores.x + BW, COL.scores.y, COL.softmax.x, COL.softmax.y, P.amber)
        if (active(7)) {
            ctx.strokeStyle = P.amber; ctx.lineWidth = 1; ctx.setLineDash([3, 2])
            ctx.beginPath(); ctx.moveTo(COL.v.x + BW, COL.v.y); ctx.lineTo(COL.output.x, COL.output.y + 10); ctx.stroke()
            ctx.setLineDash([])
        }
        if (active(7)) arrow(ctx, COL.softmax.x + BW, COL.softmax.y, COL.output.x, COL.output.y, P.purple)

        if (stateRef.current.playing && dp < 1) {
            const pts = [
                [COL.input.x + BW / 2, COL.input.y],
                [COL.wqkv.x + BW / 2, COL.wqkv.y],
                [COL.q.x + BW / 2, COL.q.y],
                [COL.scores.x + BW / 2, COL.scores.y],
                [COL.softmax.x + BW / 2, COL.softmax.y],
                [COL.output.x + BW / 2, COL.output.y],
            ]
            const seg = Math.floor(dp * (pts.length - 1))
            const t = (dp * (pts.length - 1)) - seg
            const p1 = pts[Math.min(seg, pts.length - 1)]
            const p2 = pts[Math.min(seg + 1, pts.length - 1)]
            const ex = p1[0] + (p2[0] - p1[0]) * easeInOut(t)
            const ey = p1[1] + (p2[1] - p1[1]) * easeInOut(t)
            dot(ctx, ex, ey, P.purple, 6)
        }

        ctx.restore()
    }, [T, model])

    useEffect(() => {
        draw()
        const handler = () => draw()
        window.addEventListener('resize', handler)
        return () => window.removeEventListener('resize', handler)
    }, [draw])

    function play() {
        if (animRef.current) cancelAnimationFrame(animRef.current)
        stateRef.current = { step: 0, dotProgress: 0, playing: true }
        setInfo(STEPS[0].msg)
        let start = null
        const totalMs = 5600
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
        stateRef.current = { step: 7, dotProgress: 1, playing: false }
        setInfo('click animate to watch data flow through attention')
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
            <canvas ref={canvasRef} style={{ width: '100%', height: 270, display: 'block' }} />
            <div className="kv-note">
                <strong>KV cache:</strong> K and V are computed once and stored during inference. On each new token only Q is recomputed. This is why inference is fast but memory-hungry at long context lengths.
            </div>
        </div>
    )
}