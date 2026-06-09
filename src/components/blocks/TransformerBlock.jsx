import { useState, useEffect, useRef, useCallback } from 'react'

// ── canvas helpers ──────────────────────────────────────────────────
function isDark() {
    return window.matchMedia('(prefers-color-scheme:dark)').matches
}
function theme() {
    const dark = isDark()
    return {
        fg: dark ? '#c2c0b6' : '#3d3d3a',
        fg2: dark ? '#888780' : '#73726c',
        bg: dark ? '#1a1a18' : '#ffffff',
        bg2: dark ? '#2a2a28' : '#f5f4f0',
        border: dark ? 'rgba(255,255,255,0.14)' : 'rgba(0,0,0,0.14)',
    }
}
const P = {
    purple: '#7F77DD', purpleL: '#EEEDFE', purpleD: '#3C3489',
    teal: '#1D9E75', tealL: '#E1F5EE', tealD: '#085041',
    amber: '#BA7517', amberL: '#FAEEDA', amberD: '#633806',
    coral: '#D85A30', coralL: '#FAECE7', coralD: '#712B13',
    green: '#3B6D11', greenL: '#EAF3DE',
}

function rr(ctx, x, y, w, h, r, fill, stroke, sw = 0.5) {
    ctx.beginPath()
    ctx.moveTo(x + r, y); ctx.lineTo(x + w - r, y); ctx.arcTo(x + w, y, x + w, y + r, r)
    ctx.lineTo(x + w, y + h - r); ctx.arcTo(x + w, y + h, x + w - r, y + h, r)
    ctx.lineTo(x + r, y + h); ctx.arcTo(x, y + h, x, y + h - r, r)
    ctx.lineTo(x, y + r); ctx.arcTo(x, y, x + r, y, r); ctx.closePath()
    if (fill) { ctx.fillStyle = fill; ctx.fill() }
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = sw; ctx.stroke() }
}

function arrow(ctx, x1, y1, x2, y2, color, lw = 1.2) {
    const a = Math.atan2(y2 - y1, x2 - x1), hs = 7
    ctx.strokeStyle = color; ctx.lineWidth = lw; ctx.fillStyle = color
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke()
    ctx.beginPath()
    ctx.moveTo(x2, y2)
    ctx.lineTo(x2 - hs * Math.cos(a - .4), y2 - hs * Math.sin(a - .4))
    ctx.lineTo(x2 - hs * Math.cos(a + .4), y2 - hs * Math.sin(a + .4))
    ctx.closePath(); ctx.fill()
}

function txt(ctx, t, x, y, size = 11, color, align = 'center', weight = '400') {
    ctx.font = `${weight} ${size}px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif`
    ctx.fillStyle = color; ctx.textAlign = align
    ctx.fillText(t, x, y)
}

function pill(ctx, label, cx, cy, fill, stroke, tc, fontSize = 10) {
    ctx.font = `${fontSize}px -apple-system,monospace`
    const tw = ctx.measureText(label).width
    const pw = tw + 16, ph = 20
    rr(ctx, cx - pw / 2, cy - ph / 2, pw, ph, 4, fill, stroke, 0.5)
    ctx.fillStyle = tc; ctx.textAlign = 'center'
    ctx.fillText(label, cx, cy + 4)
}

function dot(ctx, x, y, color, r = 5) {
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2)
    ctx.fillStyle = color; ctx.fill()
    ctx.beginPath(); ctx.arc(x, y, r * 0.45, 0, Math.PI * 2)
    ctx.fillStyle = 'rgba(255,255,255,0.7)'; ctx.fill()
}

// easing
function easeInOut(t) { return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t }

// ── ATTENTION CANVAS ────────────────────────────────────────────────
function AttentionCanvas({ T, model }) {
    const canvasRef = useRef(null)
    const animRef = useRef(null)
    const stateRef = useRef({ step: 0, dotProgress: 0, playing: false })
    const [info, setInfo] = useState('hit animate — watch data flow left to right')

    const STEPS = [
        { msg: 'input tensor [4×768] enters the attention block' },
        { msg: 'three weight matrices W_Q, W_K, W_V project the input' },
        { msg: 'Q [4×64], K [4×64], V [4×64] — one set per head, 12 heads total' },
        { msg: 'K and V are stored in the KV cache — reused on every new token' },
        { msg: 'Q × Kᵀ → raw attention scores [4×4] per head — how much to attend' },
        { msg: 'divide by √64=8 to stabilise, then softmax → weights sum to 1.0' },
        { msg: 'weights × V → context vector, concat 12 heads → W_O → [4×768]' },
    ]

    const draw = useCallback(() => {
        const cv = canvasRef.current
        if (!cv) return
        const dpr = window.devicePixelRatio || 1
        const W = cv.offsetWidth, H = cv.offsetHeight
        if (cv.width !== W * dpr) { cv.width = W * dpr; cv.height = H * dpr }
        const ctx = cv.getContext('2d')
        ctx.save(); ctx.scale(dpr, dpr)
        ctx.clearRect(0, 0, W, H)

        const th = theme()
        const s = stateRef.current.step
        const dp = stateRef.current.dotProgress

        // layout — left column: input, right side: branching QKV then reconverge
        const COL = {
            input: { x: 36, y: 130 },
            wqkv: { x: 150, y: 130 },
            q: { x: 280, y: 60 },
            k: { x: 280, y: 130 },
            v: { x: 280, y: 200 },
            scores: { x: 410, y: 95 },
            softmax: { x: 410, y: 165 },
            output: { x: 540, y: 130 },
        }
        const BW = 88, BH = 48, BHs = 44 // small blocks

        function active(minStep) { return s >= minStep }
        function blockFill(minStep, fill) { return active(minStep) ? fill : th.bg2 }
        function blockStroke(minStep, stroke) { return active(minStep) ? stroke : th.border }
        function blockTc(minStep, tc) { return active(minStep) ? tc : th.fg2 }

        // ── blocks ──
        // input
        rr(ctx, COL.input.x, COL.input.y - BH / 2, BW, BH, 8,
            blockFill(1, P.purpleL), blockStroke(1, P.purple), active(1) ? 1 : 0.5)
        txt(ctx, 'input', COL.input.x + BW / 2, COL.input.y - 6, 11, blockTc(1, P.purpleD), 'center', '500')
        txt(ctx, `[${T}×768]`, COL.input.x + BW / 2, COL.input.y + 10, 10, blockTc(1, P.purple))

        // wqkv
        rr(ctx, COL.wqkv.x, COL.wqkv.y - BH / 2, BW, BH, 8,
            blockFill(2, th.bg2), blockStroke(2, th.border), 0.5)
        txt(ctx, 'W_Q  W_K  W_V', COL.wqkv.x + BW / 2, COL.wqkv.y - 6, 9, blockTc(2, th.fg), 'center', '500')
        txt(ctx, 'project [768×768]', COL.wqkv.x + BW / 2, COL.wqkv.y + 9, 9, blockTc(2, th.fg2))

        // Q
        rr(ctx, COL.q.x, COL.q.y - BHs / 2, BW, BHs, 8,
            blockFill(3, P.purpleL), blockStroke(3, P.purple), active(3) ? 1 : 0.5)
        txt(ctx, 'Q', COL.q.x + BW / 2, COL.q.y - 5, 11, blockTc(3, P.purpleD), 'center', '500')
        txt(ctx, `[${T}×64]`, COL.q.x + BW / 2, COL.q.y + 9, 9, blockTc(3, P.purple))

        // K
        rr(ctx, COL.k.x, COL.k.y - BHs / 2, BW, BHs, 8,
            blockFill(3, P.tealL), blockStroke(3, P.teal), active(3) ? 1 : 0.5)
        txt(ctx, 'K', COL.k.x + BW / 2, COL.k.y - 5, 11, blockTc(3, P.tealD), 'center', '500')
        txt(ctx, `[${T}×64]`, COL.k.x + BW / 2, COL.k.y + 9, 9, blockTc(3, P.teal))

        // V
        rr(ctx, COL.v.x, COL.v.y - BHs / 2, BW, BHs, 8,
            blockFill(3, P.amberL), blockStroke(3, P.amber), active(3) ? 1 : 0.5)
        txt(ctx, 'V', COL.v.x + BW / 2, COL.v.y - 5, 11, blockTc(3, P.amberD), 'center', '500')
        txt(ctx, `[${T}×64]`, COL.v.x + BW / 2, COL.v.y + 9, 9, blockTc(3, P.amber))

        // KV cache box
        if (active(4)) {
            rr(ctx, COL.k.x - 4, COL.k.y - BHs / 2 - 28, BW + 8, BHs * 2 + BHs / 2 + 28 + 4, 8,
                null, P.teal, 1)
            ctx.setLineDash([4, 3])
            rr(ctx, COL.k.x - 4, COL.k.y - BHs / 2 - 28, BW + 8, BHs * 2 + BHs / 2 + 28 + 4, 8,
                null, P.teal, 1)
            ctx.setLineDash([])
            txt(ctx, 'KV cache', COL.k.x + BW / 2, COL.k.y - BHs / 2 - 16, 9, P.teal, 'center', '500')
        }

        // scores
        rr(ctx, COL.scores.x, COL.scores.y - BHs / 2, BW, BHs, 8,
            blockFill(5, P.amberL), blockStroke(5, P.amber), active(5) ? 1 : 0.5)
        txt(ctx, 'Q × Kᵀ', COL.scores.x + BW / 2, COL.scores.y - 5, 11, blockTc(5, P.amberD), 'center', '500')
        txt(ctx, `scores [${T}×${T}]`, COL.scores.x + BW / 2, COL.scores.y + 9, 9, blockTc(5, P.amber))

        // softmax
        rr(ctx, COL.softmax.x, COL.softmax.y - BHs / 2, BW, BHs, 8,
            blockFill(6, th.bg2), blockStroke(6, th.border), 0.5)
        txt(ctx, 'softmax', COL.softmax.x + BW / 2, COL.softmax.y - 5, 11, blockTc(6, th.fg), 'center', '500')
        txt(ctx, `÷√64 → weights`, COL.softmax.x + BW / 2, COL.softmax.y + 9, 9, blockTc(6, th.fg2))

        // output
        rr(ctx, COL.output.x, COL.output.y - BH / 2, BW, BH, 8,
            blockFill(7, P.purpleL), blockStroke(7, P.purple), active(7) ? 1 : 0.5)
        txt(ctx, 'output', COL.output.x + BW / 2, COL.output.y - 6, 11, blockTc(7, P.purpleD), 'center', '500')
        txt(ctx, `[${T}×768]`, COL.output.x + BW / 2, COL.output.y + 10, 10, blockTc(7, P.purple))

        // ── arrows ──
        // input → wqkv
        if (active(1)) arrow(ctx, COL.input.x + BW, COL.input.y, COL.wqkv.x, COL.wqkv.y, P.purple)
        // wqkv → Q
        if (active(2)) {
            ctx.strokeStyle = P.purple; ctx.lineWidth = 1; ctx.fillStyle = P.purple
            ctx.beginPath()
            ctx.moveTo(COL.wqkv.x + BW, COL.wqkv.y)
            ctx.lineTo(COL.wqkv.x + BW + 10, COL.wqkv.y)
            ctx.lineTo(COL.wqkv.x + BW + 10, COL.q.y)
            ctx.lineTo(COL.q.x, COL.q.y)
            ctx.stroke()
            dot(ctx, COL.q.x, COL.q.y, P.purple, 4)
        }
        // wqkv → K
        if (active(2)) {
            ctx.strokeStyle = P.teal; ctx.lineWidth = 1
            ctx.beginPath()
            ctx.moveTo(COL.wqkv.x + BW, COL.wqkv.y)
            ctx.lineTo(COL.k.x, COL.k.y)
            ctx.stroke()
            dot(ctx, COL.k.x, COL.k.y, P.teal, 4)
        }
        // wqkv → V
        if (active(2)) {
            ctx.strokeStyle = P.amber; ctx.lineWidth = 1
            ctx.beginPath()
            ctx.moveTo(COL.wqkv.x + BW, COL.wqkv.y)
            ctx.lineTo(COL.wqkv.x + BW + 10, COL.wqkv.y)
            ctx.lineTo(COL.wqkv.x + BW + 10, COL.v.y)
            ctx.lineTo(COL.v.x, COL.v.y)
            ctx.stroke()
            dot(ctx, COL.v.x, COL.v.y, P.amber, 4)
        }
        // Q → scores (from above)
        if (active(5)) {
            ctx.strokeStyle = P.purple; ctx.lineWidth = 1
            ctx.beginPath()
            ctx.moveTo(COL.q.x + BW, COL.q.y)
            ctx.lineTo(COL.scores.x, COL.scores.y)
            ctx.stroke()
        }
        // K → scores (from middle)
        if (active(5)) {
            ctx.strokeStyle = P.teal; ctx.lineWidth = 1
            ctx.beginPath()
            ctx.moveTo(COL.k.x + BW, COL.k.y)
            ctx.lineTo(COL.scores.x, COL.scores.y + BHs / 2 - 4)
            ctx.stroke()
        }
        // scores → softmax
        if (active(5)) arrow(ctx, COL.scores.x + BW, COL.scores.y, COL.softmax.x, COL.softmax.y, P.amber)
        // V → output (skip over softmax)
        if (active(7)) {
            ctx.strokeStyle = P.amber; ctx.lineWidth = 1; ctx.setLineDash([3, 2])
            ctx.beginPath()
            ctx.moveTo(COL.v.x + BW, COL.v.y)
            ctx.lineTo(COL.output.x, COL.output.y + 10)
            ctx.stroke(); ctx.setLineDash([])
        }
        // softmax → output
        if (active(7)) arrow(ctx, COL.softmax.x + BW, COL.softmax.y, COL.output.x, COL.output.y, P.purple)

        // ── animated dot ──
        if (stateRef.current.playing && dp < 1) {
            const paths = [
                [COL.input.x + BW / 2, COL.input.y],
                [COL.wqkv.x + BW / 2, COL.wqkv.y],
                [COL.q.x + BW / 2, COL.q.y],
                [COL.scores.x + BW / 2, COL.scores.y],
                [COL.softmax.x + BW / 2, COL.softmax.y],
                [COL.output.x + BW / 2, COL.output.y],
            ]
            const seg = Math.floor(dp * (paths.length - 1))
            const t = (dp * (paths.length - 1)) - seg
            const p1 = paths[Math.min(seg, paths.length - 1)]
            const p2 = paths[Math.min(seg + 1, paths.length - 1)]
            const ex = p1[0] + (p2[0] - p1[0]) * easeInOut(t)
            const ey = p1[1] + (p2[1] - p1[1]) * easeInOut(t)
            dot(ctx, ex, ey, P.purple, 6)
        }

        ctx.restore()
    }, [T])

    useEffect(() => {
        draw()
        window.addEventListener('resize', draw)
        return () => window.removeEventListener('resize', draw)
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
        setInfo('hit animate — watch data flow left to right')
        draw()
    }

    useEffect(() => { reset() }, [])

    return (
        <div className="anim-block">
            <div className="anim-controls">
                <button className="anim-btn" onClick={play}>▶ animate</button>
                <button className="anim-btn" onClick={reset}>reset</button>
            </div>
            <div className="anim-info">{info}</div>
            <canvas ref={canvasRef} style={{ width: '100%', height: 270, display: 'block' }} />
            <div className="kv-note">
                <strong>KV cache:</strong> K and V are computed once and stored during inference. On each new token only Q is recomputed — this is why inference is fast but memory-hungry at long context lengths.
            </div>
        </div>
    )
}

// ── FFN CANVAS ──────────────────────────────────────────────────────
function FFNCanvas({ T, model }) {
    const canvasRef = useRef(null)
    const animRef = useRef(null)
    const stateRef = useRef({ step: 0, playing: false })
    const [info, setInfo] = useState('hit animate — watch data expand and contract')

    const STEPS = [
        { msg: `input [${T}×768] — one 768-dim vector per token` },
        { msg: 'W_up [768×3072] — multiply to expand each token to 3072 dims' },
        { msg: `hidden state [${T}×3072] — 4× wider, holds richer per-token features` },
        { msg: 'GELU activation — kills negative values, adds non-linearity, shape stays [T×3072]' },
        { msg: 'W_down [3072×768] — project back down to model dimension' },
        { msg: `output [${T}×768] — same shape as input. residual (input + output) added after` },
    ]

    const draw = useCallback(() => {
        const cv = canvasRef.current
        if (!cv) return
        const dpr = window.devicePixelRatio || 1
        const W = cv.offsetWidth, H = cv.offsetHeight
        if (cv.width !== W * dpr) { cv.width = W * dpr; cv.height = H * dpr }
        const ctx = cv.getContext('2d')
        ctx.save(); ctx.scale(dpr, dpr)
        ctx.clearRect(0, 0, W, H)

        const th = theme()
        const s = stateRef.current.step

        // nodes: x position, width, height (height = visual representation of dim size)
        const CY = H / 2 / dpr
        const nodes = [
            { id: 'input', x: 20, w: 80, h: 52, label: 'input', shape: `[${T}×768]`, fill: P.purpleL, stroke: P.purple, tc: P.purpleD, minStep: 1 },
            { id: 'wup', x: 130, w: 76, h: 52, label: 'W_up', shape: '[768×3072]', fill: P.amberL, stroke: P.amber, tc: P.amberD, minStep: 2 },
            { id: 'hidden', x: 236, w: 100, h: 88, label: 'hidden', shape: `[${T}×3072]`, fill: P.amberL, stroke: P.amber, tc: P.amberD, minStep: 3 },
            { id: 'gelu', x: 366, w: 76, h: 88, label: 'GELU', shape: 'elementwise', fill: th.bg2, stroke: th.border, tc: th.fg, minStep: 4 },
            { id: 'wdown', x: 472, w: 76, h: 52, label: 'W_down', shape: '[3072×768]', fill: P.amberL, stroke: P.amber, tc: P.amberD, minStep: 5 },
            { id: 'output', x: 578, w: 80, h: 52, label: 'output', shape: `[${T}×768]`, fill: P.purpleL, stroke: P.purple, tc: P.purpleD, minStep: 6 },
        ]

        // scale to fit canvas width
        const totalNeeded = 658
        const scale = Math.min((W - 20) / totalNeeded, 1)
        ctx.save(); ctx.translate(10, 0); ctx.scale(scale, 1)

        nodes.forEach((n, i) => {
            const active = s >= n.minStep
            const fill = active ? n.fill : th.bg2
            const stroke = active ? n.stroke : th.border
            const tc = active ? n.tc : th.fg2
            const sw = active && n.stroke !== th.border ? 1 : 0.5

            rr(ctx, n.x, CY - n.h / 2, n.w, n.h, 8, fill, stroke, sw)
            txt(ctx, n.label, n.x + n.w / 2, CY - n.h / 2 + 18, 11, tc, 'center', '500')
            txt(ctx, n.shape, n.x + n.w / 2, CY - n.h / 2 + 34, 9, active ? tc : th.fg2)

            if (n.id === 'hidden' && active) {
                txt(ctx, '×4 wider', n.x + n.w / 2, CY + n.h / 2 - 10, 9, P.amberD)
            }

            // arrows
            if (i < nodes.length - 1) {
                const nx = nodes[i + 1]
                const arrowActive = s >= nx.minStep
                const acolor = arrowActive ? P.amber : th.border
                arrow(ctx, n.x + n.w, CY, nx.x, CY, acolor, arrowActive ? 1.5 : 0.5)
            }
        })

        // dim scale visual on right side
        if (s >= 3) {
            const hNode = nodes[2]
            rr(ctx, hNode.x, CY - hNode.h / 2 - 18, hNode.w, 14, 3,
                P.amberL, P.amber, 0.5)
            txt(ctx, '3072 dims', hNode.x + hNode.w / 2, CY - hNode.h / 2 - 8, 9, P.amberD)
        }

        // animated flowing dot
        if (stateRef.current.playing) {
            const p = stateRef.current.dotProgress || 0
            const totalX = nodes[nodes.length - 1].x + nodes[nodes.length - 1].w / 2
            const startX = nodes[0].x + nodes[0].w / 2
            const ex = startX + (totalX - startX) * easeInOut(p)
            dot(ctx, ex, CY, P.amber, 6)
        }

        ctx.restore()
        ctx.restore()
    }, [T, model])

    useEffect(() => {
        draw()
        window.addEventListener('resize', draw)
        return () => window.removeEventListener('resize', draw)
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
        setInfo('hit animate — watch data expand and contract')
        draw()
    }

    useEffect(() => { reset() }, [])

    return (
        <div className="anim-block">
            <div className="anim-controls">
                <button className="anim-btn" onClick={play}>▶ animate</button>
                <button className="anim-btn" onClick={reset}>reset</button>
            </div>
            <div className="anim-info">{info}</div>
            <canvas ref={canvasRef} style={{ width: '100%', height: 200, display: 'block' }} />
            <div className="kv-note">
                <strong>Why expand 4×?</strong> The wider hidden layer lets each token learn more complex transformations. The expansion ratio of 4× is a GPT-2 design choice — other models use different ratios.
            </div>
        </div>
    )
}

// ── MATMUL CANVAS ───────────────────────────────────────────────────
function MatmulCanvas() {
    const canvasRef = useRef(null)
    const animRef = useRef(null)
    const [mmRow, setMmRow] = useState(-1)
    const [mmCol, setMmCol] = useState(-1)
    const [info, setInfo] = useState('row × col → dot product = one output cell')

    const R = 4, K = 4, C = 4, CS = 32, GAP = 3
    function sr(s, i) { const x = Math.sin(s * 127.1 + i * 311.7) * 43758.5453; return x - Math.floor(x) }
    const mA = Array.from({ length: R }, (_, r) => Array.from({ length: K }, (_, c) => parseFloat((sr(r * 4 + c, 0) * 0.8 + 0.1).toFixed(1))))
    const mB = Array.from({ length: K }, (_, r) => Array.from({ length: C }, (_, c) => parseFloat((sr(r * 4 + c, 7) * 0.8 + 0.1).toFixed(1))))
    const mC = Array.from({ length: R }, (_, r) => Array.from({ length: C }, (_, c) => parseFloat((mA[r].reduce((s, v, k) => s + v * mB[k][c], 0) / K).toFixed(2))))

    const draw = useCallback((row, col) => {
        const cv = canvasRef.current
        if (!cv) return
        const dpr = window.devicePixelRatio || 1
        const W = cv.offsetWidth, H = cv.offsetHeight
        if (cv.width !== W * dpr) { cv.width = W * dpr; cv.height = H * dpr }
        const ctx = cv.getContext('2d')
        ctx.save(); ctx.scale(dpr, dpr)
        ctx.clearRect(0, 0, W, H)
        const th = theme()

        const scale = Math.min((W - 20) / 520, 1)
        ctx.save(); ctx.translate(10, 0); ctx.scale(scale, 1)

        const aX = 0, bX = aX + (K + 1) * (CS + GAP) + 44, eqX = bX + C * (CS + GAP) + 20, cX = eqX + 20
        const aY = 40, bY = 40, cY = 40

        function drawMat(mat, ox, oy, hiR, hiC, role, rw, rh) {
            const rows = mat.length, cols = mat[0].length
            txt(ctx, `${role}  [${rows}×${cols}]`, ox + (cols * (CS + GAP) - GAP) / 2, oy - 14, 10, th.fg2)
            mat.forEach((row, r) => {
                row.forEach((v, c) => {
                    const x = ox + c * (CS + GAP), y = oy + r * (CS + GAP)
                    let fill = th.bg2, stroke = th.border, sw_ = 0.5, tc = th.fg2
                    if (role === 'A' && r === hiR) { fill = P.purpleL; stroke = P.purple; sw_ = 1.5; tc = P.purpleD }
                    if (role === 'B' && c === hiC) { fill = P.tealL; stroke = P.teal; sw_ = 1.5; tc = P.tealD }
                    if (role === 'C' && r === hiR && c === hiC) { fill = P.coralL; stroke = P.coral; sw_ = 2; tc = P.coralD }
                    rr(ctx, x, y, CS, CS, 4, fill, stroke, sw_)
                    txt(ctx, v.toFixed(1), x + CS / 2, y + CS / 2 + 4, 10, tc)
                })
            })
        }

        drawMat(mA, aX, aY, row, -1, 'A')
        txt(ctx, '×', aX + K * (CS + GAP) + 14, aY + R * (CS + GAP) / 2 + 4, 18, th.fg2)
        drawMat(mB, bX, bY, -1, col, 'B')
        txt(ctx, '=', eqX, bY + K * (CS + GAP) / 2 + 4, 18, th.fg2)
        drawMat(mC, cX, cY, row, col, 'C')

        if (row >= 0 && col >= 0) {
            const dotStr = mA[row].map((v, k) => `${v.toFixed(1)}×${mB[k][col].toFixed(1)}`).join('+')
            txt(ctx, `C[${row}][${col}] = (${dotStr}) / ${K} = ${mC[row][col]}`, aX, aY + R * (CS + GAP) + 22, 10, P.coral, 'left')
        }

        // legend
        const leg = [{ c: P.purpleL, s: P.purple, l: 'active row' }, { c: P.tealL, s: P.teal, l: 'active col' }, { c: P.coralL, s: P.coral, l: 'result cell' }]
        leg.forEach((l, i) => {
            rr(ctx, aX + i * 130, aY + R * (CS + GAP) + 38, 12, 12, 2, l.c, l.s, 0.5)
            txt(ctx, l.l, aX + i * 130 + 18, aY + R * (CS + GAP) + 49, 10, th.fg2, 'left')
        })

        ctx.restore(); ctx.restore()
    }, [])

    useEffect(() => {
        draw(mmRow, mmCol)
        window.addEventListener('resize', () => draw(mmRow, mmCol))
        return () => window.removeEventListener('resize', () => draw(mmRow, mmCol))
    }, [draw, mmRow, mmCol])

    function step(d) {
        if (animRef.current) { cancelAnimationFrame(animRef.current); animRef.current = null }
        setMmRow(pr => {
            setMmCol(pc => {
                let idx = (pr < 0 ? -1 : pr * C + pc) + d
                if (idx < 0) idx = R * C - 1
                if (idx >= R * C) idx = 0
                const nr = Math.floor(idx / C), nc = idx % C
                setInfo(`row ${nr} of A  ×  col ${nc} of B  →  ${mC[nr][nc]}`)
                draw(nr, nc)
                return nc
            })
            return Math.floor(((pr < 0 ? -1 : pr * C + (mmCol)) + d + R * C) % (R * C) / C)
        })
    }

    function playMM() {
        if (animRef.current) { cancelAnimationFrame(animRef.current); animRef.current = null }
        let r = 0, c = 0
        setMmRow(r); setMmCol(c); draw(r, c)
        setInfo(`row ${r} of A  ×  col ${c} of B  →  ${mC[r][c]}`)
        function next() {
            animRef.current = setTimeout(() => {
                c++; if (c >= C) { c = 0; r++ }
                if (r >= R) { animRef.current = null; return }
                setMmRow(r); setMmCol(c); draw(r, c)
                setInfo(`row ${r} of A  ×  col ${c} of B  →  ${mC[r][c]}`)
                next()
            }, 520)
        }
        next()
    }

    return (
        <div className="anim-block">
            <div className="anim-controls">
                <button className="anim-btn" onClick={playMM}>▶ animate</button>
                <button className="anim-btn" onClick={() => step(-1)}>← prev</button>
                <button className="anim-btn" onClick={() => step(1)}>next →</button>
            </div>
            <div className="anim-info">{info}</div>
            <canvas ref={canvasRef} style={{ width: '100%', height: 220, display: 'block' }} />
        </div>
    )
}

// ── LAYER STACK ─────────────────────────────────────────────────────
function LayerStack({ model, tokens }) {
    const T = tokens.length
    const [activeLayer, setActiveLayer] = useState(null)
    const [activeTab, setActiveTab] = useState('attn')

    const SHOW = activeLayer !== null ? model.n_layers : 3

    return (
        <div>
            <div className="layer-stack-outer">
                {activeLayer === null && (
                    <>
                        <div className="layer-shadow-2" />
                        <div className="layer-shadow-1" />
                    </>
                )}
                {Array.from({ length: SHOW }, (_, i) => (
                    <div key={i}>
                        <div
                            className={`layer-card-new ${activeLayer === i ? 'layer-card-active' : ''}`}
                            onClick={() => {
                                setActiveLayer(activeLayer === i ? null : i)
                                setActiveTab('attn')
                            }}
                        >
                            <div className="layer-card-row">
                                <span className="layer-num-badge">{i}</span>
                                <span className="layer-card-label">transformer layer</span>
                                <span className="lc-a">attn</span>
                                <span className="lc-f">FFN</span>
                                <span className="layer-card-shape">[{T}×{model.d_model}] → [{T}×{model.d_model}]</span>
                                <span className="layer-card-chevron">{activeLayer === i ? '▲' : '▼'}</span>
                            </div>
                        </div>

                        {activeLayer === i && (
                            <div className="layer-detail-box">
                                <div className="layer-detail-tabs">
                                    <button
                                        className={`dtab ${activeTab === 'attn' ? 'dtab-on' : ''}`}
                                        onClick={() => setActiveTab('attn')}
                                    >self-attention</button>
                                    <button
                                        className={`dtab ${activeTab === 'ffn' ? 'dtab-on' : ''}`}
                                        onClick={() => setActiveTab('ffn')}
                                    >FFN</button>
                                    <button
                                        className={`dtab ${activeTab === 'mm' ? 'dtab-on' : ''}`}
                                        onClick={() => setActiveTab('mm')}
                                    >matmul</button>
                                </div>
                                {activeTab === 'attn' && <AttentionCanvas T={T} model={model} />}
                                {activeTab === 'ffn' && <FFNCanvas T={T} model={model} />}
                                {activeTab === 'mm' && <MatmulCanvas />}
                            </div>
                        )}
                    </div>
                ))}
                {activeLayer === null && (
                    <div className="more-layers-label">· · · layers 3 – 11, identical structure · · ·</div>
                )}
            </div>
        </div>
    )
}

// ── MAIN EXPORT ─────────────────────────────────────────────────────
export default function TransformerBlock({ model, tokens }) {
    return (
        <div className="block-content">
            <p className="block-explain">
                GPT-2 has <strong>{model.n_layers} identical transformer layers</strong> stacked on top of each other.
                Each layer runs self-attention then FFN. Input and output shape never change —
                always <strong>[{tokens.length} × {model.d_model}]</strong>.
                Click any layer to see its full architecture and animations.
            </p>

            <div className="stats-row">
                <div className="stat-card">
                    <span className="stat-val">{model.n_layers}</span>
                    <span className="stat-label">layers</span>
                </div>
                <div className="stat-card">
                    <span className="stat-val">{model.n_heads}</span>
                    <span className="stat-label">attn heads</span>
                </div>
                <div className="stat-card">
                    <span className="stat-val">{model.head_dim}</span>
                    <span className="stat-label">head dim</span>
                </div>
                <div className="stat-card">
                    <span className="stat-val">×{model.d_ff / model.d_model}</span>
                    <span className="stat-label">FFN expansion</span>
                </div>
            </div>

            <LayerStack model={model} tokens={tokens} />
        </div>
    )
}
