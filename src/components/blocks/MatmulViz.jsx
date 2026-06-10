import { useState, useEffect, useRef, useCallback } from 'react'

function rr(ctx, x, y, w, h, r, fill, stroke, sw = 1) {
    ctx.beginPath()
    ctx.moveTo(x + r, y); ctx.lineTo(x + w - r, y); ctx.arcTo(x + w, y, x + w, y + r, r)
    ctx.lineTo(x + w, y + h - r); ctx.arcTo(x + w, y + h, x + w - r, y + h, r)
    ctx.lineTo(x + r, y + h); ctx.arcTo(x, y + h, x, y + h - r, r)
    ctx.lineTo(x, y + r); ctx.arcTo(x, y, x + r, y, r); ctx.closePath()
    if (fill) { ctx.fillStyle = fill; ctx.fill() }
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = sw; ctx.stroke() }
}

function txt(ctx, t, x, y, size, color, align = 'center') {
    ctx.font = `${size}px -apple-system,monospace`
    ctx.fillStyle = color; ctx.textAlign = align
    ctx.fillText(t, x, y)
}

const P = {
    purple: '#7F77DD', purpleL: '#2a2640', purpleD: '#AFA9EC',
    teal: '#1D9E75', tealL: '#1a2e28', tealD: '#5DCAA5',
    coral: '#D85A30', coralL: '#2e2018', coralD: '#F0997B',
}

export default function MatmulViz() {
    const canvasRef = useRef(null)
    const animRef = useRef(null)
    const [mmRow, setMmRow] = useState(-1)
    const [mmCol, setMmCol] = useState(-1)
    const [info, setInfo] = useState('row x col gives a dot product for one output cell')

    const R = 4, K = 4, C = 4, CS = 30, GAP = 3

    function sr(s, i) {
        const x = Math.sin(s * 127.1 + i * 311.7) * 43758.5453
        return x - Math.floor(x)
    }

    const mA = Array.from({ length: R }, (_, r) =>
        Array.from({ length: K }, (_, c) => parseFloat((sr(r * 4 + c, 0) * 0.8 + 0.1).toFixed(1))))
    const mB = Array.from({ length: K }, (_, r) =>
        Array.from({ length: C }, (_, c) => parseFloat((sr(r * 4 + c, 7) * 0.8 + 0.1).toFixed(1))))
    const mC = Array.from({ length: R }, (_, r) =>
        Array.from({ length: C }, (_, c) =>
            parseFloat((mA[r].reduce((s, v, k) => s + v * mB[k][c], 0) / K).toFixed(2))))

    const draw = useCallback((row, col) => {
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

        const fg2 = '#5e5e5a'
        const bg2 = '#222228'
        const border = 'rgba(255,255,255,0.1)'

        const scale = Math.min((W - 16) / 500, 1)
        ctx.save(); ctx.translate(8, 0); ctx.scale(scale, 1)

        const aX = 0, bX = aX + (K + 1) * (CS + GAP) + 36, eqX = bX + C * (CS + GAP) + 16, cX = eqX + 18
        const aY = 36

        function drawMat(mat, ox, oy, hiR, hiC, role) {
            const rows = mat.length, cols = mat[0].length
            txt(ctx, `${role}  [${rows}x${cols}]`, ox + (cols * (CS + GAP) - GAP) / 2, oy - 12, 9, fg2)
            mat.forEach((rowArr, r) => {
                rowArr.forEach((v, c) => {
                    const x = ox + c * (CS + GAP), y = oy + r * (CS + GAP)
                    let fill = bg2, stroke = border, sw_ = 0.5, tc = fg2
                    if (role === 'A' && r === hiR) { fill = P.purpleL; stroke = P.purple; sw_ = 1.5; tc = P.purpleD }
                    if (role === 'B' && c === hiC) { fill = P.tealL; stroke = P.teal; sw_ = 1.5; tc = P.tealD }
                    if (role === 'C' && r === hiR && c === hiC) { fill = P.coralL; stroke = P.coral; sw_ = 2; tc = P.coralD }
                    rr(ctx, x, y, CS, CS, 4, fill, stroke, sw_)
                    txt(ctx, v.toFixed(1), x + CS / 2, y + CS / 2 + 4, 9, tc)
                })
            })
        }

        drawMat(mA, aX, aY, row, -1, 'A')
        txt(ctx, 'x', aX + K * (CS + GAP) + 10, aY + R * (CS + GAP) / 2 + 4, 16, fg2)
        drawMat(mB, bX, aY, -1, col, 'B')
        txt(ctx, '=', eqX, aY + K * (CS + GAP) / 2 + 4, 16, fg2)
        drawMat(mC, cX, aY, row, col, 'C')

        if (row >= 0 && col >= 0) {
            const dotStr = mA[row].map((v, k) => `${v.toFixed(1)}*${mB[k][col].toFixed(1)}`).join(' + ')
            txt(ctx, `C[${row}][${col}] = (${dotStr}) / ${K} = ${mC[row][col]}`, aX, aY + R * (CS + GAP) + 20, 9, P.coralD, 'left')
        }

        ctx.restore(); ctx.restore()
    }, [])

    useEffect(() => {
        draw(mmRow, mmCol)
        const handler = () => draw(mmRow, mmCol)
        window.addEventListener('resize', handler)
        return () => window.removeEventListener('resize', handler)
    }, [draw, mmRow, mmCol])

    function stepForward() {
        if (animRef.current) { clearTimeout(animRef.current); animRef.current = null }
        setMmRow(pr => {
            const pc = mmCol < 0 ? 0 : mmCol
            let idx = (pr < 0 ? -1 : pr * C + pc) + 1
            if (idx >= R * C) idx = 0
            const nr = Math.floor(idx / C), nc = idx % C
            setMmCol(nc)
            setInfo(`row ${nr} of A  x  col ${nc} of B  =  ${mC[nr][nc]}`)
            setTimeout(() => draw(nr, nc), 0)
            return nr
        })
    }

    function stepBack() {
        if (animRef.current) { clearTimeout(animRef.current); animRef.current = null }
        setMmRow(pr => {
            const pc = mmCol < 0 ? 0 : mmCol
            let idx = (pr < 0 ? R * C : pr * C + pc) - 1
            if (idx < 0) idx = R * C - 1
            const nr = Math.floor(idx / C), nc = idx % C
            setMmCol(nc)
            setInfo(`row ${nr} of A  x  col ${nc} of B  =  ${mC[nr][nc]}`)
            setTimeout(() => draw(nr, nc), 0)
            return nr
        })
    }

    function playMM() {
        if (animRef.current) { clearTimeout(animRef.current); animRef.current = null }
        let r = 0, c = 0
        setMmRow(0); setMmCol(0); draw(0, 0)
        setInfo(`row 0 of A  x  col 0 of B  =  ${mC[0][0]}`)
        function next() {
            animRef.current = setTimeout(() => {
                c++; if (c >= C) { c = 0; r++ }
                if (r >= R) { animRef.current = null; return }
                setMmRow(r); setMmCol(c); draw(r, c)
                setInfo(`row ${r} of A  x  col ${c} of B  =  ${mC[r][c]}`)
                next()
            }, 480)
        }
        next()
    }

    return (
        <div className="anim-block">
            <div className="anim-controls">
                <button className="anim-btn" onClick={playMM}>&#9654; animate</button>
                <button className="anim-btn" onClick={stepBack}>&#8592; prev</button>
                <button className="anim-btn" onClick={stepForward}>next &#8594;</button>
            </div>
            <div className="anim-info">{info}</div>
            <canvas ref={canvasRef} style={{ width: '100%', height: 210, display: 'block' }} />
            <div className="kv-note">
                <strong>Matrix multiplication</strong> is the most fundamental operation in a transformer.
                Every time the model projects into Q, K, V, computes attention scores, or runs
                the feed-forward network, it is doing a matrix multiplication. Each output cell
                is computed by multiplying one row of the first matrix with one column of the second
                and summing the results (a dot product).
            </div>
        </div>
    )
}