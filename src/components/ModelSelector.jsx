import { useState, useRef, useEffect } from 'react'
import { MODEL_REGISTRY, MODEL_GROUPS } from '../data/models'

export default function ModelSelector({ modelKey, onSelect }) {
    const [open, setOpen] = useState(false)
    const ref = useRef(null)
    const model = MODEL_REGISTRY[modelKey]

    useEffect(() => {
        function handleKey(e) {
            if (e.key === 'Escape') setOpen(false)
        }
        document.addEventListener('keydown', handleKey)
        return () => document.removeEventListener('keydown', handleKey)
    }, [])

    function handleSelect(key) {
        onSelect(key)
        setOpen(false)
    }

    return (
        <div className="model-sel" ref={ref}>
            <button className="sel-btn" onClick={() => setOpen(!open)}>
                <span>{model.label}</span>
                <span className="sel-btn-params">{model.params}</span>
                <span className={`sel-chevron ${open ? 'sel-chevron--open' : ''}`}>&#9662;</span>
            </button>

            {open && <div className="sel-backdrop" onClick={() => setOpen(false)} />}

            <div className={`sel-dropdown ${open ? 'sel-dropdown--open' : ''}`}>
                {MODEL_GROUPS.map(group => (
                    <div key={group.family} className="sel-group">
                        <div className="sel-group-label">{group.label}</div>
                        {group.models.map(key => {
                            const m = MODEL_REGISTRY[key]
                            return (
                                <button
                                    key={key}
                                    className={`sel-item ${modelKey === key ? 'sel-item--active' : ''}`}
                                    onClick={() => handleSelect(key)}
                                >
                                    <span className={`sel-item-dot sel-dot--${group.family}`} />
                                    <div className="sel-item-info">
                                        <span className="sel-item-name">{m.label}</span>
                                        <span className="sel-item-desc">{m.desc}</span>
                                    </div>
                                    <span className="sel-item-params">{m.params}</span>
                                </button>
                            )
                        })}
                    </div>
                ))}
            </div>
        </div>
    )
}