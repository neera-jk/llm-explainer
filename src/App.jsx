import { useState } from 'react'
import { AnimatePresence } from 'framer-motion'
import { useModel } from './hooks/useModel'
import { MODEL_REGISTRY, MODEL_KEYS } from './data/models'
import { PROMPT_KEYS } from './data/prompts'
import Pipeline from './components/Pipeline'
import BlockDetail from './components/BlockDetail'
import Tooltip from './components/Tooltip'

import './styles/theme.css'
import './styles/base.css'
import './styles/pipeline.css'
import './styles/blocks.css'
import './styles/transformer.css'

export default function App() {
  const { modelKey, setModelKey, model, promptKey, setPromptKey, tokens } = useModel()
  const [activeBlock, setActiveBlock] = useState(null)
  const [activeLayer, setActiveLayer] = useState(null)

  function handleModelSwitch(key) {
    setModelKey(key)
    setActiveBlock(null)
    setActiveLayer(null)
  }

  function handleBlockClick(blockId) {
    if (activeBlock === blockId && activeLayer === null) {
      setActiveBlock(null)
    } else {
      setActiveBlock(blockId)
      setActiveLayer(null)
    }
  }

  function handleLayerClick(layerIdx) {
    if (activeLayer === layerIdx) {
      setActiveLayer(null)
      setActiveBlock(null)
    } else {
      setActiveLayer(layerIdx)
      setActiveBlock('transformer')
    }
  }

  function handleClose() {
    setActiveBlock(null)
    setActiveLayer(null)
  }

  return (
    <div className="app">
      <nav className="navbar">
        <span className="nav-logo">
          <span className="nav-logo-dot" />
          LLM Explainer
          <span className="nav-tag">visual</span>
        </span>
        <div className="nav-models">
          {MODEL_KEYS.map(k => (
            <button key={k} className={`npill ${modelKey === k ? 'npill-on' : ''}`}
              onClick={() => handleModelSwitch(k)}>
              {MODEL_REGISTRY[k].label}
            </button>
          ))}
        </div>
      </nav>

      <div className="intro-strip">
        <div className="intro-content">
          <div>
            <h1 className="intro-title">{model.label} architecture</h1>
            <p className="intro-desc">{model.desc}</p>
          </div>
          <div className="intro-stats">
            {[
              { val: model.n_layers, label: 'layers', tip: 'layers' },
              { val: model.d_model, label: 'dims', tip: 'dims' },
              { val: model.n_heads, label: 'heads', tip: 'heads' },
              { val: model.vocab_size.toLocaleString(), label: 'vocab', tip: 'vocab' },
            ].map(s => (
              <div key={s.label} className="istat">
                <span className="istat-val">{s.val}</span>
                <Tooltip term={s.tip}>
                  <span className="istat-label">{s.label}</span>
                </Tooltip>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="prompt-bar">
        <div className="prompt-bar-inner">
          <div>
            <div className="prompt-label">prompt</div>
            <div className="prompt-chips">
              {PROMPT_KEYS.map(p => (
                <button key={p} className={`chip ${promptKey === p ? 'chip-active' : ''}`}
                  onClick={() => setPromptKey(p)}>
                  {p}
                </button>
              ))}
            </div>
          </div>
          <div className="tok-section">
            <div className="prompt-label">tokens</div>
            <div className="tok-row">
              {tokens.map((t, i) => (
                <div key={i} className="tok-chip">
                  <span className="tok-text">{t.token}</span>
                  <div className="tok-id-wrap">
                    <span className="tok-id">{t.id}</span>
                    <span className="tok-id-label">id</span>
                  </div>
                </div>
              ))}
              <span className="tok-hint">each word maps to an integer ID from the vocabulary</span>
            </div>
          </div>
        </div>
      </div>

      <Pipeline
        model={model} tokens={tokens}
        onBlockClick={handleBlockClick} onLayerClick={handleLayerClick}
        activeBlock={activeBlock} activeLayer={activeLayer}
      />

      <AnimatePresence mode="wait">
        {activeBlock !== null && (
          <BlockDetail
            key={activeBlock === 'transformer' ? `layer-${activeLayer}` : activeBlock}
            blockId={activeBlock} layerIndex={activeLayer}
            model={model} tokens={tokens} promptKey={promptKey}
            onClose={handleClose}
          />
        )}
      </AnimatePresence>
    </div>
  )
}