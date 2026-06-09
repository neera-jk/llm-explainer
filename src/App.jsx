import { useState } from 'react'
import { useModel } from './hooks/useModel'
import { MODEL_REGISTRY, MODEL_KEYS } from './data/models'
import { PROMPT_KEYS } from './data/prompts'
import Pipeline from './components/Pipeline'
import BlockDetail from './components/BlockDetail'

import './styles/theme.css'
import './styles/base.css'
import './styles/pipeline.css'
import './styles/blocks.css'
import './styles/transformer.css'

export default function App() {
  const { modelKey, setModelKey, model, promptKey, setPromptKey, tokens } = useModel()
  const [activeBlock, setActiveBlock] = useState(null)

  function handleModelSwitch(key) {
    setModelKey(key)
    setActiveBlock(null)
  }

  function handleBlockClick(blockId) {
    setActiveBlock(prev => prev === blockId ? null : blockId)
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
            <button
              key={k}
              className={`npill ${modelKey === k ? 'npill-on' : ''}`}
              onClick={() => handleModelSwitch(k)}
            >
              {MODEL_REGISTRY[k].label}
            </button>
          ))}
        </div>
      </nav>

      <div className="intro-strip">
        <div className="intro-content">
          <div>
            <h1 className="intro-title">{model.label} architecture</h1>
            <p className="intro-desc">decoder-only transformer · follow the data from text to probability</p>
          </div>
          <div className="intro-stats">
            {[
              { val: model.n_layers, label: 'layers' },
              { val: model.d_model, label: 'dims' },
              { val: model.n_heads, label: 'heads' },
              { val: model.vocab_size.toLocaleString(), label: 'vocab' },
            ].map(s => (
              <div key={s.label} className="istat">
                <span className="istat-val">{s.val}</span>
                <span className="istat-label">{s.label}</span>
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
                <button
                  key={p}
                  className={`chip ${promptKey === p ? 'chip-active' : ''}`}
                  onClick={() => setPromptKey(p)}
                >
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
              <span className="tok-hint">each word → integer ID from the vocabulary</span>
            </div>
          </div>

        </div>
      </div>

      <Pipeline
        model={model}
        tokens={tokens}
        onBlockClick={handleBlockClick}
        activeBlock={activeBlock}
      />

      {activeBlock && (
        <BlockDetail
          blockId={activeBlock}
          model={model}
          tokens={tokens}
          promptKey={promptKey}
          onClose={() => setActiveBlock(null)}
        />
      )}
    </div>
  )
}