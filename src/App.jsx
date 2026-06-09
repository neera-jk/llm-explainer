import { useState } from 'react'
import { useModel } from './hooks/useModel'
import { MODEL_REGISTRY, MODEL_KEYS } from './data/models'
import { PROMPT_KEYS } from './data/prompts'
import Pipeline from './components/Pipeline'
import BlockModal from './components/BlockModal'

import './styles/base.css'
import './styles/pipeline.css'
import './styles/modal.css'
import './styles/blocks.css'
import './styles/transformer.css'

export default function App() {
  const { modelKey, setModelKey, model, promptKey, setPromptKey, tokens } = useModel()
  const [activeBlock, setActiveBlock] = useState(null)

  function handleModelSwitch(key) {
    setModelKey(key)
    setActiveBlock(null)
  }

  return (
    <div className="app">
      {/* top navigation */}
      <nav className="navbar">
        <span className="nav-logo">
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

      <main className="main-body">
        <p className="model-desc">{model.desc}</p>

        {/* prompt selector */}
        <div className="prompt-section">
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

        {/* token strip — tokenized form of the selected prompt */}
        <div className="tok-section">
          <div className="tok-header">
            <span className="tok-lbl">tokens</span>
            <span className="tok-hint">each word gets an integer ID from the vocabulary</span>
          </div>
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
          </div>
        </div>

        {/* main architecture pipeline */}
        <Pipeline
          model={model}
          tokens={tokens}
          onBlockClick={setActiveBlock}
          activeBlock={activeBlock}
        />

        <p className="hint">click any block to explore its internals</p>
      </main>

      {/* block detail modal */}
      {activeBlock && (
        <BlockModal
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