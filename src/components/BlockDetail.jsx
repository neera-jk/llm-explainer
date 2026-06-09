import TokenizerBlock from './blocks/TokenizerBlock'
import EmbeddingBlock from './blocks/EmbeddingBlock'
import TransformerBlock from './blocks/TransformerBlock'
import UnembedBlock from './blocks/UnembedBlock'
import SoftmaxBlock from './blocks/SoftmaxBlock'

const BLOCK_COMPONENTS = {
  tokenizer: TokenizerBlock,
  embedding: EmbeddingBlock,
  transformer: TransformerBlock,
  unembed: UnembedBlock,
  softmax: SoftmaxBlock,
}

const BLOCK_LABELS = {
  tokenizer: 'Tokenizer',
  embedding: 'Embedding',
  transformer: 'Transformer layers',
  unembed: 'Unembed',
  softmax: 'Softmax',
}

const BLOCK_ACCENTS = {
  tokenizer: { color: '#888780', light: 'rgba(136,135,128,0.1)', border: 'rgba(136,135,128,0.25)', text: '#9c9a92' },
  embedding: { color: '#7F77DD', light: 'rgba(127,119,221,0.1)', border: 'rgba(127,119,221,0.3)', text: '#AFA9EC' },
  transformer: { color: '#BA7517', light: 'rgba(186,117,23,0.1)', border: 'rgba(186,117,23,0.3)', text: '#EF9F27' },
  unembed: { color: '#D85A30', light: 'rgba(216,90,48,0.1)', border: 'rgba(216,90,48,0.3)', text: '#F0997B' },
  softmax: { color: '#1D9E75', light: 'rgba(29,158,117,0.1)', border: 'rgba(29,158,117,0.3)', text: '#5DCAA5' },
}

export default function BlockDetail({ blockId, model, tokens, promptKey, onClose }) {
  const BlockContent = BLOCK_COMPONENTS[blockId]
  const accent = BLOCK_ACCENTS[blockId]

  return (
    <div className="block-detail-panel">
      <div className="block-detail-inner">
        <div className="block-detail-accent-bar" style={{ background: accent.color }} />
        <div className="block-detail-header">
          <h2 className="block-detail-title" style={{ color: accent.text }}>
            {BLOCK_LABELS[blockId]}
          </h2>
          <button className="block-detail-close" onClick={onClose} aria-label="Close">
            close ✕
          </button>
        </div>
        <div className="block-detail-body">
          {BlockContent && (
            <BlockContent model={model} tokens={tokens} promptKey={promptKey} accent={accent} />
          )}
        </div>
      </div>
    </div>
  )
}