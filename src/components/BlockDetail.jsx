import { motion } from 'framer-motion'
import TokenizerBlock from './blocks/TokenizerBlock'
import EmbeddingBlock from './blocks/EmbeddingBlock'
import TransformerBlock from './blocks/TransformerBlock'
import UnembedBlock from './blocks/UnembedBlock'
import SoftmaxBlock from './blocks/SoftmaxBlock'

const BLOCK_COMPONENTS = {
  tokenizer: TokenizerBlock,
  embedding: EmbeddingBlock,
  unembed: UnembedBlock,
  softmax: SoftmaxBlock,
}

const BLOCK_LABELS = {
  tokenizer: 'Tokenizer',
  embedding: 'Embedding',
  unembed: 'Unembed',
  softmax: 'Softmax',
}

const BLOCK_ACCENTS = {
  tokenizer: { color: '#888780', light: 'rgba(136,135,128,0.1)', border: 'rgba(136,135,128,0.25)', text: '#9c9a92' },
  embedding: { color: '#7F77DD', light: 'rgba(127,119,221,0.1)', border: 'rgba(127,119,221,0.3)', text: '#AFA9EC' },
  unembed: { color: '#b85428', light: 'rgba(184,84,40,0.1)', border: 'rgba(184,84,40,0.3)', text: '#d4896b' },
  softmax: { color: '#1a8a65', light: 'rgba(26,138,101,0.1)', border: 'rgba(26,138,101,0.3)', text: '#4db896' },
}

export default function BlockDetail({ blockId, layerIndex, model, tokens, promptKey, onClose }) {
  const isLayer = blockId === 'transformer'
  const accent = isLayer ? { color: '#BA7517', text: '#EF9F27' } : BLOCK_ACCENTS[blockId]
  const label = isLayer ? `Layer ${layerIndex}` : BLOCK_LABELS[blockId]
  const BlockContent = isLayer ? null : BLOCK_COMPONENTS[blockId]

  return (
    <motion.div className="detail-panel"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 20 }}
      transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}>
      <div className="detail-inner">
        <div className="detail-header">
          <h2 className="detail-title" style={{ color: accent.text }}>
            <span className="detail-title-dot" style={{ background: accent.color }} />
            {label}
          </h2>
          <button className="detail-close" onClick={onClose}>close ✕</button>
        </div>
        <div className="detail-body">
          {isLayer ? (
            <TransformerBlock model={model} tokens={tokens} layerIndex={layerIndex} />
          ) : BlockContent ? (
            <BlockContent model={model} tokens={tokens} promptKey={promptKey} accent={accent} />
          ) : null}
        </div>
      </div>
    </motion.div>
  )
}