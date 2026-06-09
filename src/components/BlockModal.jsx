import { motion, AnimatePresence } from 'framer-motion'
import TokenizerBlock from './blocks/TokenizerBlock'
import EmbeddingBlock from './blocks/EmbeddingBlock'
import TransformerBlock from './blocks/TransformerBlock'
import UnembedBlock from './blocks/UnembedBlock'
import SoftmaxBlock from './blocks/SoftmaxBlock'

/** Maps each block ID to its detail component */
const BLOCK_COMPONENTS = {
    tokenizer: TokenizerBlock,
    embedding: EmbeddingBlock,
    transformer: TransformerBlock,
    unembed: UnembedBlock,
    softmax: SoftmaxBlock,
}

/** Display labels for the modal header */
const BLOCK_LABELS = {
    tokenizer: 'Tokenizer',
    embedding: 'Embedding',
    transformer: 'Transformer layers',
    unembed: 'Unembed',
    softmax: 'Softmax',
}

/**
 * Modal that slides up when a pipeline block is clicked.
 * Clicking the overlay background closes it.
 */
export default function BlockModal({ blockId, model, tokens, promptKey, onClose }) {
    const BlockContent = BLOCK_COMPONENTS[blockId]

    return (
        <AnimatePresence>
            <motion.div
                className="modal-overlay"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                onClick={(e) => e.target === e.currentTarget && onClose()}
            >
                <motion.div
                    className="modal"
                    initial={{ y: 80, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    exit={{ y: 80, opacity: 0 }}
                    transition={{ type: 'spring', damping: 28, stiffness: 320 }}
                >
                    <div className="modal-header">
                        <h2 className="modal-title">{BLOCK_LABELS[blockId]}</h2>
                        <button className="modal-close" onClick={onClose} aria-label="Close">✕</button>
                    </div>
                    <div className="modal-body">
                        {BlockContent && (
                            <BlockContent model={model} tokens={tokens} promptKey={promptKey} />
                        )}
                    </div>
                </motion.div>
            </motion.div>
        </AnimatePresence>
    )
}