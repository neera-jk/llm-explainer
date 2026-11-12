// Feature values the explainer has written explanations for. A value not listed
// here gets a generic description instead of borrowing another type's text.
export const KNOWN_FEATURES = {
    attn: ['MHA', 'GQA', 'MQA', 'MLA'],
    norm: ['LayerNorm', 'RMSNorm'],
    posEncoding: ['learned', 'rope', 'irope'],
    ffn: ['GELU', 'SwiGLU', 'GeGLU'],
}

export const FEATURE_NAMES = {
    attn: 'attention type',
    norm: 'normalization',
    posEncoding: 'position encoding',
    ffn: 'feed-forward type',
}

export function isKnownFeature(field, value) {
    return (KNOWN_FEATURES[field] || []).includes(value)
}

export function unknownFeatures(features) {
    return Object.keys(KNOWN_FEATURES)
        .filter(field => !isKnownFeature(field, features[field]))
        .map(field => ({ field, value: features[field] }))
}

// GQA and MQA (GQA with a single K/V head) both let query heads share K and V
export function sharesKV(family) {
    return family.attn === 'GQA' || family.attn === 'MQA'
}

// Gated FFNs (SwiGLU, GeGLU, or an importer-named GLU(act)) use three weight
// matrices; anything else is drawn as a plain up, activate, down FFN.
export function ffnShape(family) {
    if (family.ffn === 'SwiGLU') return { gated: true, activation: 'swish' }
    if (family.ffn === 'GeGLU') return { gated: true, activation: 'GELU' }
    const glu = /^GLU\((.+)\)$/.exec(family.ffn || '')
    if (glu) return { gated: true, activation: glu[1] }
    return { gated: false, activation: family.ffn }
}

// expert FFNs can be narrower than the model, so "wider" isn't always true
export function widthChange(dFF, dModel) {
    const ratio = dFF / dModel
    return ratio >= 1 ? `${ratio.toFixed(1)}x wider` : `${(1 / ratio).toFixed(1)}x narrower`
}
