function previewSrc(preview) {
  if (!preview?.preview_jpeg_b64) return null
  return `data:image/jpeg;base64,${preview.preview_jpeg_b64}`
}

export function FrameImg({ preview, alt, style = {}, className }) {
  const src = previewSrc(preview)
  if (!src) return null
  return (
    <img
      src={src}
      alt={alt || preview.timestamp || 'Session frame'}
      className={className}
      style={{ display: 'block', width: '100%', border: '1px solid var(--rule)', ...style }}
    />
  )
}

export function SupervisionList({ prompts }) {
  if (!prompts?.length) return null
  return (
    <div style={{ marginTop: 14 }}>
      <div className="eyebrow">Supervision prompts</div>
      <ul className="small" style={{ paddingLeft: 18, lineHeight: 1.65, marginTop: 8 }}>
        {prompts.map((p, i) => (
          <li key={i} style={{ marginBottom: 8 }}>{p}</li>
        ))}
      </ul>
    </div>
  )
}

export function FrameCard({ preview, caption, question, tags }) {
  if (!previewSrc(preview)) return null
  return (
    <figure
      style={{
        margin: 0,
        border: '1px solid var(--rule)',
        background: 'var(--paper)',
      }}
    >
      <FrameImg preview={preview} style={{ maxHeight: 160, objectFit: 'cover' }} />
      <figcaption style={{ padding: '10px 12px', fontSize: 13, lineHeight: 1.55 }}>
        <span className="mono" style={{ fontWeight: 600 }}>{preview.timestamp}</span>
        {tags?.length > 0 && (
          <div style={{ marginTop: 6, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
            {tags.map((tag) => (
              <span key={tag} className="chip">{tag.replace(/_/g, ' ')}</span>
            ))}
          </div>
        )}
        {caption && <p style={{ margin: '8px 0 0' }}>{caption}</p>}
        {question && (
          <p className="small muted" style={{ marginTop: 8, fontStyle: 'italic' }}>
            {question}
          </p>
        )}
      </figcaption>
    </figure>
  )
}

export default function FramePreview({ preview, caption, question, style = {} }) {
  return (
    <FrameCard preview={preview} caption={caption} question={question} style={style} />
  )
}

export function PreviewGallery({ items = [], title = 'Frame previews' }) {
  if (!items?.length) return null
  return (
    <div style={{ marginTop: 16 }}>
      <div className="eyebrow">{title}</div>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
          gap: 14,
          marginTop: 10,
        }}
      >
        {items.map((p, i) => (
          <FrameCard key={`${p.timestamp}-${i}`} preview={p} />
        ))}
      </div>
    </div>
  )
}
