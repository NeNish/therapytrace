import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'

import { FrameCard, FrameImg, SupervisionList } from '../components/FramePreview'
import { api, DIMENSION_LABELS } from '../lib/api'

function ribbonColor(v) {
  if (v >= 0.58) return 'var(--gain)'
  if (v <= 0.44) return 'var(--regress)'
  return 'var(--hold)'
}

function Evidence({ evidence }) {
  if (!evidence || !Object.keys(evidence).length) return null
  return (
    <div style={{ marginTop: 6 }}>
      {Object.entries(evidence).map(([k, terms]) =>
        terms.map((t) => (
          <span key={`${k}-${t}`} className={`chip ${k.endsWith('+') ? 'pos' : k.endsWith('-') ? 'neg' : ''}`}>
            {t}
          </span>
        ))
      )}
    </div>
  )
}

export default function SessionView() {
  const { sid } = useParams()
  const [s, setS] = useState(null)
  const [error, setError] = useState(null)
  const [pick, setPick] = useState(null)

  useEffect(() => {
    api.session(sid).then(setS).catch((e) => setError(e.message))
  }, [sid])

  if (error) return <div className="notice">{error}</div>
  if (!s) return <p className="spinner">Loading session…</p>

  const series = s.utterance_series || []
  const notes = s.parse_info?.confidence_notes || []
  const warnings = s.parse_info?.warnings || []
  const picked = pick != null ? series[pick] : null

  return (
    <>
      <div className="eyebrow">
        <Link to="/">Cases</Link> / <Link to={`/case/${s.client_id}`}>case</Link> / session {s.session_number}
      </div>
      <h1>Session {s.session_number}</h1>
      <p className="lede">
        TPI {s.tpi.toFixed(1)} · confidence {(s.confidence * 100).toFixed(0)}% ·
        {' '}{s.parse_info?.n_client_turns} client turns,{' '}
        {s.parse_info?.n_therapist_turns} therapist turns
      </p>

      {(notes.length > 0 || warnings.length > 0) && (
        <div className="notice">
          {[...warnings, ...notes].map((n, i) => <div key={i}>{n}</div>)}
        </div>
      )}

      {/* ---------------- utterance ribbon ---------------- */}
      <div className="panel">
        <h2>Turn by turn</h2>
        <p className="small muted" style={{ marginTop: -6 }}>
          One bar per client turn, in order. Height is turn length, colour is
          process score. Click a bar to read it.
        </p>
        <div className="ribbon" role="list">
          {series.map((u, i) => (
            <i
              key={i}
              role="listitem"
              title={`Turn ${i + 1} · ${u.process_mean.toFixed(2)}`}
              onClick={() => setPick(i)}
              style={{
                height: `${Math.max(8, Math.min(60, 8 + u.n_words * 0.55))}px`,
                background: ribbonColor(u.process_mean),
                outline: pick === i ? '2px solid var(--ink)' : 'none',
              }}
            />
          ))}
        </div>

        {picked && (
          <div style={{ marginTop: 16 }}>
            <blockquote className={picked.process_mean >= 0.55 ? 'gain' : picked.process_mean <= 0.45 ? 'regress' : ''}>
              <span className="who">Client · turn {pick + 1} · process {picked.process_mean.toFixed(2)}</span>
              {series[pick].text || <em className="muted">(text not stored for this turn)</em>}
            </blockquote>
            <div className="mono small">
              {Object.keys(DIMENSION_LABELS).map((k) => (
                <span key={k} style={{ marginRight: 16 }}>
                  {DIMENSION_LABELS[k]}: {picked[k]?.toFixed(2)}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* ---------------- drivers ---------------- */}
      <div className="grid-2">
        <div className="panel">
          <div className="topline" style={{ background: 'var(--gain)' }} />
          <h2>Turns that lifted the session</h2>
          {(s.drivers?.strongest || []).map((d, i) => (
            <div key={i} style={{ marginBottom: 16 }}>
              <blockquote className="gain">
                <span className="who">Turn {d.turn + 1} · {d.process_mean.toFixed(2)}</span>
                {d.text}
              </blockquote>
              <Evidence evidence={d.evidence} />
            </div>
          ))}
        </div>

        <div className="panel">
          <div className="topline" style={{ background: 'var(--regress)' }} />
          <h2>Turns that pulled it down</h2>
          {(s.drivers?.weakest || []).map((d, i) => (
            <div key={i} style={{ marginBottom: 16 }}>
              <blockquote className="regress">
                <span className="who">Turn {d.turn + 1} · {d.process_mean.toFixed(2)}</span>
                {d.text}
              </blockquote>
              <Evidence evidence={d.evidence} />
            </div>
          ))}
        </div>
      </div>

      {/* ---------------- therapist moments ---------------- */}
      <div className="panel">
        <h2>Exchanges that opened something up</h2>
        <p className="small muted" style={{ marginTop: -6 }}>
          Therapist turns after which the client's very next turn scored well
          above their level just before it.
        </p>
        {(s.therapist_moments?.openings || []).slice(0, 3).map((m, i) => (
          <div key={i} style={{ marginBottom: 18 }}>
            <blockquote>
              <span className="who">Therapist · {m.intervention} · lift {m.lift > 0 ? '+' : ''}{m.lift}</span>
              {m.therapist_text}
            </blockquote>
            <blockquote className="gain" style={{ marginLeft: 20 }}>
              <span className="who">Client, next turn</span>
              {m.client_next_text}
            </blockquote>
          </div>
        ))}
        {!(s.therapist_moments?.openings || []).length && (
          <p className="small muted">
            No usable turn pairs in this session — the client's replies were too
            short to measure a shift.
          </p>
        )}
      </div>

      {s.parse_info?.multimodal?.insights && (
        <div className="panel">
          <h2>Audio & video insights</h2>
          <p className="small muted" style={{ marginTop: -6 }}>
            From recordings attached when this session was added.
          </p>
          {s.parse_info.multimodal.insights.audio?.available && (
            <div style={{ marginBottom: 16 }}>
              <div className="eyebrow">How it sounded</div>
              <ul className="small" style={{ paddingLeft: 18, lineHeight: 1.7 }}>
                {s.parse_info.multimodal.insights.audio.bullets?.map((b, i) => (
                  <li key={i}>{b}</li>
                ))}
              </ul>
            </div>
          )}
          {s.parse_info.multimodal.insights.audio?.supervision_prompts && (
            <SupervisionList prompts={s.parse_info.multimodal.insights.audio.supervision_prompts} />
          )}
          {s.parse_info.multimodal.insights.video?.available && (
            <>
              <div className="eyebrow">What the body did</div>
              <p style={{ fontSize: 15, lineHeight: 1.65 }}>
                {s.parse_info.multimodal.insights.video.summary}
              </p>
              <SupervisionList prompts={s.parse_info.multimodal.insights.video.supervision_prompts} />
              {s.parse_info.multimodal.preview_gallery?.length > 0 && (
                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
                    gap: 12,
                    marginTop: 16,
                  }}
                >
                  {s.parse_info.multimodal.preview_gallery.map((p, i) => (
                    <FrameCard key={i} preview={p} />
                  ))}
                </div>
              )}
              {s.parse_info.multimodal.insights.video.timeline?.length > 0 && (
                <div style={{ marginTop: 16 }}>
                  {s.parse_info.multimodal.insights.video.timeline.map((t, i) => (
                    <div
                      key={i}
                      style={{
                        display: 'grid',
                        gridTemplateColumns: t.preview ? '100px 1fr' : '1fr',
                        gap: 12,
                        marginBottom: 14,
                        paddingBottom: 14,
                        borderBottom: '1px solid var(--rule)',
                      }}
                    >
                      {t.preview && <FrameImg preview={t.preview} style={{ maxHeight: 80, objectFit: 'cover' }} />}
                      <div>
                        <span className="mono">{t.timestamp}</span>
                        <span className="small muted"> · {t.kind?.replace(/_/g, ' ')}</span>
                        <p className="small" style={{ margin: '4px 0' }}>{t.sentence}</p>
                        {t.supervision_question && (
                          <p className="small muted" style={{ fontStyle: 'italic' }}>{t.supervision_question}</p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
          {!s.parse_info.multimodal.insights.audio?.available &&
           !s.parse_info.multimodal.insights.video?.available && (
            <p className="small muted">No usable audio or video was attached to this session.</p>
          )}
        </div>
      )}

      <details className="panel">
        <summary className="mono small" style={{ cursor: 'pointer' }}>Show raw transcript</summary>
        <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'var(--data)', fontSize: 12.5, marginTop: 12 }}>
          {s.transcript}
        </pre>
      </details>
    </>
  )
}
