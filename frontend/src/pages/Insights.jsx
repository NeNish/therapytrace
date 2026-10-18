import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'

import { api } from '../lib/api'

const LEVEL_STYLE = {
  none: { color: 'var(--hold)', wash: '#E6EAEA', label: 'Nothing standing out' },
  watch: { color: 'var(--mark)', wash: 'var(--mark-wash)', label: 'Worth watching' },
  elevated: { color: 'var(--regress)', wash: 'var(--regress-wash)', label: 'Raise in supervision' },
}

export default function Insights() {
  const { id } = useParams()
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    api.insights(id).then(setData).catch((e) => setError(e.message))
  }, [id])

  if (error) return <div className="notice">{error}</div>
  if (!data) return <p className="spinner">Reading the case…</p>

  const note = data.narrative || {}
  const rec = data.recommendation || {}
  const warn = rec.early_warning || {}
  const target = rec.targeting || {}
  const next = rec.next_session || {}
  const level = LEVEL_STYLE[warn.level] || LEVEL_STYLE.none

  return (
    <>
      <div className="eyebrow">
        <Link to="/">Cases</Link> / <Link to={`/case/${id}`}>{data.client_code}</Link> / insights
      </div>
      <h1>Session insights</h1>
      <p className="lede">
        Every sentence below reports something measured. None of it is generated
        by a language model, so the same session always produces the same note
        and nothing can be invented.
      </p>

      {/* ---------- the note ---------- */}
      <div className="panel">
        <div className="topline" style={{ background: 'var(--ink)' }} />
        <h2>What happened in the latest session</h2>
        {note.sentences?.length ? (
          <div style={{ fontSize: 16, lineHeight: 1.65 }}>
            {note.sentences.map((s, i) => (
              <p key={i} style={{ margin: '0 0 10px' }}>{s}</p>
            ))}
          </div>
        ) : (
          <p className="muted small">Not enough sessions scored yet.</p>
        )}

        {note.flags?.length > 0 && (
          <div style={{ marginTop: 14 }}>
            {note.flags.map((f, i) => (
              <span key={i} className="chip neg">{f.kind.replace(/_/g, ' ')}</span>
            ))}
          </div>
        )}
      </div>

      {/* ---------- early warning ---------- */}
      <div className="panel">
        <div className="topline" style={{ background: level.color }} />
        <div className="panel-head">
          <h2>Early warning</h2>
          <span className="state-tag" style={{ color: level.color, background: level.wash }}>
            {level.label}
          </span>
        </div>
        <p style={{ marginTop: -2 }}>{warn.message || 'Not enough sessions to assess.'}</p>
        {warn.risk_score != null && (
          <>
            <div className="meter" style={{ marginTop: 10 }}>
              <i style={{ left: 0, width: `${warn.risk_score * 100}%`, background: level.color }} />
            </div>
            <div className="mono small muted" style={{ marginTop: 4 }}>
              risk {(warn.risk_score * 100).toFixed(0)}%
            </div>
          </>
        )}
        {warn.caveat && <p className="footnote" style={{ marginTop: 12 }}>{warn.caveat}</p>}
      </div>

      {/* ---------- where to focus ---------- */}
      <div className="grid-2">
        <div className="panel" style={{ marginBottom: 0 }}>
          <h2>Where to focus</h2>
          {target.available ? (
            <>
              <p className="small">{target.summary}</p>
              <p style={{ fontWeight: 600 }}>{target.suggestion}</p>
              <table style={{ marginTop: 12 }}>
                <thead>
                  <tr><th>Dimension</th><th className="num">vs baseline</th></tr>
                </thead>
                <tbody>
                  {(target.all || []).map((r) => (
                    <tr key={r.dimension}>
                      <td>{r.label}</td>
                      <td className="num" style={{
                        color: r.recent_z > 0 ? 'var(--gain)' : 'var(--regress)',
                      }}>
                        {r.recent_z > 0 ? '+' : ''}{r.recent_z.toFixed(2)} SD
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          ) : (
            <p className="muted small">{target.reason || 'Needs more sessions.'}</p>
          )}
        </div>

        {/* ---------- what to try ---------- */}
        <div className="panel" style={{ marginBottom: 0 }}>
          <div className="topline" style={{ background: 'var(--gain)' }} />
          <h2>What to consider next</h2>
          {next.available ? (
            <>
              <p style={{ fontWeight: 600, fontSize: 15 }}>{next.suggestion}</p>
              <p className="small">{next.rationale}</p>
              {next.avoid && <p className="small muted">{next.avoid}</p>}

              <div className="eyebrow" style={{ marginTop: 14 }}>
                Evidence: {next.evidence_source}
              </div>
              <table>
                <thead>
                  <tr><th>Move</th><th className="num">Score</th><th className="num">n</th></tr>
                </thead>
                <tbody>
                  {(next.ranked || []).map((r) => (
                    <tr key={r.intervention}>
                      <td>{r.label}</td>
                      <td className="num">{r.score}</td>
                      <td className="num">{r.n || '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="footnote" style={{ marginTop: 10 }}>{next.caveat}</p>
            </>
          ) : (
            <p className="muted small">{next.reason || 'Needs more sessions.'}</p>
          )}
        </div>
      </div>

      {/* ---------- suggested questions for next session ---------- */}
      {rec.suggested_questions?.available && (
        <div className="panel" style={{ marginTop: 20 }}>
          <div className="topline" style={{ background: 'var(--mark)' }} />
          <div className="panel-head">
            <div>
              <h2>Suggested Questions to Ask for Session {(rec.suggested_questions.session_number || 0) + 1}</h2>
              <p className="small muted" style={{ margin: 0 }}>
                Synthesized dynamically from the most recent session's data (Session {rec.suggested_questions.session_number}).
              </p>
            </div>
            <span className="state-tag" style={{ color: 'var(--mark)', background: 'var(--mark-wash)', whiteSpace: 'nowrap' }}>
              Targeting: {rec.suggested_questions.target_dimension?.replace(/_/g, ' ')} ({rec.suggested_questions.target_z > 0 ? '+' : ''}{rec.suggested_questions.target_z} SD)
            </span>
          </div>
          <p className="small" style={{ margin: '8px 0 16px', color: 'var(--ink-2)' }}>
            {rec.suggested_questions.clinical_rationale}
          </p>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: 14 }}>
            {rec.suggested_questions.questions?.map((q, idx) => (
              <div key={idx} style={{
                background: '#fff',
                border: '1px solid var(--rule)',
                borderLeft: '4px solid var(--mark)',
                padding: '14px 18px',
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                  <span className="eyebrow" style={{ color: 'var(--ink)', margin: 0, fontWeight: 600 }}>
                    {q.category}
                  </span>
                  {q.metric_driver && (
                    <span className="mono small" style={{
                      background: 'var(--mark-wash)',
                      color: 'var(--mark)',
                      padding: '2px 7px',
                      border: '1px solid var(--mark)',
                      fontSize: 11,
                    }}>
                      Trigger: {q.metric_driver}
                    </span>
                  )}
                </div>
                <div style={{ fontFamily: 'var(--body)', fontSize: 16, fontWeight: 500, color: 'var(--ink)', marginBottom: 8, lineHeight: 1.5 }}>
                  "{q.question}"
                </div>
                <div className="small muted" style={{ borderTop: '1px solid var(--rule-soft)', paddingTop: 6 }}>
                  <strong>Clinical Rationale:</strong> {q.clinical_intent}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="notice plain" style={{ marginTop: 20 }}>
        {data.principle}
      </div>
    </>
  )
}
