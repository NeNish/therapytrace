import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'

import { api, DIMENSION_LABELS } from '../lib/api'

export default function SessionDiff() {
  const { id } = useParams()
  const [trajData, setTrajData] = useState(null)
  const [sessAId, setSessAId] = useState('')
  const [sessBId, setSessBId] = useState('')
  const [sessA, setSessA] = useState(null)
  const [sessB, setSessB] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    api.trajectory(id)
      .then((data) => {
        setTrajData(data)
        const sessions = data.sessions || []
        if (sessions.length >= 2) {
          setSessAId(String(sessions[0].id))
          setSessBId(String(sessions[sessions.length - 1].id))
        } else if (sessions.length === 1) {
          setSessAId(String(sessions[0].id))
          setSessBId(String(sessions[0].id))
        }
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [id])

  useEffect(() => {
    if (sessAId) {
      api.session(sessAId).then(setSessA).catch(() => {})
    }
  }, [sessAId])

  useEffect(() => {
    if (sessBId) {
      api.session(sessBId).then(setSessB).catch(() => {})
    }
  }, [sessBId])

  if (loading) return <p className="spinner">Loading comparison data…</p>
  if (error) return <div className="notice">{error}</div>

  const client = trajData?.client || {}
  const sessions = trajData?.sessions || []

  const tpiA = sessA?.tpi ?? 0
  const tpiB = sessB?.tpi ?? 0
  const tpiDelta = tpiB - tpiA

  return (
    <>
      <div className="eyebrow">
        <Link to="/">Cases</Link> / <Link to={`/case/${id}`}>{client.code}</Link> / Comparative Analysis
      </div>
      <h1>Linguistic Shift Comparator</h1>
      <p className="lede">
        Side-by-side comparative inspection between two sessions for <strong>{client.code}</strong>.
        Examines how the client's language evolved across the five core therapeutic process dimensions.
      </p>

      {/* Selector Controls */}
      <div className="panel">
        <div className="grid-2">
          <div className="field">
            <label htmlFor="sess-a">Baseline / Earlier Session (Session A)</label>
            <select
              id="sess-a"
              value={sessAId}
              onChange={(e) => setSessAId(e.target.value)}
            >
              {sessions.map((s) => (
                <option key={s.id} value={s.id}>
                  Session {s.session_number} (TPI: {s.tpi.toFixed(1)})
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label htmlFor="sess-b">Comparison / Later Session (Session B)</label>
            <select
              id="sess-b"
              value={sessBId}
              onChange={(e) => setSessBId(e.target.value)}
            >
              {sessions.map((s) => (
                <option key={s.id} value={s.id}>
                  Session {s.session_number} (TPI: {s.tpi.toFixed(1)})
                </option>
              ))}
            </select>
          </div>
        </div>

        {sessA && sessB && (
          <div className="readout" style={{ marginTop: 8 }}>
            <div className="readout-item">
              <div className="k">Session A TPI</div>
              <div className="v">{tpiA.toFixed(1)}</div>
            </div>
            <div className="readout-item">
              <div className="k">Session B TPI</div>
              <div className="v">{tpiB.toFixed(1)}</div>
            </div>
            <div className="readout-item">
              <div className="k">Net Movement (Δ)</div>
              <div
                className="v"
                style={{ color: tpiDelta >= 0 ? 'var(--gain)' : 'var(--regress)' }}
              >
                {tpiDelta > 0 ? '+' : ''}{tpiDelta.toFixed(1)}
              </div>
            </div>
            <div className="readout-item">
              <div className="k">Client word count Δ</div>
              <div className="v">
                {(sessB.features?.n_client_words || 0) - (sessA.features?.n_client_words || 0)}
              </div>
            </div>
          </div>
        )}
      </div>

      {sessA && sessB && (
        <>
          {/* Dimension Comparison Matrix */}
          <div className="panel">
            <div className="topline" style={{ background: tpiDelta >= 0 ? 'var(--gain)' : 'var(--regress)' }} />
            <h2>5-Dimension Shift Matrix</h2>
            <p className="small muted" style={{ marginTop: -4 }}>
              Comparing within-person z-scores against client's baseline. Positive Δ indicates movement toward constructive therapeutic speech.
            </p>
            <table>
              <thead>
                <tr>
                  <th>Dimension</th>
                  <th className="num">Sess {sessA.session_number} (z)</th>
                  <th className="num">Sess {sessB.session_number} (z)</th>
                  <th className="num">Delta (Δz)</th>
                  <th style={{ width: '40%' }}>Linguistic Trajectory</th>
                </tr>
              </thead>
              <tbody>
                {Object.keys(DIMENSION_LABELS).map((k) => {
                  const zA = sessA.z_scores?.[k] ?? 0
                  const zB = sessB.z_scores?.[k] ?? 0
                  const dz = zB - zA
                  const pct = Math.max(-50, Math.min(50, (dz / 2.5) * 50))
                  return (
                    <tr key={k}>
                      <td><strong>{DIMENSION_LABELS[k]}</strong></td>
                      <td className="num">{zA.toFixed(2)}</td>
                      <td className="num">{zB.toFixed(2)}</td>
                      <td
                        className="num"
                        style={{
                          fontWeight: 600,
                          color: dz > 0 ? 'var(--gain)' : dz < 0 ? 'var(--regress)' : 'var(--ink-3)',
                        }}
                      >
                        {dz > 0 ? '+' : ''}{dz.toFixed(2)}
                      </td>
                      <td>
                        <div className="divergent">
                          <span className="zero" />
                          <i
                            style={{
                              left: dz >= 0 ? '50%' : `${50 + pct}%`,
                              width: `${Math.abs(pct)}%`,
                              background: dz >= 0 ? 'var(--gain)' : 'var(--regress)',
                            }}
                          />
                        </div>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>

          {/* Side by side highest-impact turns */}
          <div className="grid-2">
            <div className="panel">
              <div className="topline" style={{ background: 'var(--ink-2)' }} />
              <h2>Session {sessA.session_number} — Key Driver Turns</h2>
              <div className="eyebrow" style={{ marginTop: 8 }}>Highest process turns:</div>
              {(sessA.drivers?.strongest || []).slice(0, 2).map((d, i) => (
                <blockquote key={i} className="gain">
                  <span className="who">Turn {d.turn + 1} · {d.process_mean.toFixed(2)}</span>
                  {d.text}
                </blockquote>
              ))}
              <div className="eyebrow" style={{ marginTop: 14 }}>Lowest process turns:</div>
              {(sessA.drivers?.weakest || []).slice(0, 2).map((d, i) => (
                <blockquote key={i} className="regress">
                  <span className="who">Turn {d.turn + 1} · {d.process_mean.toFixed(2)}</span>
                  {d.text}
                </blockquote>
              ))}
            </div>

            <div className="panel">
              <div className="topline" style={{ background: 'var(--gain)' }} />
              <h2>Session {sessB.session_number} — Key Driver Turns</h2>
              <div className="eyebrow" style={{ marginTop: 8 }}>Highest process turns:</div>
              {(sessB.drivers?.strongest || []).slice(0, 2).map((d, i) => (
                <blockquote key={i} className="gain">
                  <span className="who">Turn {d.turn + 1} · {d.process_mean.toFixed(2)}</span>
                  {d.text}
                </blockquote>
              ))}
              <div className="eyebrow" style={{ marginTop: 14 }}>Lowest process turns:</div>
              {(sessB.drivers?.weakest || []).slice(0, 2).map((d, i) => (
                <blockquote key={i} className="regress">
                  <span className="who">Turn {d.turn + 1} · {d.process_mean.toFixed(2)}</span>
                  {d.text}
                </blockquote>
              ))}
            </div>
          </div>
        </>
      )}

      <div className="row" style={{ marginTop: 16 }}>
        <Link className="btn ghost" to={`/case/${id}`}>← Back to Case Trace</Link>
        <Link className="btn ghost" to={`/case/${id}/insights`}>Insights & Questions →</Link>
      </div>
    </>
  )
}
