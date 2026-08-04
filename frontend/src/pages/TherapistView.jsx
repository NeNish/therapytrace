import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'

import { api } from '../lib/api'

export default function TherapistView() {
  const { id } = useParams()
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    api.therapistImpact(id).then(setData).catch((e) => setError(e.message))
  }, [id])

  if (error) return <div className="notice">{error}</div>
  if (!data) return <p className="spinner">Loading…</p>

  const max = Math.max(0.02, ...data.interventions.map((r) => Math.abs(r.mean_lift)))

  return (
    <>
      <div className="eyebrow">
        <Link to="/">Cases</Link> / <Link to={`/case/${id}`}>{data.client_code}</Link> / therapist
      </div>
      <h1>Therapist contribution</h1>
      <p className="lede">
        For every therapist turn, the client's next turn is compared with their
        turn just before it. A positive lift means the client's language opened
        up right after that kind of move.
      </p>

      <div className="notice">{data.caveat}</div>

      <div className="panel flush">
        <table>
          <thead>
            <tr>
              <th>Intervention</th>
              <th className="num">Uses</th>
              <th className="num">Mean lift</th>
              <th style={{ width: '45%' }}>Direction</th>
            </tr>
          </thead>
          <tbody>
            {data.interventions.map((r) => {
              const pct = Math.max(-50, Math.min(50, (r.mean_lift / max) * 50))
              return (
                <tr key={r.intervention}>
                  <td>{r.label}</td>
                  <td className="num">{r.n}</td>
                  <td className="num" style={{ color: r.mean_lift >= 0 ? 'var(--gain)' : 'var(--regress)' }}>
                    {r.mean_lift > 0 ? '+' : ''}{r.mean_lift.toFixed(3)}
                  </td>
                  <td>
                    <div className="divergent">
                      <span className="zero" />
                      <i
                        style={{
                          left: r.mean_lift >= 0 ? '50%' : `${50 + pct}%`,
                          width: `${Math.abs(pct)}%`,
                          background: r.mean_lift >= 0 ? 'var(--gain)' : 'var(--regress)',
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

      <h2 style={{ marginTop: 28 }}>Moments worth rewatching</h2>
      <p className="small muted" style={{ marginTop: -4, marginBottom: 14 }}>
        The single exchanges across this case with the largest next-turn shift.
      </p>
      {data.best_moments.map((m, i) => (
        <div className="panel" key={i}>
          <div className="eyebrow">
            Session {m.session_number} · {m.intervention} · lift +{m.lift}
          </div>
          <blockquote>
            <span className="who">Therapist</span>
            {m.therapist_text}
          </blockquote>
          <blockquote className="gain" style={{ marginLeft: 20 }}>
            <span className="who">Client, next turn</span>
            {m.client_next_text}
          </blockquote>
        </div>
      ))}
      {!data.best_moments.length && (
        <div className="empty">
          <h3>Nothing to show yet</h3>
          <p className="small">Add a few more sessions and the strongest exchanges will surface here.</p>
        </div>
      )}
    </>
  )
}
