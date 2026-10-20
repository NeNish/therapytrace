import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'

import { api, DIMENSION_LABELS } from '../lib/api'

/** One-page brief for the next session. Printable. */
export default function PrepCard() {
  const { id } = useParams()
  const [card, setCard] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => { api.prepCard(id).then(setCard).catch((e) => setError(e.message)) }, [id])

  if (error) return <div className="notice">{error}</div>
  if (!card) return <p className="spinner">Preparing card…</p>
  if (!card.available) {
    return (
      <>
        <div className="eyebrow"><Link to={`/case/${id}`}>Back to case</Link></div>
        <div className="empty"><h3>No card yet</h3><p>{card.reason}</p></div>
      </>
    )
  }

  return (
    <div className="prep">
      <div className="eyebrow no-print">
        <Link to="/">Cases</Link> / <Link to={`/case/${id}`}>{card.client_code}</Link> / prep card
      </div>
      <div className="panel-head">
        <h1 style={{ margin: 0 }}>Session {card.next_session} prep — {card.client_code}</h1>
        <button type="button" className="no-print" onClick={() => window.print()}>Print / save PDF</button>
      </div>
      <p className="small muted">
        Based on session {card.based_on_session} (TPI {card.latest_tpi}). Built only
        from measurements of this client's own language.
      </p>

      {card.decline_alert?.active && (
        <div className="alert-banner">
          <div>
            <strong>Downward run</strong>
            <p className="small">{card.decline_alert.message}</p>
          </div>
        </div>
      )}

      <div className="panel">
        <h3>Where to look</h3>
        {card.focus && (
          <p>
            <strong>{DIMENSION_LABELS[card.focus.dimension]}</strong> was furthest below this
            client's baseline last session ({card.focus.z > 0 ? '+' : ''}{card.focus.z} SD).
            Listen for {card.focus.look_for}.
          </p>
        )}
        {card.strength && (
          <p className="small">
            Strongest: {DIMENSION_LABELS[card.strength.dimension]} ({card.strength.z > 0 ? '+' : ''}{card.strength.z} SD).
          </p>
        )}

        {card.revisit?.length > 0 && (
          <>
            <h3>Phrases worth returning to</h3>
            {card.revisit.map((r) => (
              <blockquote key={r.turn}>"{r.text}"</blockquote>
            ))}
          </>
        )}

        <h3>What has been followed by movement with this client</h3>
        {card.what_has_worked?.length > 0 ? (
          <ul>
            {card.what_has_worked.map((m) => (
              <li key={m.intervention}>
                {m.label} — mean lift {m.mean_lift > 0 ? '+' : ''}{m.mean_lift.toFixed(3)} over {m.n} instances
              </li>
            ))}
          </ul>
        ) : (
          <p className="small muted">Not enough of this client's own data yet.</p>
        )}
        {card.suggestion && (
          <p className="small">
            {card.suggestion}{' '}
            <span className="muted">
              ({card.suggestion_source === 'corpus'
                ? 'based on 123 AnnoMI conversations, since this client has too little data of their own'
                : "based on this client's own history"})
            </span>
          </p>
        )}

        {(card.open_goals?.length > 0 || card.recent_goal_results?.length > 0) && (
          <>
            <h3>Goals</h3>
            <ul>
              {card.open_goals.map((g) => (
                <li key={g.id}>Session {g.session_number}: work on {DIMENSION_LABELS[g.dimension]}
                  {g.note ? ` — ${g.note}` : ''}</li>
              ))}
              {card.recent_goal_results.map((g) => (
                <li key={g.id}>Session {g.session_number} ({DIMENSION_LABELS[g.dimension]}): {g.evaluation.detail}</li>
              ))}
            </ul>
          </>
        )}

        {card.watch && (
          <>
            <h3>Keep an eye on</h3>
            <p className="small">{card.watch}</p>
          </>
        )}
      </div>

      <p className="footnote">{card.caveat}</p>
    </div>
  )
}
