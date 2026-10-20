import { useCallback, useEffect, useState } from 'react'

import { api, DIMENSION_LABELS, GOAL_STATUS } from '../lib/api'

/**
 * Session goals: the therapist names one dimension to work on in a given
 * session; once that session is scored, the dimension is compared with the
 * previous session on the client's own scale.
 */
export default function GoalTracker({ clientId, nextSession, refreshKey }) {
  const [goals, setGoals] = useState([])
  const [form, setForm] = useState({ session_number: '', dimension: 'future_orientation', note: '' })
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)

  const load = useCallback(
    () => api.goals(clientId).then((r) => setGoals(r.goals)).catch((e) => setError(e.message)),
    [clientId]
  )
  useEffect(() => { load() }, [load, refreshKey])

  const submit = async (e) => {
    e.preventDefault()
    setBusy(true); setError(null)
    try {
      const r = await api.addGoal(clientId, {
        session_number: Number(form.session_number || nextSession),
        dimension: form.dimension,
        note: form.note || undefined,
      })
      setGoals(r.goals)
      setForm({ ...form, session_number: '', note: '' })
    } catch (e) { setError(e.message) } finally { setBusy(false) }
  }

  const remove = async (gid) => {
    try { await api.deleteGoal(gid); await load() } catch (e) { setError(e.message) }
  }

  return (
    <form className="panel" onSubmit={submit}>
      <h2>Session goals</h2>
      <p className="small muted" style={{ marginTop: -6 }}>
        Pick one dimension to work on in a session. After that session is
        scored, it is compared with the session before, on this client's own
        scale (a change under 0.3 SD counts as holding steady).
      </p>

      {error && <div className="notice">{error}</div>}

      {goals.length > 0 && (
        <ul className="goal-list">
          {goals.map((g) => {
            const st = GOAL_STATUS[g.evaluation?.status] || { text: g.evaluation?.status, cls: '' }
            return (
              <li key={g.id}>
                <div>
                  <span className="mono small">S{g.session_number}</span>{' '}
                  <strong>{DIMENSION_LABELS[g.dimension] || g.label}</strong>
                  {g.note && <span className="small muted"> — {g.note}</span>}
                  {g.evaluation?.detail && g.evaluation.status !== 'pending' && (
                    <div className="small muted">{g.evaluation.detail}</div>
                  )}
                </div>
                <div style={{ whiteSpace: 'nowrap' }}>
                  <span className={`chip ${st.cls}`}>{st.text}</span>{' '}
                  <button type="button" className="linkish" onClick={() => remove(g.id)}>remove</button>
                </div>
              </li>
            )
          })}
        </ul>
      )}

      <div className="row" style={{ flexWrap: 'wrap' }}>
        <div className="field" style={{ width: 110, marginBottom: 0 }}>
          <label htmlFor="g-s">Session</label>
          <input id="g-s" type="number" min="1" placeholder={String(nextSession)}
                 value={form.session_number}
                 onChange={(e) => setForm({ ...form, session_number: e.target.value })} />
        </div>
        <div className="field" style={{ flex: 1, minWidth: 180, marginBottom: 0 }}>
          <label htmlFor="g-d">Work on</label>
          <select id="g-d" value={form.dimension}
                  onChange={(e) => setForm({ ...form, dimension: e.target.value })}>
            {Object.entries(DIMENSION_LABELS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
          </select>
        </div>
        <div className="field" style={{ flex: 2, minWidth: 180, marginBottom: 0 }}>
          <label htmlFor="g-n">Note (optional)</label>
          <input id="g-n" maxLength={240} value={form.note} placeholder="e.g. plans for the weekend"
                 onChange={(e) => setForm({ ...form, note: e.target.value })} />
        </div>
        <button type="submit" disabled={busy}>Set goal</button>
      </div>
    </form>
  )
}
