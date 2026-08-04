import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { api, STATE_COPY } from '../lib/api'

export default function Cases() {
  const [clients, setClients] = useState(null)
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)
  const [form, setForm] = useState({ code: '', presenting_issue: '', modality: '', therapist_code: '' })
  const [showForm, setShowForm] = useState(false)
  const navigate = useNavigate()

  const load = () =>
    api.listClients().then(setClients).catch((e) => setError(e.message))

  useEffect(() => { load() }, [])

  const seed = async () => {
    setBusy(true)
    try { await api.seedDemo(); await load() }
    catch (e) { setError(e.message) }
    finally { setBusy(false) }
  }

  const create = async (e) => {
    e.preventDefault()
    setBusy(true)
    try {
      const c = await api.createClient({
        code: form.code,
        presenting_issue: form.presenting_issue || null,
        modality: form.modality || null,
        therapist_code: form.therapist_code || null,
      })
      setShowForm(false)
      setForm({ code: '', presenting_issue: '', modality: '', therapist_code: '' })
      navigate(`/case/${c.id}`)
    } catch (e) { setError(e.message) }
    finally { setBusy(false) }
  }

  return (
    <>
      <div className="eyebrow">Caseload</div>
      <h1>Cases</h1>
      <p className="lede">
        One row per case. The index compares each session against that person's
        own earliest sessions, so two people with very different natural speaking
        styles can both be read for movement.
      </p>

      {error && <div className="notice">{error}</div>}

      <div className="row" style={{ marginBottom: 20 }}>
        <button onClick={() => setShowForm((v) => !v)} className="ghost">
          {showForm ? 'Cancel' : 'Add a case'}
        </button>
        <button onClick={seed} disabled={busy} className="ghost">
          {busy ? 'Loading…' : 'Load synthetic cases'}
        </button>
      </div>

      {showForm && (
        <form className="panel" onSubmit={create}>
          <h2>New case</h2>
          <p className="small muted" style={{ marginTop: -4 }}>
            Use a pseudonymous code. Never enter a name, a date of birth, or
            anything else that could identify the person.
          </p>
          <div className="grid-2">
            <div className="field">
              <label htmlFor="code">Case code</label>
              <input id="code" required placeholder="CL-014" value={form.code}
                     onChange={(e) => setForm({ ...form, code: e.target.value })} />
            </div>
            <div className="field">
              <label htmlFor="therapist">Therapist code</label>
              <input id="therapist" placeholder="TH-02" value={form.therapist_code}
                     onChange={(e) => setForm({ ...form, therapist_code: e.target.value })} />
            </div>
            <div className="field">
              <label htmlFor="issue">Presenting issue</label>
              <input id="issue" placeholder="Generalised anxiety" value={form.presenting_issue}
                     onChange={(e) => setForm({ ...form, presenting_issue: e.target.value })} />
            </div>
            <div className="field">
              <label htmlFor="modality">Modality</label>
              <input id="modality" placeholder="CBT" value={form.modality}
                     onChange={(e) => setForm({ ...form, modality: e.target.value })} />
            </div>
          </div>
          <button type="submit" disabled={busy}>Create case</button>
        </form>
      )}

      {clients === null && <p className="spinner">Loading caseload…</p>}

      {clients && clients.length === 0 && (
        <div className="empty">
          <h3>No cases yet</h3>
          <p className="small">
            Add a case and upload a transcript, or load the synthetic cases to
            see how a trace behaves when the true trajectory is known.
          </p>
        </div>
      )}

      {clients && clients.length > 0 && (
        <div className="panel flush">
          <table>
            <thead>
              <tr>
                <th>Case</th>
                <th>Presenting issue</th>
                <th>Modality</th>
                <th className="num">Sessions</th>
                <th className="num">Latest TPI</th>
                <th>Momentum</th>
              </tr>
            </thead>
            <tbody>
              {clients.map((c) => (
                <tr key={c.id} className="clickable" onClick={() => navigate(`/case/${c.id}`)}>
                  <td className="mono"><strong>{c.code}</strong></td>
                  <td>{c.presenting_issue || <span className="muted">—</span>}</td>
                  <td className="mono small">{c.modality || '—'}</td>
                  <td className="num">{c.n_sessions}</td>
                  <td className="num">{c.latest_tpi?.toFixed(1) ?? '—'}</td>
                  <td>
                    {c.momentum ? (
                      <span className={`state-tag state-${c.momentum}`}>
                        {STATE_COPY[c.momentum] || c.momentum}
                      </span>
                    ) : <span className="muted small">not scored</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  )
}
