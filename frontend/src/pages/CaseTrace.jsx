import { useCallback, useEffect, useRef, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'

import DimensionGrid from '../components/DimensionGrid'
import Trace from '../components/Trace'
import { api, DIMENSION_LABELS, STATE_COPY, stateColor } from '../lib/api'

export default function CaseTrace() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [selected, setSelected] = useState(null)
  const [transcript, setTranscript] = useState('')
  const [busy, setBusy] = useState(false)
  const [measure, setMeasure] = useState({ session_number: '', instrument: 'PHQ-9', score: '' })
  const fileRef = useRef(null)

  const load = useCallback(
    () => api.trajectory(id).then(setData).catch((e) => setError(e.message)),
    [id]
  )

  useEffect(() => { load() }, [load])

  const addSession = async (e) => {
    e.preventDefault()
    if (transcript.trim().length < 20) {
      setError('That transcript is too short to score. Paste a full session.')
      return
    }
    setBusy(true); setError(null)
    try {
      await api.addSession(id, { transcript })
      setTranscript('')
      await load()
    } catch (e) { setError(e.message) } finally { setBusy(false) }
  }

  const uploadFile = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    setBusy(true); setError(null)
    try { await api.uploadSession(id, file); await load() }
    catch (e) { setError(e.message) }
    finally { setBusy(false); if (fileRef.current) fileRef.current.value = '' }
  }

  const addMeasure = async (e) => {
    e.preventDefault()
    setBusy(true); setError(null)
    try {
      await api.addMeasure(id, {
        session_number: Number(measure.session_number),
        instrument: measure.instrument,
        score: Number(measure.score),
      })
      setMeasure({ ...measure, session_number: '', score: '' })
      await load()
    } catch (e) { setError(e.message) } finally { setBusy(false) }
  }

  if (error && !data) return <div className="notice">{error}</div>
  if (!data) return <p className="spinner">Loading case…</p>

  const { client, sessions, tpi_series, dimension_series, trajectory, baseline, measures, validation } = data
  const momentum = trajectory?.momentum
  const cp = trajectory?.change_point
  const sel = selected != null ? sessions[selected] : sessions[sessions.length - 1]

  return (
    <>
      <div className="eyebrow">
        <Link to="/">Cases</Link> / {client.code}
      </div>
      <h1>{client.code}</h1>
      <p className="lede">
        {client.presenting_issue || 'No presenting issue recorded'}
        {client.modality ? ` · ${client.modality}` : ''}
        {client.therapist_code ? ` · therapist ${client.therapist_code}` : ''}
      </p>

      {error && <div className="notice">{error}</div>}

      {/* ---------------- the trace ---------------- */}
      <div className="panel">
        <div
          className="topline"
          style={{ background: stateColor(momentum?.state) }}
        />
        <div className="panel-head">
          <div>
            <h2>The trace</h2>
            <p className="small muted" style={{ margin: 0 }}>
              Inside the band is this person talking the way they usually talk.
              Above it, their language has moved past their own norm.
            </p>
          </div>
          {momentum && (
            <span className={`state-tag state-${momentum.state}`}>
              {STATE_COPY[momentum.state] || momentum.state}
            </span>
          )}
        </div>

        <Trace
          series={tpi_series}
          changePoint={cp}
          measures={measures}
          selected={selected}
          onSelect={(i) => setSelected(i)}
        />

        <div className="readout" style={{ marginTop: 18 }}>
          <div className="readout-item">
            <div className="k">Latest TPI</div>
            <div className="v">{tpi_series.length ? tpi_series[tpi_series.length - 1].toFixed(1) : '—'}</div>
          </div>
          <div className="readout-item">
            <div className="k">Since session 1</div>
            <div className="v" style={{ color: trajectory.net_change >= 0 ? 'var(--gain)' : 'var(--regress)' }}>
              {trajectory.net_change > 0 ? '+' : ''}{trajectory.net_change}
            </div>
          </div>
          <div className="readout-item">
            <div className="k">Recent slope</div>
            <div className="v">{trajectory.recent_trend?.slope_per_session ?? '—'}</div>
          </div>
          <div className="readout-item">
            <div className="k">Sessions</div>
            <div className="v">{trajectory.n_sessions}</div>
          </div>
          <div className="readout-item">
            <div className="k">Volatility (SD)</div>
            <div className="v">{momentum?.volatility ?? '—'}</div>
          </div>
        </div>

        {momentum && (
          <p className="small" style={{ marginTop: 12, marginBottom: 0 }}>
            {momentum.detail}{' '}
            {cp && (
              <>
                A regime change was detected at session {cp.index + 1}: the mean
                moved from {cp.mean_before} to {cp.mean_after}. Worth asking what
                happened around then.
              </>
            )}
          </p>
        )}
      </div>

      {/* ---------------- dimensions ---------------- */}
      <h2 style={{ marginTop: 30 }}>What moved</h2>
      <p className="small muted" style={{ marginTop: -4, marginBottom: 14 }}>
        Dotted line is this client's own baseline on each dimension.
      </p>
      <DimensionGrid dimensionSeries={dimension_series} baseline={baseline} />

      {/* ---------------- selected session ---------------- */}
      {sel && (
        <div className="panel" style={{ marginTop: 24 }}>
          <div className="panel-head">
            <h2>Session {sel.session_number} — contribution breakdown</h2>
            <Link className="btn ghost" to={`/session/${sel.id}`}>Open session</Link>
          </div>
          <p className="small muted" style={{ marginTop: -6 }}>
            How each dimension pushed this session's TPI away from 50. Bars right
            of centre pulled the score up.
          </p>
          <table>
            <thead>
              <tr>
                <th>Dimension</th>
                <th className="num">Raw</th>
                <th className="num">z</th>
                <th style={{ width: '45%' }}>Contribution to TPI</th>
              </tr>
            </thead>
            <tbody>
              {Object.keys(DIMENSION_LABELS).map((k) => {
                const contrib = sel.contributions?.[k] ?? 0
                const scale = 6
                const pct = Math.max(-50, Math.min(50, (contrib / scale) * 50))
                return (
                  <tr key={k}>
                    <td>{DIMENSION_LABELS[k]}</td>
                    <td className="num">{sel.features?.[k]?.toFixed(3)}</td>
                    <td className="num">{sel.z_scores?.[k]?.toFixed(2)}</td>
                    <td>
                      <div className="divergent">
                        <span className="zero" />
                        <i
                          style={{
                            left: contrib >= 0 ? '50%' : `${50 + pct}%`,
                            width: `${Math.abs(pct)}%`,
                            background: contrib >= 0 ? 'var(--gain)' : 'var(--regress)',
                          }}
                        />
                      </div>
                      <span className="mono small">{contrib > 0 ? '+' : ''}{contrib}</span>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* ---------------- validation ---------------- */}
      {validation && (
        <div className="panel">
          <h2>Against self-report</h2>
          <p className="small muted" style={{ marginTop: -6 }}>
            The index should move opposite to a symptom scale. A negative
            correlation is the result you want here.
          </p>
          <div className="readout">
            <div className="readout-item">
              <div className="k">{validation.instrument} · Pearson r</div>
              <div className="v" style={{ color: validation.pearson_r < 0 ? 'var(--gain)' : 'var(--regress)' }}>
                {validation.pearson_r}
              </div>
            </div>
            <div className="readout-item">
              <div className="k">Spearman ρ</div>
              <div className="v">{validation.spearman_rho}</div>
            </div>
            <div className="readout-item">
              <div className="k">Paired sessions</div>
              <div className="v">{validation.n_paired_sessions}</div>
            </div>
          </div>
          {validation.language_leads_symptoms_r != null && (
            <p className="small" style={{ marginTop: 12, marginBottom: 0 }}>
              Lead-lag: language change predicting next-session symptom change,
              r = {validation.language_leads_symptoms_r}. The reverse direction,
              r = {validation.symptoms_lead_language_r}. If the first is larger,
              the transcript is carrying information the questionnaire has not
              caught up with yet.
            </p>
          )}
        </div>
      )}

      {/* ---------------- sessions ---------------- */}
      <div className="panel flush" style={{ marginTop: 24 }}>
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th className="num">TPI</th>
              <th className="num">Δ</th>
              <th className="num">Confidence</th>
              <th className="num">Client words</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {sessions.map((s, i) => {
              const delta = i === 0 ? null : s.tpi - sessions[i - 1].tpi
              return (
                <tr key={s.id} className="clickable" onClick={() => navigate(`/session/${s.id}`)}>
                  <td className="mono">{s.session_number}</td>
                  <td className="num"><strong>{s.tpi.toFixed(1)}</strong></td>
                  <td className="num" style={{ color: delta > 0 ? 'var(--gain)' : delta < 0 ? 'var(--regress)' : 'var(--ink-3)' }}>
                    {delta == null ? '—' : `${delta > 0 ? '+' : ''}${delta.toFixed(1)}`}
                  </td>
                  <td className="num">{(s.confidence * 100).toFixed(0)}%</td>
                  <td className="num">{s.features?.n_client_words ?? '—'}</td>
                  <td className="small muted">open →</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      <div className="grid-2">
        <form className="panel" onSubmit={addSession}>
          <h2>Add a session</h2>
          <p className="small muted" style={{ marginTop: -6 }}>
            Paste a speaker-labelled transcript. THERAPIST: / CLIENT: works, so
            do T:, C:, Counselor:, Patient: and diarised SPEAKER_00 labels.
          </p>
          <div className="field">
            <label htmlFor="tr">Transcript</label>
            <textarea
              id="tr" rows={9} value={transcript}
              placeholder={'THERAPIST: How has the week been?\nCLIENT: ...'}
              onChange={(e) => setTranscript(e.target.value)}
            />
          </div>
          <div className="row">
            <button type="submit" disabled={busy}>{busy ? 'Scoring…' : 'Score session'}</button>
            <label htmlFor="file" className="btn ghost" style={{ cursor: 'pointer', textTransform: 'none', letterSpacing: 0, fontSize: 13 }}>
              Upload a .txt file
            </label>
            <input id="file" ref={fileRef} type="file" accept=".txt,.md,.csv"
                   onChange={uploadFile} className="sr-only" />
          </div>
        </form>

        <form className="panel" onSubmit={addMeasure}>
          <h2>Record a self-report score</h2>
          <p className="small muted" style={{ marginTop: -6 }}>
            Adding PHQ-9, GAD-7 or MADRS scores turns the trace into something
            testable rather than something to look at.
          </p>
          <div className="field">
            <label htmlFor="ms">Session number</label>
            <input id="ms" type="number" min="1" required value={measure.session_number}
                   onChange={(e) => setMeasure({ ...measure, session_number: e.target.value })} />
          </div>
          <div className="field">
            <label htmlFor="mi">Instrument</label>
            <select id="mi" value={measure.instrument}
                    onChange={(e) => setMeasure({ ...measure, instrument: e.target.value })}>
              <option>PHQ-9</option><option>GAD-7</option><option>MADRS</option><option>ORS</option>
            </select>
          </div>
          <div className="field">
            <label htmlFor="msc">Score</label>
            <input id="msc" type="number" step="0.1" required value={measure.score}
                   onChange={(e) => setMeasure({ ...measure, score: e.target.value })} />
          </div>
          <button type="submit" disabled={busy}>Save score</button>
        </form>
      </div>

      <div className="row" style={{ marginTop: 8 }}>
        <Link className="btn ghost" to={`/case/${id}/therapist`}>Therapist contribution →</Link>
      </div>
    </>
  )
}
