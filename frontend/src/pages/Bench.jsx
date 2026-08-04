import { useState } from 'react'

import { api, DIMENSION_LABELS } from '../lib/api'

const SAMPLE = `THERAPIST: How has the week been?
CLIENT: Bad. He always does this and there's nothing I can do about it. I had to just sit there and take it, like always.
THERAPIST: It sounds like part of you wanted to say something and part of you needed it to stay quiet.
CLIENT: Yeah. I think I shut down because I was scared he'd escalate. I felt resentful, and also a bit relieved that it ended quickly. That's odd to say out loud.
THERAPIST: What would you want to say, if the fear wasn't in the way?
CLIENT: Maybe that I need ten minutes before we talk. I've decided I'm going to try it on Tuesday.`

export default function Bench() {
  const [text, setText] = useState(SAMPLE)
  const [out, setOut] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  const run = async () => {
    setBusy(true); setError(null)
    try { setOut(await api.analyze(text)) }
    catch (e) { setError(e.message) }
    finally { setBusy(false) }
  }

  return (
    <>
      <div className="eyebrow">Scratch</div>
      <h1>Bench</h1>
      <p className="lede">
        Score a transcript without attaching it to a case. Nothing here is saved.
        Useful for checking how the extractor reads a passage before you commit
        a whole case to it.
      </p>

      {error && <div className="notice">{error}</div>}

      <div className="panel">
        <div className="field">
          <label htmlFor="bench">Transcript</label>
          <textarea id="bench" rows={12} value={text} onChange={(e) => setText(e.target.value)} />
        </div>
        <button onClick={run} disabled={busy}>{busy ? 'Scoring…' : 'Score it'}</button>
      </div>

      {out && (
        <>
          <div className="panel">
            <h2>Session features</h2>
            <table>
              <thead>
                <tr><th>Dimension</th><th className="num">Score</th><th style={{ width: '50%' }}></th></tr>
              </thead>
              <tbody>
                {Object.keys(DIMENSION_LABELS).map((k) => (
                  <tr key={k}>
                    <td>{DIMENSION_LABELS[k]}</td>
                    <td className="num">{out.features[k]?.toFixed(3)}</td>
                    <td>
                      <div className="meter">
                        <i style={{ left: 0, width: `${out.features[k] * 100}%`, background: 'var(--gain)' }} />
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="small muted" style={{ marginTop: 12, marginBottom: 0 }}>
              Auxiliary — hopelessness {out.features.hopelessness?.toFixed(3)},
              rumination {out.features.rumination?.toFixed(3)},
              absolutism {out.features.absolutism?.toFixed(3)},
              solution focus {out.features.solution_focus?.toFixed(3)}.
              With no prior sessions there is no personal baseline, so the TPI
              here sits at 50 by definition.
            </p>
          </div>

          <div className="panel">
            <h2>Therapist turn pairs</h2>
            {out.therapist_impact.length ? (
              <table>
                <thead>
                  <tr><th>Intervention</th><th className="num">n</th><th className="num">Mean lift</th></tr>
                </thead>
                <tbody>
                  {out.therapist_impact.map((r) => (
                    <tr key={r.intervention}>
                      <td>{r.label}</td>
                      <td className="num">{r.n}</td>
                      <td className="num" style={{ color: r.mean_lift >= 0 ? 'var(--gain)' : 'var(--regress)' }}>
                        {r.mean_lift > 0 ? '+' : ''}{r.mean_lift.toFixed(3)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="small muted">No turn pairs long enough to measure.</p>
            )}
          </div>
        </>
      )}
    </>
  )
}
