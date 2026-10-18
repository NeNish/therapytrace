import { useCallback, useEffect, useRef, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'

import ChangeTalk from '../components/ChangeTalk'
import Resolvability from '../components/Resolvability'
import DimensionGrid from '../components/DimensionGrid'
import LiveRecorder from '../components/LiveRecorder'
import Trace from '../components/Trace'
import { api, DIMENSION_LABELS, STATE_COPY, stateColor } from '../lib/api'

/**
 * A labelled rule between stretches of the page.
 *
 * The case view runs to about a dozen panels. Rendered as one flat stack they
 * all carry the same weight, so a reader has no way to tell the trace itself
 * from the form that adds a session, and the page reads as a dump rather than
 * a document. These break it into named stretches that can be scanned.
 */
function SectionRule({ label, hint }) {
  return (
    <div className="section-rule">
      <span className="section-rule-label">{label}</span>
      {hint && <span className="section-rule-hint">{hint}</span>}
    </div>
  )
}

export default function CaseTrace() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [selected, setSelected] = useState(null)
  const [transcript, setTranscript] = useState('')
  const [busy, setBusy] = useState(false)
  const [measure, setMeasure] = useState({ session_number: '', instrument: 'PHQ-9', score: '' })
  const [audioPath, setAudioPath] = useState('')
  const [videoPath, setVideoPath] = useState('')
  const [audioInfo, setAudioInfo] = useState(null)
  const [videoInfo, setVideoInfo] = useState(null)
  const [uploading, setUploading] = useState(null)
  const [sessionInputTab, setSessionInputTab] = useState('upload') // 'upload' | 'record'
  const fileRef = useRef(null)

  const load = useCallback(
    () => api.trajectory(id).then(setData).catch((e) => setError(e.message)),
    [id]
  )

  useEffect(() => { load() }, [load])

  const uploadMedia = async (file, kind) => {
    if (!file) return
    setUploading(kind); setError(null)
    try {
      const info = await api.uploadMedia(file)
      if (kind === 'video') { setVideoPath(info.path); setVideoInfo(info) }
      else { setAudioPath(info.path); setAudioInfo(info) }
    } catch (e) { setError(e.message) } finally { setUploading(null) }
  }

  const addSession = async (e) => {
    e.preventDefault()
    if (transcript.trim().length < 20) {
      // A video or audio file attached here is NOT ignored — it is fused on top
      // of the language score. But the trajectory index is built from language,
      // so a session row cannot exist without a transcript. Say that, rather
      // than letting the attachment look like it was thrown away.
      setError(
        (videoPath || audioPath)
          ? 'The recording was uploaded, but a session still needs its transcript: '
            + 'the progress index is computed from language, and the recording is '
            + 'fused on top of it. Paste the transcript and both are used. '
            + 'To analyse the recording on its own, use the Multimodal page.'
          : 'That transcript is too short to score. Paste a full session.'
      )
      return
    }
    setBusy(true); setError(null)
    try {
      await api.addSession(id, {
        transcript,
        audio_path: audioPath || undefined,
        video_path: videoPath || undefined,
      })
      setTranscript('')
      setAudioPath('')
      setVideoPath('')
      setAudioInfo(null)
      setVideoInfo(null)
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
        {/* Written as a sentence. The old form chained the three facts with
            middle dots, which reads as metadata rather than as a description of
            a person someone is treating. */}
        {client.presenting_issue || 'No presenting issue recorded'}
        {client.modality ? `, seen in ${client.modality}` : ''}
        {client.therapist_code ? ` with ${client.therapist_code}` : ''}
      </p>

      {error && <div className="notice">{error}</div>}

      {/* Summary band. The page runs to a dozen panels, and before this you had
          to scroll past two charts to learn anything at all. These five numbers
          are what someone opening a case wants first; everything below is the
          working out. Momentum carries the colour because it is the only item
          here that is a judgement rather than a measurement. */}
      {tpi_series.length > 0 && (
        <div className="case-summary">
          <div className="case-summary-item">
            <span className="k">Latest index</span>
            <span className="v">{tpi_series[tpi_series.length - 1].toFixed(1)}</span>
          </div>
          <div className="case-summary-item">
            <span className="k">Since session 1</span>
            <span
              className="v"
              style={{ color: trajectory.net_change >= 0 ? 'var(--gain)' : 'var(--regress)' }}
            >
              {trajectory.net_change > 0 ? '+' : ''}{trajectory.net_change}
            </span>
          </div>
          <div className="case-summary-item">
            <span className="k">Per session</span>
            {/* Two decimals, like every other figure in the band. The raw
                value arrives with four, which read as spurious precision on a
                slope estimated from a dozen noisy points. */}
            <span className="v">
              {trajectory.recent_trend?.slope_per_session != null
                ? trajectory.recent_trend.slope_per_session.toFixed(2)
                : '—'}
            </span>
          </div>
          <div className="case-summary-item">
            <span className="k">Sessions</span>
            <span className="v">{trajectory.n_sessions}</span>
          </div>
          <div className="case-summary-item is-state">
            <span className="k">Momentum</span>
            <span
              className="v"
              style={{
                color: momentum?.state === 'improving' ? 'var(--gain)'
                  : momentum?.state === 'regressing' ? 'var(--regress)' : 'var(--hold)',
              }}
            >
              {momentum?.state === 'improving' ? 'Gaining'
                : momentum?.state === 'regressing' ? 'Losing ground'
                : momentum?.state === 'plateauing' ? 'Plateau' : 'Mixed'}
            </span>
          </div>
        </div>
      )}

      {sessions.length > 0 && sessions[sessions.length - 1].confidence < 0.55 && (
        <div className="notice">
          Session {sessions[sessions.length - 1].session_number} has{' '}
          {(sessions[sessions.length - 1].confidence * 100).toFixed(0)}% confidence
          because it is short ({sessions[sessions.length - 1].features?.n_client_words ?? '—'} client words).
          The index needs roughly 300+ client words and 10+ turns for a reliable score.
          Add a fuller transcript or treat this session as exploratory.
        </div>
      )}

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
          forecast={trajectory?.forecast}
          rci={trajectory?.rci}
          selected={selected}
          onSelect={(i) => setSelected(i)}
        />

        {/* Second panel, same x-axis, its own scale. Renders nothing until at
            least two sessions carry a classifier estimate. */}
        <ChangeTalk
          points={data.change_talk_series || []}
          selected={selected}
          onSelect={(i) => setSelected(i)}
        />

        {/* States which reading the evidence supports, instead of leaving a
            reader to infer it from the width of an error bar. */}
        <Resolvability
          summary={data.resolvability}
          points={data.change_talk_series || []}
        />

        {/* The five headline numbers moved to the summary band at the top of
            the page; repeating them here only taught a reader to skip one of
            the two copies. What stays is the sentence that reads them. */}
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

      <SectionRule label="This session" hint="Where the latest session sits against the client's own norm" />

      {/* ---------------- Session DNA & Turning Point Detector ---------------- */}
      <div className="grid-2">
        {/* Session DNA Pentagram (Latest vs Baseline) */}
        <div className="panel">
          <div className="topline" style={{ background: 'var(--ink)' }} />
          <div className="panel-head">
            <div>
              <h2>Session DNA (5 Dimensions)</h2>
              <p className="small muted" style={{ margin: 0 }}>
                Visual footprint of the latest session (teal) against baseline norm (dashed black).
              </p>
            </div>
            <span className="state-tag" style={{ background: 'var(--panel)', border: '1px solid var(--rule)' }}>
              Latest: Session {sessions.length}
            </span>
          </div>
          {(() => {
            const dims = ['self_agency', 'future_orientation', 'emotional_granularity', 'problem_ownership', 'reflection_depth']
            const labels = ['Agency', 'Future', 'Granularity', 'Ownership', 'Reflection']
            const latestSess = sessions[sessions.length - 1]
            const zScores = latestSess?.z_scores || {}

            const size = 220
            const center = size / 2
            const radius = 70

            // Helper to get polar coordinates for a pentagon
            const getPoint = (valZ, index) => {
              // Clamp z between -2.5 and +2.5, map to fraction 0.1 to 1.0
              const clamped = Math.max(-2.5, Math.min(2.5, valZ || 0))
              const r = radius * (0.55 + (clamped / 2.5) * 0.45)
              const angle = (index * 2 * Math.PI) / 5 - Math.PI / 2
              return {
                x: center + r * Math.cos(angle),
                y: center + r * Math.sin(angle),
              }
            }

            const getBasePoint = (index) => {
              const r = radius * 0.55 // z = 0 baseline
              const angle = (index * 2 * Math.PI) / 5 - Math.PI / 2
              return {
                x: center + r * Math.cos(angle),
                y: center + r * Math.sin(angle),
              }
            }

            const currentPts = dims.map((d, i) => getPoint(zScores[d] ?? 0, i))
            const currentPath = currentPts.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(' ') + ' Z'

            const basePts = dims.map((_, i) => getBasePoint(i))
            const basePath = basePts.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(' ') + ' Z'

            return (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', marginTop: 10 }}>
                <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
                  {/* Background concentric rings */}
                  {[0.25, 0.55, 0.85, 1.0].map((frac, idx) => (
                    <polygon
                      key={idx}
                      points={dims.map((_, i) => {
                        const a = (i * 2 * Math.PI) / 5 - Math.PI / 2
                        const rx = center + radius * frac * Math.cos(a)
                        const ry = center + radius * frac * Math.sin(a)
                        return `${rx.toFixed(1)},${ry.toFixed(1)}`
                      }).join(' ')}
                      fill="none"
                      stroke="var(--rule)"
                      strokeWidth={idx === 1 ? '1.5' : '0.8'}
                      strokeDasharray={idx === 1 ? '3 3' : 'none'}
                    />
                  ))}

                  {/* Spokes */}
                  {dims.map((_, i) => {
                    const a = (i * 2 * Math.PI) / 5 - Math.PI / 2
                    return (
                      <line
                        key={i}
                        x1={center} y1={center}
                        x2={center + radius * Math.cos(a)}
                        y2={center + radius * Math.sin(a)}
                        stroke="var(--rule)"
                        strokeWidth="1"
                      />
                    )
                  })}

                  {/* Baseline Shape */}
                  <path d={basePath} fill="none" stroke="var(--ink-3)" strokeWidth="1.2" strokeDasharray="3 3" />

                  {/* Client Current Shape */}
                  <path d={currentPath} fill="var(--gain-wash)" stroke="var(--gain)" strokeWidth="2" opacity="0.85" />

                  {/* Vertices & Labels */}
                  {dims.map((d, i) => {
                    const p = currentPts[i]
                    const a = (i * 2 * Math.PI) / 5 - Math.PI / 2
                    const lx = center + (radius + 20) * Math.cos(a)
                    const ly = center + (radius + 20) * Math.sin(a) + 3
                    const z = zScores[d] ?? 0
                    return (
                      <g key={d}>
                        <circle cx={p.x} cy={p.y} r="3.5" fill="var(--panel)" stroke="var(--gain)" strokeWidth="2" />
                        <text
                          x={lx} y={ly}
                          textAnchor="middle"
                          fontFamily="var(--data)"
                          fontSize="9"
                          fill="var(--ink)"
                          fontWeight="600"
                        >
                          {labels[i]} ({z >= 0 ? '+' : ''}{z.toFixed(1)})
                        </text>
                      </g>
                    )
                  })}
                </svg>

                <div className="small muted" style={{ textAlign: 'center', marginTop: 8 }}>
                  Solid green is this session's contour; dashed circle is the client's own baseline.
                </div>
              </div>
            )
          })()}
        </div>

        {/* Turning Point Detector */}
        <div className="panel">
          <div className="topline" style={{ background: 'var(--mark)' }} />
          <div className="panel-head">
            <div>
              <h2>Turning Point Detector</h2>
              <p className="small muted" style={{ margin: 0 }}>
                Single session with the largest upward momentum jump across the full timeline.
              </p>
            </div>
            <span className="state-tag" style={{ background: 'var(--mark-wash)', color: 'var(--mark)' }}>
              Milestone Shift
            </span>
          </div>

          {(() => {
            if (sessions.length < 2) {
              return <p className="small muted" style={{ marginTop: 14 }}>Requires at least 2 sessions to detect velocity shifts.</p>
            }

            let maxDelta = -Infinity
            let turningIdx = -1
            for (let i = 1; i < sessions.length; i++) {
              const delta = sessions[i].tpi - sessions[i - 1].tpi
              if (delta > maxDelta) {
                maxDelta = delta
                turningIdx = i
              }
            }

            const turningSess = sessions[turningIdx]
            const prevSess = sessions[turningIdx - 1]

            return (
              <div style={{ marginTop: 12 }}>
                <div className="readout" style={{ gap: 16 }}>
                  <div className="readout-item">
                    <div className="k">Turning Point</div>
                    <div className="v" style={{ fontSize: 20 }}>Session {turningSess.session_number}</div>
                  </div>
                  <div className="readout-item">
                    <div className="k">Max Single Jump</div>
                    <div className="v" style={{ fontSize: 20, color: maxDelta > 0 ? 'var(--gain)' : 'var(--regress)' }}>
                      {maxDelta > 0 ? '+' : ''}{maxDelta.toFixed(1)} pts
                    </div>
                  </div>
                  <div className="readout-item">
                    <div className="k">Shift Range</div>
                    <div className="v" style={{ fontSize: 20 }}>
                      {prevSess.tpi.toFixed(1)} &rarr; {turningSess.tpi.toFixed(1)}
                    </div>
                  </div>
                </div>

                <div style={{ marginTop: 14, padding: '10px 14px', background: 'var(--panel)', border: '1px solid var(--rule)' }}>
                  <div className="eyebrow" style={{ color: 'var(--mark)', marginBottom: 4 }}>
                    Leading Factor in Session {turningSess.session_number}
                  </div>
                  <p className="small" style={{ margin: 0, lineHeight: 1.5 }}>
                    {(() => {
                      const zEntries = Object.entries(turningSess.z_scores || {})
                      zEntries.sort((a, b) => b[1] - a[1])
                      if (zEntries.length > 0) {
                        const [bestDim, bestZ] = zEntries[0]
                        return `The jump was led by a surge in ${bestDim.replace(/_/g, ' ')} (+${bestZ.toFixed(2)} SD), signaling an inflection in client autonomy and psychological movement.`
                      }
                      return `Score elevated significantly from Session ${prevSess.session_number} to Session ${turningSess.session_number}.`
                    })()}
                  </p>
                </div>

                <div className="row" style={{ marginTop: 12 }}>
                  <Link className="btn ghost small" to={`/session/${turningSess.id}`}>
                    Examine Turning Session {turningSess.session_number} &rarr;
                  </Link>
                </div>
              </div>
            )
          })()}
        </div>
      </div>

      {/* Trajectory Forecast Details */}
      {trajectory?.forecast?.available && trajectory.forecast.projections?.length > 0 && (
        <div className="panel">
          <div className="topline" style={{ background: 'var(--gain)' }} />
          <div className="panel-head">
            {/* Named for what it is. The projection is a damped linear trend,
                not a Bayesian model: there is no prior and no posterior, so
                calling its bands "credible intervals" would invite a question
                the method cannot answer. They are projection bands that widen
                with distance, and the caveat below says what they are worth. */}
            <h2>Projected trajectory (next 3 sessions)</h2>
            <span className="state-tag state-gaining">Forward Outlook</span>
          </div>
          <p className="small" style={{ marginTop: -4 }}>
            {trajectory.forecast.outlook} The recent slope is extrapolated and
            damped toward zero as it runs forward, so the projection flattens
            rather than racing away. Bands widen with distance.
          </p>
          <table>
            <thead>
              <tr>
                <th>Projected Session</th>
                <th className="num">Expected TPI</th>
                <th className="num">Expected Δ vs Today</th>
                <th className="num">80% band</th>
                <th className="num">95% band</th>
              </tr>
            </thead>
            <tbody>
              {trajectory.forecast.projections.map((p) => (
                <tr key={p.session_number}>
                  <td className="mono">Session {p.session_number} (projected)</td>
                  <td className="num"><strong>{p.projected_tpi.toFixed(1)}</strong></td>
                  <td className="num" style={{ color: p.expected_change >= 0 ? 'var(--gain)' : 'var(--regress)' }}>
                    {p.expected_change > 0 ? '+' : ''}{p.expected_change.toFixed(1)}
                  </td>
                  <td className="num mono small muted">[{p.ci_80[0]} – {p.ci_80[1]}]</td>
                  <td className="num mono small muted">[{p.ci_95[0]} – {p.ci_95[1]}]</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* ---------------- dimensions ---------------- */}
      <SectionRule label="Dimension by dimension" hint="Each process dimension against this client's baseline" />

      <h2 className="section-sub">What moved</h2>
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
        <>
        <SectionRule label="Evidence" hint="How the index lines up with the questionnaire, session by session" />
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
        </>
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
                  <td className="num" title={
                    s.confidence < 0.55
                      ? 'Low confidence — short session or fewer than 300 client words'
                      : ''
                  } style={{ color: s.confidence < 0.55 ? 'var(--regress)' : undefined }}>
                    {(s.confidence * 100).toFixed(0)}%
                    {s.confidence < 0.55 && ' ⚠'}
                  </td>
                  <td className="num">{s.features?.n_client_words ?? '—'}</td>
                  <td className="small muted">open →</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      <SectionRule label="Add data" hint="Score a new session, or record a questionnaire result" />

      <div className="grid-2">
        <div className="panel">
          <div className="panel-head" style={{ marginBottom: 10 }}>
            <h2>Add a session</h2>
            <div className="mode-toggle-group">
              <button
                type="button"
                className={`mode-btn ${sessionInputTab === 'upload' ? 'active' : ''}`}
                onClick={() => setSessionInputTab('upload')}
              >
                📝 Paste / File
              </button>
              <button
                type="button"
                className={`mode-btn ${sessionInputTab === 'record' ? 'active' : ''}`}
                onClick={() => setSessionInputTab('record')}
              >
                🎙️ Record Live
              </button>
            </div>
          </div>

          {sessionInputTab === 'record' ? (
            <div>
              <p className="small muted" style={{ marginTop: -4, marginBottom: 12 }}>
                Record live audio, video, and spoken dialogue. Spoken words will automatically transcribe into speaker-labelled turns.
              </p>
              <LiveRecorder
                defaultMode="dictation"
                initialTranscript={transcript}
                onComplete={({ transcript: recTr, audioPath: aP, videoPath: vP, mediaInfo }) => {
                  if (recTr) setTranscript(recTr)
                  if (aP) { setAudioPath(aP); setAudioInfo(mediaInfo) }
                  if (vP) { setVideoPath(vP); setVideoInfo(mediaInfo) }
                  setSessionInputTab('upload')
                }}
              />
            </div>
          ) : (
            <form onSubmit={addSession}>
              <p className="small muted" style={{ marginTop: -4 }}>
                Paste a speaker-labelled transcript. Optionally attach audio and/or
                video from the same session for posture and vocal insights.
                Sessions under 300 client words score with reduced confidence.
              </p>
              <div className="field">
                <label htmlFor="tr">Transcript</label>
                <textarea
                  id="tr" rows={7} value={transcript}
                  placeholder={'THERAPIST: How has the week been?\nCLIENT: ...'}
                  onChange={(e) => setTranscript(e.target.value)}
                />
              </div>
              <div className="grid-2" style={{ gap: 10, marginBottom: 12 }}>
                <div>
                  <label htmlFor="sess-audio" className="btn ghost small"
                         style={{ cursor: 'pointer', display: 'block', textAlign: 'center' }}>
                    {uploading === 'audio' ? 'Uploading…' : audioPath ? '✓ Audio attached' : 'Attach audio (optional)'}
                  </label>
                  <input id="sess-audio" type="file" accept="audio/*,.wav,.mp3,.m4a" className="sr-only"
                         onChange={(e) => uploadMedia(e.target.files?.[0], 'audio')} />
                  {audioInfo && (
                    <p className="mono small" style={{ marginTop: 4, color: 'var(--gain)' }}>
                      {audioInfo.filename} ({audioInfo.size_mb} MB)
                    </p>
                  )}
                </div>
                <div>
                  <label htmlFor="sess-video" className="btn ghost small"
                         style={{ cursor: 'pointer', display: 'block', textAlign: 'center' }}>
                    {uploading === 'video' ? 'Uploading…' : videoPath ? '✓ Video attached' : 'Attach video (optional)'}
                  </label>
                  <input id="sess-video" type="file" accept="video/*,.mp4,.mov,.m4v" className="sr-only"
                         onChange={(e) => uploadMedia(e.target.files?.[0], 'video')} />
                  {videoInfo && (
                    <p className="mono small" style={{ marginTop: 4, color: 'var(--gain)' }}>
                      {videoInfo.filename} ({videoInfo.size_mb} MB)
                    </p>
                  )}
                </div>
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
          )}
        </div>

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
        <Link className="btn" to={`/case/${id}/insights`}>Insights & Questions →</Link>
        <Link className="btn ghost" to={`/case/${id}/compare`}>Compare Sessions (Diff) →</Link>
        <Link className="btn ghost" to={`/case/${id}/therapist`}>Therapist contribution →</Link>
      </div>
    </>
  )
}
