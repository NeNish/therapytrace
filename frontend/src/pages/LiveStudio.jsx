import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import LiveRecorder from '../components/LiveRecorder'
import { api, STATE_COPY, stateColor } from '../lib/api'

export default function LiveStudio() {
  const navigate = useNavigate()
  const [clients, setClients] = useState([])
  const [selectedClientId, setSelectedClientId] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)
  const [quickResult, setQuickResult] = useState(null)

  useEffect(() => {
    api
      .listClients()
      .then((data) => {
        setClients(data || [])
        if (data && data.length > 0) {
          setSelectedClientId(data[0].id.toString())
        }
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  const selectedClient = clients.find((c) => c.id.toString() === selectedClientId)

  // Handle completion from LiveRecorder
  const handleSessionComplete = async ({ transcript, audioPath, videoPath }) => {
    if (!transcript || transcript.trim().length < 20) {
      setError('The transcript is too short to score. Ensure at least one or two dialogue turns are recorded.')
      return
    }

    setBusy(true)
    setError(null)

    try {
      if (selectedClientId === 'sandbox') {
        // Run stateless analysis
        const res = await api.multimodal({
          transcript,
          audio_path: audioPath || undefined,
          video_path: videoPath || undefined,
          max_seconds: 120,
        })
        setQuickResult(res)
      } else {
        // Save to client
        const sess = await api.addSession(Number(selectedClientId), {
          transcript,
          audio_path: audioPath || undefined,
          video_path: videoPath || undefined,
        })
        navigate(`/session/${sess.id}`)
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="live-studio-page">
      <div className="eyebrow">Recording Studio</div>
      <h1>Live In-Session Capture</h1>
      <p className="lede">
        Record audio, video, and spoken dialogue in real time directly from your browser.
        Speech-to-text automatically formats turns with speaker roles, and high-resolution
        audio is captured for acoustic prosody scoring.
      </p>

      {error && <div className="notice">{error}</div>}

      <div className="panel" style={{ marginBottom: 20 }}>
        <div className="row" style={{ justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ flex: 1, maxWidth: 360 }}>
            <label htmlFor="client-select">TARGET CASE FOR THIS SESSION</label>
            <select
              id="client-select"
              value={selectedClientId}
              onChange={(e) => {
                setSelectedClientId(e.target.value)
                setQuickResult(null)
              }}
              disabled={loading || busy}
            >
              <option value="sandbox">⚡ Sandbox / Practice Mode (Stateless, no DB save)</option>
              {clients.map((c) => (
                <option key={c.id} value={c.id.toString()}>
                  {c.code} {c.presenting_issue ? `— ${c.presenting_issue}` : ''} ({c.n_sessions || 0} sessions)
                </option>
              ))}
            </select>
          </div>

          {selectedClient && selectedClientId !== 'sandbox' && (
            <div className="row" style={{ gap: 20, alignItems: 'center' }}>
              <div className="readout-item">
                <div className="k">Latest TPI</div>
                <div className="v" style={{ fontSize: 22 }}>
                  {selectedClient.latest_tpi != null ? selectedClient.latest_tpi.toFixed(1) : '—'}
                </div>
              </div>
              <div className="readout-item">
                <div className="k">Momentum</div>
                <div className="v">
                  <span
                    className={`state-tag state-${selectedClient.momentum || 'steady'}`}
                    style={{ fontSize: 11 }}
                  >
                    {STATE_COPY[selectedClient.momentum]?.tag || 'Steady'}
                  </span>
                </div>
              </div>
              <div className="readout-item">
                <div className="k">Case Link</div>
                <Link to={`/case/${selectedClient.id}`} className="small mono">
                  Open {selectedClient.code} &rarr;
                </Link>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Main Studio Live Recorder */}
      <LiveRecorder
        onComplete={handleSessionComplete}
        standalone={true}
        defaultMode="dictation"
      />

      {busy && (
        <div className="panel" style={{ marginTop: 20, textAlign: 'center' }}>
          <p className="spinner">Processing recording and scoring linguistic process dimensions…</p>
        </div>
      )}

      {/* Sandbox Instant Result */}
      {quickResult && (
        <div className="panel" style={{ marginTop: 24 }}>
          <div className="topline" style={{ background: 'var(--gain)' }} />
          <div className="panel-head">
            <h2>Sandbox Analysis Complete</h2>
            <span className="state-tag state-gaining">Instant Readout</span>
          </div>
          <p className="small">
            Calculated across {quickResult.n_turns || 0} turns ({quickResult.client_turns || 0} client utterances).
          </p>

          {quickResult.insights?.text?.dimensions && (
            <table>
              <thead>
                <tr>
                  <th>Dimension</th>
                  <th className="num">Score</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(quickResult.insights.text.dimensions).map(([k, v]) => (
                  <tr key={k}>
                    <td>{k.replace(/_/g, ' ')}</td>
                    <td className="num">{(v * 100).toFixed(0)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          {quickResult.insights?.text?.summary && (
            <p style={{ marginTop: 14, fontSize: 15 }}>
              {quickResult.insights.text.summary}
            </p>
          )}
        </div>
      )}
    </div>
  )
}
