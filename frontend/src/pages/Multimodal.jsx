import { useState } from 'react'

import { api } from '../lib/api'

const SAMPLE = `THERAPIST: How has the week been?
CLIENT: I decided not to go on Saturday, and I told her why. I felt anxious saying it and also steadier afterwards.
THERAPIST: What was that like?
CLIENT: I noticed the pattern this time. When I feel dismissed I withdraw, and then I read the silence as proof I was right.
THERAPIST: You noticed that yourself, which is not a small thing.
CLIENT: My part in it is that I wait until I'm furious and then I deliver it like an accusation. That's on me.`

function Channel({ tag, title, role, primary, children }) {
  return (
    <div
      className="panel"
      style={{
        marginBottom: 0,
        borderColor: primary ? 'var(--gain)' : 'var(--rule)',
        borderWidth: primary ? 2 : 1,
      }}
    >
      <div
        className="topline"
        style={{ background: primary ? 'var(--gain)' : 'var(--rule)' }}
      />
      <div className="eyebrow" style={{ color: primary ? 'var(--gain)' : 'var(--ink-3)' }}>
        {tag} {primary && '· REQUIRED'}
      </div>
      <h2 style={{ marginBottom: 4 }}>{title}</h2>
      <p className="small muted" style={{ marginTop: 0, minHeight: 34 }}>{role}</p>
      {children}
    </div>
  )
}

export default function Multimodal() {
  const [transcript, setTranscript] = useState(SAMPLE)
  const [audioPath, setAudioPath] = useState('')
  const [videoPath, setVideoPath] = useState('')
  const [out, setOut] = useState(null)
  const [audioInfo, setAudioInfo] = useState(null)
  const [videoInfo, setVideoInfo] = useState(null)
  const [uploading, setUploading] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  const upload = async (file, kind) => {
    if (!file) return
    setUploading(kind); setError(null)
    try {
      const info = await api.uploadMedia(file)
      if (kind === 'video') { setVideoPath(info.path); setVideoInfo(info) }
      else { setAudioPath(info.path); setAudioInfo(info) }
    } catch (e) { setError(e.message) } finally { setUploading(null) }
  }

  const run = async () => {
    setBusy(true); setError(null)
    try {
      setOut(await api.multimodal({
        transcript,
        audio_path: audioPath || undefined,
        video_path: videoPath || undefined,
        max_seconds: 30,
      }))
    } catch (e) { setError(e.message) } finally { setBusy(false) }
  }

  const ch = out?.channels || {}

  return (
    <>
      <div className="eyebrow">Three channels</div>
      <h1>Multimodal session analysis</h1>
      <p className="lede">
        Text is the primary channel and is always analysed. Audio and video are
        optional: they were tested against text alone and did not improve
        measurement accuracy, so they enter as bounded modifiers rather than
        equal votes. Each channel degrades on its own.
      </p>

      {error && <div className="notice">{error}</div>}

      <div className="grid-3">
        <Channel tag="CHANNEL 1" title="Text" primary
                 role="Transcript. Produces the score, curve, note and recommendation.">
          <textarea rows={9} value={transcript}
                    onChange={(e) => setTranscript(e.target.value)} />
        </Channel>

        <Channel tag="CHANNEL 2" title="Audio"
                 role="Pitch, pauses, vocal effort. 48 features per utterance.">
          <label htmlFor="af" className="btn ghost"
                 style={{ cursor: 'pointer', display: 'block', textAlign: 'center' }}>
            {uploading === 'audio' ? 'Uploading…' : 'Choose an audio file'}
          </label>
          <input id="af" type="file" accept="audio/*,.wav,.mp3,.m4a" className="sr-only"
                 onChange={(e) => upload(e.target.files?.[0], 'audio')} />
          {audioInfo && (
            <p className="mono small" style={{ marginTop: 8, color: 'var(--gain)' }}>
              {audioInfo.filename} · {audioInfo.size_mb} MB
            </p>
          )}
          <p className="small muted" style={{ marginTop: 8 }}>
            Optional. .wav, .mp3 or .m4a. Leave blank to run on text alone.
          </p>
        </Channel>

        <Channel tag="CHANNEL 3" title="Video"
                 role="Posture, movement, gaze. Observation only — no emotion label.">
          <label htmlFor="vf" className="btn ghost"
                 style={{ cursor: 'pointer', display: 'block', textAlign: 'center' }}>
            {uploading === 'video' ? 'Uploading…' : 'Choose a video file'}
          </label>
          <input id="vf" type="file" accept="video/*,.mp4,.mov,.m4v" className="sr-only"
                 onChange={(e) => upload(e.target.files?.[0], 'video')} />
          {videoInfo && (
            <p className="mono small" style={{ marginTop: 8, color: 'var(--gain)' }}>
              {videoInfo.filename} · {videoInfo.duration_s}s · {videoInfo.resolution}
              {videoInfo.warning && (
                <span style={{ color: 'var(--regress)' }}> · {videoInfo.warning}</span>
              )}
            </p>
          )}
          <p className="small muted" style={{ marginTop: 8 }}>
            Optional. .mp4 or .mov, straight from a phone. First 30 seconds are analysed.
          </p>
        </Channel>
      </div>

      <div className="row" style={{ marginTop: 20 }}>
        <button onClick={run} disabled={busy}>
          {busy ? 'Analysing…' : 'Analyse session'}
        </button>
        <span className="small muted" style={{ alignSelf: 'center' }}>
          Video takes roughly half the clip length to process.
        </span>
      </div>

      {out && (
        <>
          <h2 style={{ marginTop: 28 }}>
            Channels active: {out.channels_active} of 3
          </h2>
          <div className="panel flush">
            <table>
              <thead>
                <tr>
                  <th>Channel</th><th>Status</th><th>Detail</th><th>Role</th>
                </tr>
              </thead>
              <tbody>
                {['text', 'audio', 'video'].map((k) => {
                  const c = ch[k] || {}
                  return (
                    <tr key={k}>
                      <td style={{ textTransform: 'capitalize', fontWeight: 600 }}>{k}</td>
                      <td>
                        <span className={`state-tag ${c.available ? 'state-gaining' : 'state-steady'}`}>
                          {c.available ? 'active' : 'not supplied'}
                        </span>
                      </td>
                      <td className="mono small">
                        {k === 'text' && c.available &&
                          `TPI ${c.tpi} · confidence ${(c.confidence * 100).toFixed(0)}% · ${c.n_client_turns} turns`}
                        {k === 'audio' && c.available &&
                          `${c.n_features} features · f0 ${c.f0_mean} Hz · pause ratio ${c.pause_ratio}`}
                        {k === 'video' && c.available &&
                          `pose ${(c.pose_detection_rate * 100).toFixed(0)}% · face ${(c.face_detection_rate * 100).toFixed(0)}% · ${c.n_findings} findings`}
                        {!c.available && <span className="muted">{c.reason}</span>}
                      </td>
                      <td className="small muted">{c.role || '—'}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>

          <div className="panel">
            <div className="topline" style={{ background: 'var(--gain)' }} />
            <h2>Combined brief</h2>
            <div style={{ fontSize: 15.5, lineHeight: 1.65 }}>
              {out.brief?.what_was_said && (
                <>
                  <div className="eyebrow">What was said</div>
                  <p>{out.brief.what_was_said}</p>
                </>
              )}
              {out.brief?.what_the_body_did && (
                <>
                  <div className="eyebrow" style={{ marginTop: 12 }}>What the body did</div>
                  <p>{out.brief.what_the_body_did}</p>
                </>
              )}
              {out.brief?.how_it_sounded && (
                <>
                  <div className="eyebrow" style={{ marginTop: 12 }}>How it sounded</div>
                  <p>{out.brief.how_it_sounded}</p>
                </>
              )}
            </div>
            <p className="footnote" style={{ marginTop: 12 }}>{out.brief?.principle}</p>
          </div>

          {out.fusion && (
            <div className="panel">
              <h2>Fused index</h2>
              <div className="readout">
                <div className="readout-item">
                  <div className="k">Text only</div>
                  <div className="v">{out.fusion.tpi_text}</div>
                </div>
                <div className="readout-item">
                  <div className="k">With other channels</div>
                  <div className="v">{out.fusion.tpi_multimodal}</div>
                </div>
                <div className="readout-item">
                  <div className="k">Shift applied</div>
                  <div className="v">
                    {out.fusion.total_shift > 0 ? '+' : ''}{out.fusion.total_shift}
                  </div>
                </div>
                <div className="readout-item">
                  <div className="k">Agreement</div>
                  <div className="v" style={{ fontSize: 19 }}>{out.fusion.agreement}</div>
                </div>
              </div>
              <p className="footnote" style={{ marginTop: 12 }}>{out.fusion.note}</p>
            </div>
          )}

          {out.moments?.length > 0 && (
            <div className="panel">
              <h2>Moments worth reviewing</h2>
              <table>
                <thead><tr><th>Time</th><th>Salience</th><th>What was being said</th></tr></thead>
                <tbody>
                  {out.moments.map((m, i) => (
                    <tr key={i}>
                      <td className="mono">{m.timestamp}</td>
                      <td className="num">{m.salience}</td>
                      <td className="small">{m.utterance || <span className="muted">—</span>}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          <div className="notice plain">{out.principle}</div>
        </>
      )}
    </>
  )
}
