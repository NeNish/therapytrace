import { useState } from 'react'

import { FrameCard, FrameImg, PreviewGallery, SupervisionList } from '../components/FramePreview'
import { api, DIMENSION_LABELS } from '../lib/api'

const SAMPLE = `THERAPIST: How has the week been?
CLIENT: I decided not to go on Saturday, and I told her why. I felt anxious saying it and also steadier afterwards.
THERAPIST: What was that like?
CLIENT: I noticed the pattern this time. When I feel dismissed I withdraw, and then I read the silence as proof I was right.
THERAPIST: You noticed that yourself, which is not a small thing.
CLIENT: My part in it is that I wait until I'm furious and then I deliver it like an accusation. That's on me.`

function Channel({ tag, title, role, primary, active, children }) {
  return (
    <div
      className="panel"
      style={{
        marginBottom: 0,
        borderColor: active ? 'var(--gain)' : primary ? 'var(--ink-3)' : 'var(--rule)',
        borderWidth: active ? 2 : 1,
        opacity: active === false ? 0.72 : 1,
      }}
    >
      <div className="topline" style={{ background: active ? 'var(--gain)' : 'var(--rule)' }} />
      <div className="eyebrow" style={{ color: active ? 'var(--gain)' : 'var(--ink-3)' }}>
        {tag} {active && '· ACTIVE'}
      </div>
      <h2 style={{ marginBottom: 4 }}>{title}</h2>
      <p className="small muted" style={{ marginTop: 0, minHeight: 34 }}>{role}</p>
      {children}
    </div>
  )
}

function InsightPanel({ title, insights, kind }) {
  if (!insights?.available) {
    return (
      <div className="panel" style={{ opacity: 0.7 }}>
        <h2>{title}</h2>
        <p className="small muted">
          {insights?.reason || `No ${kind} input supplied — upload a file or paste a transcript to activate this channel.`}
        </p>
      </div>
    )
  }

  return (
    <div className="panel">
      <div className="topline" style={{ background: 'var(--gain)' }} />
      <h2>{title}</h2>
      {insights.summary && <p style={{ fontSize: 15.5, lineHeight: 1.65 }}>{insights.summary}</p>}
      {insights.engagement_label && (
        <p className="small" style={{ marginTop: 8 }}>
          <strong>Engagement arc:</strong> {insights.engagement_label}
        </p>
      )}
      {insights.bullets?.length > 0 && (
        <ul className="small" style={{ marginTop: 12, paddingLeft: 18, lineHeight: 1.7 }}>
          {insights.bullets.map((b, i) => <li key={i}>{b}</li>)}
        </ul>
      )}
      <SupervisionList prompts={insights.supervision_prompts} />

      {kind === 'text' && insights.dimensions && (
        <table style={{ marginTop: 14 }}>
          <thead><tr><th>Dimension</th><th className="num">Score</th></tr></thead>
          <tbody>
            {Object.entries(insights.dimensions).map(([k, v]) => (
              <tr key={k}>
                <td>{DIMENSION_LABELS[k] || k}</td>
                <td className="num">{(v * 100).toFixed(0)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {kind === 'text' && insights.weakest_turns?.length > 0 && (
        <div style={{ marginTop: 14 }}>
          <div className="eyebrow">Turns to explore</div>
          {insights.weakest_turns.map((t, i) => (
            <blockquote key={i} className="regress" style={{ marginBottom: 8 }}>
              <span className="who">Low process · {t.process_mean?.toFixed(2)}</span>
              {t.text?.slice(0, 180)}
            </blockquote>
          ))}
        </div>
      )}

      {kind === 'video' && insights.timeline?.length > 0 && (
        <div style={{ marginTop: 14 }}>
          <div className="eyebrow">Posture timeline</div>
          {insights.timeline.map((t, i) => (
            <div
              key={i}
              style={{
                display: 'grid',
                gridTemplateColumns: t.preview ? '120px 1fr' : '1fr',
                gap: 12,
                marginTop: 12,
                paddingBottom: 12,
                borderBottom: '1px solid var(--rule)',
              }}
            >
              {t.preview && (
                <FrameImg preview={t.preview} style={{ maxHeight: 90, objectFit: 'cover' }} />
              )}
              <div>
                <div className="mono small" style={{ fontWeight: 600 }}>{t.timestamp}</div>
                <div className="small muted">{t.kind?.replace(/_/g, ' ')}</div>
                <p className="small" style={{ margin: '4px 0' }}>{t.sentence}</p>
                {t.clinical_note && (
                  <p className="small muted" style={{ margin: '4px 0 0' }}>{t.clinical_note}</p>
                )}
                {t.supervision_question && (
                  <p className="small" style={{ margin: '6px 0 0', fontStyle: 'italic' }}>
                    {t.supervision_question}
                  </p>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {kind === 'audio' && insights.metrics && (
        <div className="readout" style={{ marginTop: 14 }}>
          {Object.entries(insights.metrics).map(([k, v]) => (
            <div className="readout-item" key={k}>
              <div className="k">{k.replace(/_/g, ' ')}</div>
              <div className="v" style={{ fontSize: 17 }}>{typeof v === 'number' ? v.toFixed(2) : v}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default function Multimodal() {
  const [transcript, setTranscript] = useState(SAMPLE)
  const [includeText, setIncludeText] = useState(true)
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

  const clearChannel = (kind) => {
    if (kind === 'audio') { setAudioPath(''); setAudioInfo(null) }
    if (kind === 'video') { setVideoPath(''); setVideoInfo(null) }
    if (kind === 'text') { setTranscript(''); setIncludeText(false) }
  }

  const run = async () => {
    setBusy(true); setError(null)
    try {
      const payload = {
        audio_path: audioPath || undefined,
        video_path: videoPath || undefined,
        max_seconds: 60,
      }
      if (includeText && transcript.trim().length >= 20) {
        payload.transcript = transcript
      }
      setOut(await api.multimodal(payload))
    } catch (e) { setError(e.message) } finally { setBusy(false) }
  }

  const canRun = (includeText && transcript.trim().length >= 20) || audioPath || videoPath
  const ch = out?.channels || {}
  const ins = out?.insights || {}

  const momentCards = (out?.moments || []).filter(
    (m) => m.preview?.preview_jpeg_b64 || m.preview_jpeg_b64,
  )

  return (
    <>
      <div className="eyebrow">Multimodal analysis</div>
      <h1>Analyse what you have</h1>
      <p className="lede">
        Supply any combination — transcript, video, audio, or all three.
        Video analysis includes frame previews at key timestamps and supervision
        questions a therapist can use in review (observations only, never emotion labels).
      </p>

      {error && <div className="notice">{error}</div>}

      <div className="panel" style={{ marginBottom: 20 }}>
        <h2>What are you analysing?</h2>
        <div className="row" style={{ gap: 12, flexWrap: 'wrap' }}>
          {[
            ['text', 'Transcript', includeText],
            ['audio', 'Audio', !!audioPath],
            ['video', 'Video', !!videoPath],
          ].map(([k, label, on]) => (
            <span
              key={k}
              className={`state-tag ${on ? 'state-gaining' : 'state-steady'}`}
              style={{ fontSize: 13 }}
            >
              {on ? `${label} ✓` : `${label} — not supplied`}
            </span>
          ))}
        </div>
      </div>

      <div className="grid-3">
        <Channel tag="CHANNEL 1" title="Text" primary={includeText} active={!!ch.text?.available}
                 role="Language process markers — strongest validated signal.">
          <label className="small" style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 8 }}>
            <input type="checkbox" checked={includeText} onChange={(e) => setIncludeText(e.target.checked)} />
            Include transcript
          </label>
          {includeText && (
            <>
              <textarea rows={8} value={transcript} onChange={(e) => setTranscript(e.target.value)} />
              <button type="button" className="btn ghost small" style={{ marginTop: 6 }}
                      onClick={() => clearChannel('text')}>Clear text</button>
            </>
          )}
        </Channel>

        <Channel tag="CHANNEL 2" title="Audio" active={!!ch.audio?.available}
                 role="Pitch, pauses, vocal effort — with supervision listening points.">
          <label htmlFor="af" className="btn ghost"
                 style={{ cursor: 'pointer', display: 'block', textAlign: 'center' }}>
            {uploading === 'audio' ? 'Uploading…' : 'Choose an audio file'}
          </label>
          <input id="af" type="file" accept="audio/*,.wav,.mp3,.m4a" className="sr-only"
                 onChange={(e) => upload(e.target.files?.[0], 'audio')} />
          {audioInfo && (
            <>
              <p className="mono small" style={{ marginTop: 8, color: 'var(--gain)' }}>
                {audioInfo.filename} · {audioInfo.size_mb} MB
              </p>
              <button type="button" className="btn ghost small" onClick={() => clearChannel('audio')}>Remove</button>
            </>
          )}
          <p className="small muted" style={{ marginTop: 8 }}>.wav, .mp3 or .m4a</p>
        </Channel>

        <Channel tag="CHANNEL 3" title="Video" active={!!ch.video?.available}
                 role="Posture, movement, frame previews — observation only.">
          <label htmlFor="vf" className="btn ghost"
                 style={{ cursor: 'pointer', display: 'block', textAlign: 'center' }}>
            {uploading === 'video' ? 'Uploading…' : 'Choose a video file'}
          </label>
          <input id="vf" type="file" accept="video/*,.mp4,.mov,.m4v" className="sr-only"
                 onChange={(e) => upload(e.target.files?.[0], 'video')} />
          {videoInfo && (
            <>
              <p className="mono small" style={{ marginTop: 8, color: 'var(--gain)' }}>
                {videoInfo.filename}
                {videoInfo.duration_s != null && ` · ${videoInfo.duration_s}s`}
                {videoInfo.resolution && ` · ${videoInfo.resolution}`}
              </p>
              <button type="button" className="btn ghost small" onClick={() => clearChannel('video')}>Remove</button>
            </>
          )}
          <p className="small muted" style={{ marginTop: 8 }}>.mp4 or .mov — first 60 seconds analysed</p>
        </Channel>
      </div>

      <div className="row" style={{ marginTop: 20 }}>
        <button onClick={run} disabled={busy || !canRun}>
          {busy ? 'Analysing…' : 'Analyse session'}
        </button>
        {!canRun && (
          <span className="small muted">Provide at least one channel to run analysis.</span>
        )}
      </div>

      {out && (
        <>
          <div className="panel" style={{ marginTop: 28 }}>
            <div className="topline" style={{ background: 'var(--gain)' }} />
            <div className="eyebrow">{out.input_mode_label}</div>
            <h2>{out.channels_active} channel{out.channels_active !== 1 ? 's' : ''} active</h2>
            <div className="readout">
              {out.fusion?.tpi_text != null && (
                <div className="readout-item">
                  <div className="k">Text process index</div>
                  <div className="v">{out.fusion.tpi_text}</div>
                </div>
              )}
              <div className="readout-item">
                <div className="k">{out.fusion?.tpi_text != null ? 'Fused index' : 'Channel index'}</div>
                <div className="v">{out.fusion?.tpi_multimodal}</div>
              </div>
              {out.fusion?.total_shift !== 0 && (
                <div className="readout-item">
                  <div className="k">Channel shift</div>
                  <div className="v" style={{ color: out.fusion.total_shift > 0 ? 'var(--gain)' : 'var(--regress)' }}>
                    {out.fusion.total_shift > 0 ? '+' : ''}{out.fusion.total_shift}
                  </div>
                </div>
              )}
              <div className="readout-item">
                <div className="k">Agreement</div>
                <div className="v" style={{ fontSize: 17 }}>{out.fusion?.agreement?.replace(/_/g, ' ')}</div>
              </div>
            </div>
            <p className="footnote" style={{ marginTop: 12 }}>{out.fusion?.note}</p>
          </div>

          {/* Frame gallery — headline feature for video */}
          {(momentCards.length > 0 || out.preview_gallery?.length > 0) && (
            <div className="panel">
              <div className="topline" style={{ background: 'var(--mark, var(--gain))' }} />
              <h2>Frame previews at key timestamps</h2>
              <p className="small muted" style={{ marginTop: -6 }}>
                Jump to these moments in the recording. Overlays show measured
                gesture level — not inferred emotion.
              </p>
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))',
                  gap: 16,
                  marginTop: 16,
                }}
              >
                {(momentCards.length ? momentCards : out.preview_gallery).map((m, i) => (
                  <FrameCard
                    key={i}
                    preview={m.preview || m}
                    caption={m.description || m.observation || m.sentence}
                    question={m.supervision_question}
                    tags={m.tags}
                  />
                ))}
              </div>
            </div>
          )}

          <h2 style={{ marginTop: 28 }}>Insights by channel</h2>
          <div className="grid-3" style={{ alignItems: 'start' }}>
            <InsightPanel title="Text insights" insights={ins.text} kind="text" />
            <InsightPanel title="Audio insights" insights={ins.audio} kind="audio" />
            <InsightPanel title="Video insights" insights={ins.video} kind="video" />
          </div>

          {out.brief?.brief && (
            <div className="panel">
              <h2>Combined brief</h2>
              <div style={{ fontSize: 15.5, lineHeight: 1.65, whiteSpace: 'pre-wrap' }}>{out.brief.brief}</div>
            </div>
          )}

          {out.moments?.length > 0 && (
            <div className="panel">
              <h2>Key moments — detailed review</h2>
              <table>
                <thead>
                  <tr>
                    <th style={{ width: 100 }}>Preview</th>
                    <th>Time</th>
                    <th>Salience</th>
                    <th>Observation</th>
                    <th>Supervision question</th>
                    <th>Words (if aligned)</th>
                  </tr>
                </thead>
                <tbody>
                  {out.moments.map((m, i) => (
                    <tr key={i}>
                      <td>
                        {m.preview ? (
                          <FrameImg preview={m.preview} style={{ maxWidth: 96, maxHeight: 72, objectFit: 'cover' }} />
                        ) : (
                          <span className="muted small">—</span>
                        )}
                      </td>
                      <td className="mono">{m.timestamp}</td>
                      <td className="num">{m.salience}</td>
                      <td className="small">{m.observation || m.description || m.why}</td>
                      <td className="small muted">{m.supervision_question || '—'}</td>
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
