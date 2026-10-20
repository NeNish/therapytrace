import { useRef, useState } from 'react'

import { api, DIMENSION_LABELS } from '../lib/api'

const SAMPLES = {
  ta: 'சிகிச்சையாளர்: இந்த வாரம் எப்படி இருந்தது?\nவாடிக்கையாளர்: நான் அவளிடம் பேச முடிவு செய்தேன். கொஞ்சம் பயமாக இருந்தது, ஆனால் அடுத்த வாரம் மீண்டும் முயற்சிப்பேன்.',
  hi: 'चिकित्सक: यह हफ्ता कैसा रहा?\nक्लाइंट: मैंने उससे बात करने का फैसला किया। थोड़ा डर लगा, लेकिन अगले हफ्ते मैं फिर कोशिश करूँगा।',
}

const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`

/** Score Tamil or Hindi sessions, from a pasted transcript or a recording. */
export default function Multilingual() {
  const [tab, setTab] = useState('text')
  const [res, setRes] = useState(null)
  const [error, setError] = useState(null)

  return (
    <>
      <div className="eyebrow">Multilingual · exploratory</div>
      <h1>Tamil and Hindi sessions</h1>
      <p className="lede">
        Paste a transcript or upload a recording. Each turn is translated to
        English and then scored by the same pipeline. The lexicon was built for
        English, so treat these scores as a first look, and check the original
        beside the translation.
      </p>

      <div className="row" style={{ marginBottom: 16 }}>
        <button type="button" className={tab === 'text' ? '' : 'ghost'}
                onClick={() => { setTab('text'); setRes(null); setError(null) }}>Paste transcript</button>
        <button type="button" className={tab === 'audio' ? '' : 'ghost'}
                onClick={() => { setTab('audio'); setRes(null); setError(null) }}>Upload or record audio</button>
      </div>

      {tab === 'text'
        ? <TextInput onResult={setRes} onError={setError} />
        : <AudioInput onResult={setRes} onError={setError} />}

      {error && <div className="notice">{error}</div>}
      {res && <Results res={res} />}
    </>
  )
}

function TextInput({ onResult, onError }) {
  const [text, setText] = useState('')
  const [lang, setLang] = useState('auto')
  const [busy, setBusy] = useState(false)

  const run = async (e) => {
    e.preventDefault()
    setBusy(true); onError(null); onResult(null)
    try { onResult(await api.analyzeTranslated(text, lang)) }
    catch (err) { onError(err.message) } finally { setBusy(false) }
  }

  return (
    <form className="panel" onSubmit={run}>
      <div className="field">
        <label htmlFor="ml-t">Transcript (speaker-labelled; Tamil, Hindi or English labels all work)</label>
        <textarea id="ml-t" rows={7} value={text} onChange={(e) => setText(e.target.value)}
                  placeholder={'சிகிச்சையாளர்: ...\nவாடிக்கையாளர்: ...'} />
      </div>
      <div className="row" style={{ flexWrap: 'wrap' }}>
        <LanguageSelect id="ml-l" value={lang} onChange={setLang} />
        <button type="submit" disabled={busy || text.trim().length < 5}>
          {busy ? 'Translating and scoring…' : 'Translate and score'}
        </button>
        <button type="button" className="ghost" onClick={() => { setText(SAMPLES.ta); setLang('auto') }}>Tamil sample</button>
        <button type="button" className="ghost" onClick={() => { setText(SAMPLES.hi); setLang('auto') }}>Hindi sample</button>
      </div>
    </form>
  )
}

function AudioInput({ onResult, onError }) {
  const [file, setFile] = useState(null)
  const [lang, setLang] = useState('auto')
  const [busy, setBusy] = useState(null)
  const [tx, setTx] = useState(null)
  const [recording, setRecording] = useState(false)
  const recRef = useRef(null)
  const chunksRef = useRef([])

  const startRec = async () => {
    onError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const rec = new MediaRecorder(stream)
      chunksRef.current = []
      rec.ondataavailable = (e) => { if (e.data.size) chunksRef.current.push(e.data) }
      rec.onstop = () => {
        stream.getTracks().forEach((t) => t.stop())
        const blob = new Blob(chunksRef.current, { type: rec.mimeType || 'audio/webm' })
        const ext = (rec.mimeType || '').includes('mp4') ? 'm4a' : 'webm'
        setFile(new File([blob], `recording.${ext}`, { type: blob.type }))
      }
      rec.start()
      recRef.current = rec
      setRecording(true)
    } catch (err) {
      onError(`Microphone not available: ${err.message}`)
    }
  }

  const stopRec = () => { recRef.current?.stop(); setRecording(false) }

  const transcribe = async () => {
    if (!file) return
    setBusy('transcribe'); onError(null); onResult(null); setTx(null)
    try { setTx(await api.transcribeAudio(file, lang)) }
    catch (err) { onError(err.message) } finally { setBusy(null) }
  }

  const flip = (i) => setTx({
    ...tx,
    lines: tx.lines.map((ln) => ln.id === i
      ? { ...ln, speaker: ln.speaker === 'CLIENT' ? 'THERAPIST' : 'CLIENT' } : ln),
  })

  const score = async () => {
    setBusy('score'); onError(null); onResult(null)
    try {
      onResult(await api.analyzeTurns(tx.language, tx.lines.map((ln) => ({
        speaker: ln.speaker, original: ln.original, english: ln.english || ln.original,
      }))))
    } catch (err) { onError(err.message) } finally { setBusy(null) }
  }

  const untranslated = tx && tx.language !== 'en' && tx.lines.some((ln) => !ln.english)

  return (
    <>
      <div className="panel">
        <p className="small muted" style={{ marginTop: 0 }}>
          The recording is transcribed on this machine with Whisper and deleted
          straight after. Supported: Tamil, Hindi, English (.wav, .mp3, .m4a, .webm).
        </p>
        <div className="row" style={{ flexWrap: 'wrap' }}>
          <label htmlFor="ml-a" className="btn ghost" style={{ cursor: 'pointer' }}>
            {file ? `Selected: ${file.name}` : 'Choose audio file'}
          </label>
          <input id="ml-a" type="file" className="sr-only"
                 accept="audio/*,.wav,.mp3,.m4a,.flac,.ogg,.webm"
                 onChange={(e) => setFile(e.target.files?.[0] || null)} />
          {!recording
            ? <button type="button" className="ghost" onClick={startRec}>Record from mic</button>
            : <button type="button" onClick={stopRec}>Stop recording</button>}
          <LanguageSelect id="ml-al" value={lang} onChange={setLang} />
          <button type="button" disabled={!file || recording || busy} onClick={transcribe}>
            {busy === 'transcribe' ? 'Transcribing… (first run loads the model)' : 'Transcribe'}
          </button>
        </div>
      </div>

      {tx && (
        <div className="panel">
          <div className="panel-head">
            <h2 style={{ margin: 0 }}>Check who said what</h2>
            <span className="chip">
              {tx.language_name}
              {tx.language_probability != null ? ` (${Math.round(tx.language_probability * 100)}% sure)` : ''}
              {' · '}{tx.speech_model}
              {tx.translation_provider ? ` · translated via ${tx.translation_provider}` : ''}
            </span>
          </div>
          <p className="small muted">{tx.note} Click a label to switch it.</p>
          {tx.translation_error && <div className="notice">{tx.translation_error}</div>}
          <table>
            <thead>
              <tr><th style={{ width: 60 }}>Time</th><th style={{ width: 120 }}>Speaker</th><th>Original</th><th>English</th></tr>
            </thead>
            <tbody>
              {tx.lines.map((ln) => (
                <tr key={ln.id}>
                  <td className="mono small">{fmt(ln.start)}</td>
                  <td>
                    <button type="button" className={ln.speaker === 'CLIENT' ? '' : 'ghost'}
                            style={{ padding: '4px 10px', fontSize: 12 }} onClick={() => flip(ln.id)}>
                      {ln.speaker === 'CLIENT' ? 'Client' : 'Therapist'}
                    </button>
                  </td>
                  <td>{ln.original}</td>
                  <td>{ln.english ?? <span className="muted">—</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="row" style={{ marginTop: 14 }}>
            <button type="button" disabled={busy || untranslated} onClick={score}>
              {busy === 'score' ? 'Scoring…' : 'Score this transcript'}
            </button>
            {untranslated && <span className="small muted">Translation is needed before scoring.</span>}
          </div>
        </div>
      )}
    </>
  )
}

function LanguageSelect({ id, value, onChange }) {
  return (
    <div className="field" style={{ width: 170, marginBottom: 0 }}>
      <label htmlFor={id}>Language</label>
      <select id={id} value={value} onChange={(e) => onChange(e.target.value)}>
        <option value="auto">Detect automatically</option>
        <option value="ta">Tamil</option>
        <option value="hi">Hindi</option>
        <option value="en">English</option>
      </select>
    </div>
  )
}

function Results({ res }) {
  const feats = res?.analysis?.features || {}
  return (
    <>
      <div className="panel">
        <div className="panel-head">
          <h2>Scores</h2>
          <span className="chip">{res.language_name} · via {res.provider}</span>
        </div>
        <div className="readout">
          <div className="readout-item">
            <div className="k">Process index (absolute)</div>
            <div className="v">{res.analysis?.score?.tpi?.toFixed?.(1) ?? '—'}</div>
          </div>
          {Object.entries(DIMENSION_LABELS).map(([k, v]) => (
            <div className="readout-item" key={k}>
              <div className="k">{v}</div>
              <div className="v">{feats[k] != null ? feats[k].toFixed(2) : '—'}</div>
            </div>
          ))}
        </div>
        {res.caveat && <p className="footnote" style={{ marginTop: 12 }}>{res.caveat}</p>}
      </div>

      <div className="panel flush">
        <div className="side-by-side">
          <div><strong>Original</strong></div>
          <div><strong>English (what was scored)</strong></div>
          {res.turns.map((t, i) => (
            <TurnRow key={i} t={t} />
          ))}
        </div>
      </div>
    </>
  )
}

function TurnRow({ t }) {
  return (
    <>
      <div><span className="spk">{t.original_speaker || t.speaker || ''}</span>{t.original}</div>
      <div><span className="spk">{t.speaker || ''}</span>{t.english}</div>
    </>
  )
}
