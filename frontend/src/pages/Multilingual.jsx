import { useState } from 'react'

import { api, DIMENSION_LABELS } from '../lib/api'

const SAMPLES = {
  ta: 'சிகிச்சையாளர்: இந்த வாரம் எப்படி இருந்தது?\nவாடிக்கையாளர்: நான் அவளிடம் பேச முடிவு செய்தேன். கொஞ்சம் பயமாக இருந்தது, ஆனால் அடுத்த வாரம் மீண்டும் முயற்சிப்பேன்.',
  hi: 'चिकित्सक: यह हफ्ता कैसा रहा?\nक्लाइंट: मैंने उससे बात करने का फैसला किया। थोड़ा डर लगा, लेकिन अगले हफ्ते मैं फिर कोशिश करूँगा।',
}

/** Score a Tamil or Hindi transcript via turn-by-turn translation to English. */
export default function Multilingual() {
  const [text, setText] = useState('')
  const [lang, setLang] = useState('auto')
  const [res, setRes] = useState(null)
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)

  const run = async (e) => {
    e.preventDefault()
    setBusy(true); setError(null); setRes(null)
    try { setRes(await api.analyzeTranslated(text, lang)) }
    catch (e) { setError(e.message) } finally { setBusy(false) }
  }

  const feats = res?.analysis?.features || {}

  return (
    <>
      <div className="eyebrow">Multilingual · exploratory</div>
      <h1>Tamil and Hindi transcripts</h1>
      <p className="lede">
        Each turn is translated to English and then scored by the same pipeline.
        The lexicon was built for English, so treat these scores as a first look,
        and check the original beside the translation.
      </p>

      <form className="panel" onSubmit={run}>
        <div className="field">
          <label htmlFor="ml-t">Transcript (speaker-labelled; Tamil, Hindi or English labels all work)</label>
          <textarea id="ml-t" rows={7} value={text} onChange={(e) => setText(e.target.value)}
                    placeholder={'சிகிச்சையாளர்: ...\nவாடிக்கையாளர்: ...'} />
        </div>
        <div className="row" style={{ flexWrap: 'wrap' }}>
          <div className="field" style={{ width: 170, marginBottom: 0 }}>
            <label htmlFor="ml-l">Language</label>
            <select id="ml-l" value={lang} onChange={(e) => setLang(e.target.value)}>
              <option value="auto">Detect automatically</option>
              <option value="ta">Tamil</option>
              <option value="hi">Hindi</option>
              <option value="en">English</option>
            </select>
          </div>
          <button type="submit" disabled={busy || text.trim().length < 5}>
            {busy ? 'Translating and scoring…' : 'Translate and score'}
          </button>
          <button type="button" className="ghost" onClick={() => { setText(SAMPLES.ta); setLang('auto') }}>Tamil sample</button>
          <button type="button" className="ghost" onClick={() => { setText(SAMPLES.hi); setLang('auto') }}>Hindi sample</button>
        </div>
      </form>

      {error && <div className="notice">{error}</div>}

      {res && (
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
                <FragmentRow key={i} t={t} />
              ))}
            </div>
          </div>
        </>
      )}
    </>
  )
}

function FragmentRow({ t }) {
  return (
    <>
      <div><span className="spk">{t.original_speaker || t.speaker || ''}</span>{t.original}</div>
      <div><span className="spk">{t.speaker || ''}</span>{t.english}</div>
    </>
  )
}
