import { useEffect, useRef, useState } from 'react'
import { api } from '../lib/api'
import { downsampleBuffer, encodeWAV } from '../lib/audioRecorder'

const SpeechRecognition =
  typeof window !== 'undefined'
    ? window.SpeechRecognition || window.webkitSpeechRecognition
    : null

export default function LiveRecorder({
  onComplete,
  defaultMode = 'audio',
  showTranscript = true,
  initialTranscript = '',
  allowModeSwitch = true,
  standalone = false,
}) {
  const [mode, setMode] = useState(defaultMode) // 'audio' | 'video' | 'dictation'
  const [status, setStatus] = useState('idle') // 'idle' | 'recording' | 'paused' | 'finished'
  const [duration, setDuration] = useState(0)
  const [micLevel, setMicLevel] = useState(0)
  const [speaker, setSpeaker] = useState('THERAPIST') // 'THERAPIST' | 'CLIENT'
  const [transcript, setTranscript] = useState(initialTranscript)
  const [interimText, setInterimText] = useState('')
  const [recordedMedia, setRecordedMedia] = useState(null) // { blob, url, kind, filename }
  const [uploadedInfo, setUploadedInfo] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState(null)
  const [hasSpeechApi, setHasSpeechApi] = useState(!!SpeechRecognition)

  // Hardware device selection
  const [audioDevices, setAudioDevices] = useState([])
  const [videoDevices, setVideoDevices] = useState([])
  const [selectedAudioId, setSelectedAudioId] = useState('')
  const [selectedVideoId, setSelectedVideoId] = useState('')
  const [previewingVideo, setPreviewingVideo] = useState(false)
  const previewStreamRef = useRef(null)

  // Media references
  const streamRef = useRef(null)
  const mediaRecorderRef = useRef(null)
  const audioContextRef = useRef(null)
  const analyserRef = useRef(null)
  const animationFrameRef = useRef(null)
  const pcmDataRef = useRef([])
  const timerRef = useRef(null)
  const recognitionRef = useRef(null)
  const canvasRef = useRef(null)
  const videoPreviewRef = useRef(null)
  const playbackVideoRef = useRef(null)

  // Enumerate input devices
  const refreshDevices = async (requestPermission = false) => {
    if (!navigator.mediaDevices?.enumerateDevices) return
    try {
      if (requestPermission) {
        // Request temporary stream to obtain hardware labels
        const tempStream = await navigator.mediaDevices.getUserMedia({
          audio: true,
          video: mode === 'video',
        })
        tempStream.getTracks().forEach((t) => t.stop())
      }
      const devices = await navigator.mediaDevices.enumerateDevices()
      const audios = devices.filter((d) => d.kind === 'audioinput')
      const videos = devices.filter((d) => d.kind === 'videoinput')

      setAudioDevices(audios)
      setVideoDevices(videos)

      if (audios.length > 0 && !selectedAudioId) {
        setSelectedAudioId(audios[0].deviceId)
      }
      if (videos.length > 0 && !selectedVideoId) {
        setSelectedVideoId(videos[0].deviceId)
      }
    } catch (e) {
      console.warn('Device refresh warning:', e)
    }
  }

  // Toggle live video preview before recording starts
  const toggleVideoPreview = async () => {
    if (previewingVideo) {
      if (previewStreamRef.current) {
        previewStreamRef.current.getTracks().forEach((t) => t.stop())
        previewStreamRef.current = null
      }
      if (videoPreviewRef.current) {
        videoPreviewRef.current.srcObject = null
      }
      setPreviewingVideo(false)
      return
    }

    try {
      const pStream = await navigator.mediaDevices.getUserMedia({
        video: {
          deviceId: selectedVideoId ? { exact: selectedVideoId } : undefined,
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
      })
      previewStreamRef.current = pStream
      if (videoPreviewRef.current) {
        videoPreviewRef.current.srcObject = pStream
        videoPreviewRef.current.play().catch(() => {})
      }
      setPreviewingVideo(true)
    } catch (err) {
      setError(`Camera preview error: ${err.message}`)
    }
  }

  // Check speech recognition & enumerate devices on mount
  useEffect(() => {
    setHasSpeechApi(!!SpeechRecognition)
    refreshDevices(false)

    const handleDeviceChange = () => refreshDevices(false)
    navigator.mediaDevices?.addEventListener?.('devicechange', handleDeviceChange)
    return () => {
      navigator.mediaDevices?.removeEventListener?.('devicechange', handleDeviceChange)
    }
  }, [mode])

  // Format seconds to mm:ss
  const formatTime = (secs) => {
    const m = Math.floor(secs / 60)
    const s = secs % 60
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
  }

  // Draw real-time animated waveform on canvas
  const drawWaveform = () => {
    if (!analyserRef.current || !canvasRef.current) return
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    const analyser = analyserRef.current
    const bufferLength = analyser.frequencyBinCount
    const dataArray = new Uint8Array(bufferLength)

    const render = () => {
      animationFrameRef.current = requestAnimationFrame(render)
      analyser.getByteTimeDomainData(dataArray)

      // Calculate simple RMS volume
      let sumSquares = 0
      for (let i = 0; i < bufferLength; i++) {
        const norm = (dataArray[i] - 128) / 128
        sumSquares += norm * norm
      }
      const rms = Math.sqrt(sumSquares / bufferLength)
      setMicLevel(Math.min(100, Math.round(rms * 250)))

      // Canvas dimensions
      const width = canvas.width
      const height = canvas.height
      ctx.clearRect(0, 0, width, height)

      // Background chart-paper grid lines
      ctx.strokeStyle = 'rgba(18, 28, 27, 0.05)'
      ctx.lineWidth = 1
      ctx.beginPath()
      for (let x = 0; x < width; x += 20) {
        ctx.moveTo(x, 0)
        ctx.lineTo(x, height)
      }
      for (let y = 0; y < height; y += 15) {
        ctx.moveTo(0, y)
        ctx.lineTo(width, y)
      }
      ctx.stroke()

      // Center reference line
      ctx.strokeStyle = 'rgba(90, 110, 107, 0.2)'
      ctx.beginPath()
      ctx.moveTo(0, height / 2)
      ctx.lineTo(width, height / 2)
      ctx.stroke()

      // Waveform line
      ctx.lineWidth = 2.2
      ctx.strokeStyle = status === 'recording' ? '#176352' : '#8b2646'
      ctx.beginPath()

      const sliceWidth = width / bufferLength
      let x = 0

      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0
        const y = (v * height) / 2

        if (i === 0) {
          ctx.moveTo(x, y)
        } else {
          ctx.lineTo(x, y)
        }
        x += sliceWidth
      }

      ctx.lineTo(width, height / 2)
      ctx.stroke()
    }

    render()
  }

  // Handle hotkey for switching speaker: Tab key
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (status === 'recording' && e.key === 'Tab') {
        e.preventDefault()
        setSpeaker((prev) => (prev === 'THERAPIST' ? 'CLIENT' : 'THERAPIST'))
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [status])

  // Setup Web Speech Recognition
  const initSpeechRecognition = () => {
    if (!SpeechRecognition) return null
    try {
      const recognition = new SpeechRecognition()
      recognition.continuous = true
      recognition.interimResults = true
      recognition.lang = 'en-US'

      recognition.onresult = (event) => {
        let interim = ''
        let final = ''

        for (let i = event.resultIndex; i < event.results.length; i++) {
          const res = event.results[i]
          if (res.isFinal) {
            final += res[0].transcript
          } else {
            interim += res[0].transcript
          }
        }

        setInterimText(interim)

        if (final.trim()) {
          setTranscript((prev) => {
            const trimmed = final.trim()
            const speakerTag = `${speaker}: `
            const lines = prev.split('\n')
            const lastLine = lines[lines.length - 1] || ''

            // Check if last line starts with same speaker
            if (lastLine.startsWith(speakerTag)) {
              lines[lines.length - 1] = `${lastLine} ${trimmed}`
              return lines.join('\n')
            } else {
              const prefix = prev.trim() ? '\n' : ''
              return `${prev.trim()}${prefix}${speakerTag}${trimmed}`
            }
          })
        }
      }

      recognition.onerror = (e) => {
        if (e.error !== 'no-speech') {
          console.warn('Speech recognition warning:', e.error)
        }
      }

      recognition.onend = () => {
        // Automatically restart if still in recording status
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
          try {
            recognition.start()
          } catch {
            // Already started or busy
          }
        }
      }

      return recognition
    } catch (e) {
      console.warn('Could not initialize SpeechRecognition:', e)
      return null
    }
  }

  // Start live recording
  const startRecording = async () => {
    setError(null)
    setDuration(0)
    setRecordedMedia(null)
    setUploadedInfo(null)
    pcmDataRef.current = []

    if (previewStreamRef.current) {
      previewStreamRef.current.getTracks().forEach((t) => t.stop())
      previewStreamRef.current = null
      setPreviewingVideo(false)
    }

    try {
      const constraints = {
        audio: {
          deviceId: selectedAudioId ? { exact: selectedAudioId } : undefined,
          echoCancellation: true,
          noiseSuppression: true,
          channelCount: 1,
        },
        video: mode === 'video' ? {
          deviceId: selectedVideoId ? { exact: selectedVideoId } : undefined,
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: selectedVideoId ? undefined : 'user',
        } : false,
      }

      const stream = await navigator.mediaDevices.getUserMedia(constraints)
      streamRef.current = stream

      // Set video preview if mode is video
      if (mode === 'video' && videoPreviewRef.current) {
        videoPreviewRef.current.srcObject = stream
        videoPreviewRef.current.play().catch(() => {})
      }

      // AudioContext + Analyser for visualizer & WAV capture
      const AudioCtx = window.AudioContext || window.webkitAudioContext
      const audioCtx = new AudioCtx()
      audioContextRef.current = audioCtx
      const source = audioCtx.createMediaStreamSource(stream)
      const analyser = audioCtx.createAnalyser()
      analyser.fftSize = 1024
      analyserRef.current = analyser
      source.connect(analyser)

      // ScriptProcessorNode for pure 16kHz PCM audio collection (if audio only)
      if (mode !== 'video') {
        const processor = audioCtx.createScriptProcessor(4096, 1, 1)
        processor.onaudioprocess = (e) => {
          if (status === 'recording' || mediaRecorderRef.current?.state === 'recording') {
            const input = e.inputBuffer.getChannelData(0)
            const copy = new Float32Array(input)
            pcmDataRef.current.push(copy)
          }
        }
        source.connect(processor)
        processor.connect(audioCtx.destination)
      }

      // MediaRecorder fallback / video container
      let mimeType = mode === 'video' ? 'video/webm' : 'audio/webm'
      if (!MediaRecorder.isTypeSupported(mimeType)) {
        mimeType = '' // let browser select default
      }

      const mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined)
      mediaRecorderRef.current = mediaRecorder
      const chunks = []

      mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) chunks.push(e.data)
      }

      mediaRecorder.onstop = async () => {
        let finalBlob
        let filename
        const kind = mode === 'video' ? 'video' : 'audio'

        if (mode === 'video') {
          finalBlob = new Blob(chunks, { type: mediaRecorder.mimeType || 'video/webm' })
          filename = `session_rec_${Date.now()}.webm`
        } else {
          // If audio, prefer pristine 16kHz WAV encoded from PCM buffer
          if (pcmDataRef.current.length > 0) {
            const totalSamples = pcmDataRef.current.reduce((acc, c) => acc + c.length, 0)
            const merged = new Float32Array(totalSamples)
            let offset = 0
            for (const chunk of pcmDataRef.current) {
              merged.set(chunk, offset)
              offset += chunk.length
            }
            const downsampled = downsampleBuffer(merged, audioCtx.sampleRate, 16000)
            finalBlob = encodeWAV(downsampled, 16000)
            filename = `session_mic_${Date.now()}.wav`
          } else {
            finalBlob = new Blob(chunks, { type: 'audio/webm' })
            filename = `session_mic_${Date.now()}.webm`
          }
        }

        const url = URL.createObjectURL(finalBlob)
        setRecordedMedia({ blob: finalBlob, url, kind, filename })
        setStatus('finished')
      }

      mediaRecorder.start(250) // collect chunks every 250ms

      // Start Speech Recognition if enabled
      if ((mode === 'dictation' || showTranscript) && hasSpeechApi) {
        const recognition = initSpeechRecognition()
        if (recognition) {
          recognitionRef.current = recognition
          try {
            recognition.start()
          } catch (err) {
            console.warn('Speech recognition start failed:', err)
          }
        }
      }

      setStatus('recording')
      drawWaveform()

      // Start duration timer
      timerRef.current = setInterval(() => {
        setDuration((prev) => prev + 1)
      }, 1000)
    } catch (err) {
      console.error('Media stream error:', err)
      setError(`Microphone/camera access error: ${err.message || 'Permission denied.'}`)
      setStatus('idle')
    }
  }

  // Stop recording
  const stopRecording = () => {
    if (timerRef.current) clearInterval(timerRef.current)
    if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current)

    if (recognitionRef.current) {
      try {
        recognitionRef.current.stop()
      } catch {}
    }

    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop()
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }

    if (previewStreamRef.current) {
      previewStreamRef.current.getTracks().forEach((track) => track.stop())
      previewStreamRef.current = null
      setPreviewingVideo(false)
    }

    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close().catch(() => {})
    }

    setInterimText('')
  }

  // Reset recorder
  const resetRecorder = () => {
    stopRecording()
    if (previewStreamRef.current) {
      previewStreamRef.current.getTracks().forEach((track) => track.stop())
      previewStreamRef.current = null
      setPreviewingVideo(false)
    }
    setStatus('idle')
    setDuration(0)
    setRecordedMedia(null)
    setUploadedInfo(null)
    setError(null)
  }

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopRecording()
      if (previewStreamRef.current) {
        previewStreamRef.current.getTracks().forEach((track) => track.stop())
      }
    }
  }, [])

  // Upload recorded media to backend
  const uploadToBackend = async () => {
    if (!recordedMedia?.blob) return
    setUploading(true)
    setError(null)

    try {
      const file = new File([recordedMedia.blob], recordedMedia.filename, {
        type: recordedMedia.blob.type,
      })
      const info = await api.uploadMedia(file)
      setUploadedInfo(info)

      if (onComplete) {
        onComplete({
          transcript,
          audioPath: info.kind === 'audio' ? info.path : undefined,
          videoPath: info.kind === 'video' ? info.path : undefined,
          mediaInfo: info,
          audioBlob: info.kind === 'audio' ? recordedMedia.blob : undefined,
          videoBlob: info.kind === 'video' ? recordedMedia.blob : undefined,
        })
      }
    } catch (err) {
      setError(`Upload failed: ${err.message}`)
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className={`live-recorder-card panel ${standalone ? 'standalone' : ''}`}>
      <div className="topline" style={{ background: status === 'recording' ? 'var(--regress)' : 'var(--gain)' }} />

      <div className="panel-head" style={{ marginBottom: 12 }}>
        <div>
          <div className="eyebrow" style={{ color: status === 'recording' ? 'var(--regress)' : 'var(--gain)' }}>
            <span className={`rec-dot ${status === 'recording' ? 'active' : ''}`} />
            {status === 'recording' ? 'LIVE CAPTURE ACTIVE' : 'IN-BROWSER RECORDING STUDIO'}
          </div>
          <h2 style={{ margin: 0 }}>
            {status === 'recording' ? `Recording: ${formatTime(duration)}` : 'Live Session Recorder'}
          </h2>
        </div>

        {/* Mode Selector */}
        {allowModeSwitch && status === 'idle' && (
          <div className="mode-toggle-group">
            <button
              type="button"
              className={`mode-btn ${mode === 'audio' ? 'active' : ''}`}
              onClick={() => setMode('audio')}
            >
              🎙️ Audio
            </button>
            <button
              type="button"
              className={`mode-btn ${mode === 'video' ? 'active' : ''}`}
              onClick={() => setMode('video')}
            >
              📹 Video & Audio
            </button>
            <button
              type="button"
              className={`mode-btn ${mode === 'dictation' ? 'active' : ''}`}
              onClick={() => setMode('dictation')}
              title={hasSpeechApi ? 'Live speech-to-text dictation' : 'Speech recognition not supported in this browser'}
            >
              🗣️ Speech-to-Text {hasSpeechApi ? '' : '(No STT)'}
            </button>
          </div>
        )}
      </div>

      {error && <div className="notice" style={{ marginBottom: 12 }}>{error}</div>}

      {/* Hardware Device Selection (Microphone & Webcam) */}
      {status === 'idle' && (
        <div className="device-selection-bar">
          <div className="device-select-item">
            <label htmlFor="mic-select">🎙️ MICROPHONE SOURCE</label>
            <select
              id="mic-select"
              value={selectedAudioId}
              onChange={(e) => setSelectedAudioId(e.target.value)}
            >
              {audioDevices.length === 0 && <option value="">Default System Microphone</option>}
              {audioDevices.map((d, i) => (
                <option key={d.deviceId || i} value={d.deviceId}>
                  {d.label || `Microphone ${i + 1}`}
                </option>
              ))}
            </select>
          </div>

          {mode === 'video' && (
            <div className="device-select-item">
              <label htmlFor="cam-select">📹 WEBCAM SOURCE</label>
              <div className="row" style={{ gap: 6, alignItems: 'center' }}>
                <select
                  id="cam-select"
                  value={selectedVideoId}
                  onChange={(e) => setSelectedVideoId(e.target.value)}
                  style={{ flex: 1 }}
                >
                  {videoDevices.length === 0 && <option value="">Default System Webcam</option>}
                  {videoDevices.map((d, i) => (
                    <option key={d.deviceId || i} value={d.deviceId}>
                      {d.label || `Camera ${i + 1}`}
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  className={`btn ghost small ${previewingVideo ? 'active' : ''}`}
                  onClick={toggleVideoPreview}
                  style={{ whiteSpace: 'nowrap', padding: '7px 11px', fontSize: 12 }}
                >
                  {previewingVideo ? 'Hide Camera' : 'Preview Cam'}
                </button>
              </div>
            </div>
          )}

          <div style={{ alignSelf: 'flex-end' }}>
            <button
              type="button"
              className="btn ghost small"
              onClick={() => refreshDevices(true)}
              style={{ whiteSpace: 'nowrap', padding: '7px 11px', fontSize: 12 }}
              title="Request browser permission to display exact device names"
            >
              ↻ Detect Devices
            </button>
          </div>
        </div>
      )}

      {/* Main Studio Viewport */}
      <div className="recorder-viewport">
        {/* Video Mode Viewfinder (Recording OR Live Camera Preview) */}
        {mode === 'video' && (status === 'recording' || previewingVideo) && (
          <div className="video-viewfinder-wrapper">
            <video ref={videoPreviewRef} playsInline muted className="video-viewfinder" />
            <div className="framing-overlay">
              <div className="framing-crosshairs" />
              <div className="framing-guide-text">
                {status === 'recording'
                  ? 'Center face and upper torso for posture tracking'
                  : 'Live Camera Preview — check framing & lighting'}
              </div>
            </div>
          </div>
        )}

        {/* Real-time Oscilloscope Waveform Canvas */}
        <div className="canvas-wrapper">
          <canvas ref={canvasRef} width={640} height={110} className="waveform-canvas" />
          <div className="waveform-meta">
            <div className="timer-display mono">
              <span className={`pulse-indicator ${status === 'recording' ? 'pulsing' : ''}`} />
              {formatTime(duration)}
            </div>
            <div className="mic-meter-wrapper" title={`Input level: ${micLevel}%`}>
              <span className="mono small muted">LEVEL</span>
              <div className="mic-meter-bar">
                <div
                  className="mic-meter-fill"
                  style={{
                    width: `${micLevel}%`,
                    background: micLevel > 80 ? 'var(--mark)' : 'var(--gain)',
                  }}
                />
              </div>
            </div>
          </div>
        </div>

        {/* Live Controls */}
        <div className="recorder-controls-row">
          {status === 'idle' && (
            <button type="button" className="btn btn-record" onClick={startRecording}>
              <span className="rec-circle" />
              Start Recording
            </button>
          )}

          {status === 'recording' && (
            <div className="row" style={{ alignItems: 'center', width: '100%', justifyContent: 'space-between' }}>
              <div className="speaker-switcher">
                <span className="small mono muted" style={{ marginRight: 6 }}>ACTIVE SPEAKER:</span>
                <button
                  type="button"
                  className={`speaker-pill ${speaker === 'THERAPIST' ? 'active therapist' : ''}`}
                  onClick={() => setSpeaker('THERAPIST')}
                >
                  ● Therapist
                </button>
                <button
                  type="button"
                  className={`speaker-pill ${speaker === 'CLIENT' ? 'active client' : ''}`}
                  onClick={() => setSpeaker('CLIENT')}
                >
                  ● Client
                </button>
                <span className="key-hint" title="Press Tab to toggle speaker">[Tab to switch]</span>
              </div>

              <button type="button" className="btn btn-stop" onClick={stopRecording}>
                ⏹ Stop & Review
              </button>
            </div>
          )}

          {status === 'finished' && (
            <div className="row" style={{ width: '100%', justifyContent: 'space-between', alignItems: 'center' }}>
              <div className="row" style={{ gap: 8 }}>
                <button
                  type="button"
                  className="btn"
                  onClick={uploadToBackend}
                  disabled={uploading}
                >
                  {uploading ? 'Analyzing Media…' : '✓ Use Recording in Session'}
                </button>
                <button type="button" className="btn ghost" onClick={resetRecorder}>
                  ↻ Record Again
                </button>
              </div>

              {recordedMedia && (
                <a
                  href={recordedMedia.url}
                  download={recordedMedia.filename}
                  className="btn ghost small"
                  title="Save local backup copy"
                >
                  💾 Download {recordedMedia.kind === 'video' ? '.webm' : '.wav'}
                </a>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Finished Preview Player */}
      {status === 'finished' && recordedMedia && (
        <div className="recorded-preview-panel" style={{ marginTop: 14 }}>
          <div className="eyebrow">Recording Preview ({recordedMedia.filename})</div>
          {recordedMedia.kind === 'video' ? (
            <video
              ref={playbackVideoRef}
              src={recordedMedia.url}
              controls
              style={{ width: '100%', maxHeight: 240, borderRadius: 3, background: '#000' }}
            />
          ) : (
            <audio src={recordedMedia.url} controls style={{ width: '100%', height: 40 }} />
          )}
          {uploadedInfo && (
            <div className="notice plain" style={{ marginTop: 8, padding: '6px 12px' }}>
              ✓ Saved to server temp storage: <code className="mono">{uploadedInfo.path}</code> ({uploadedInfo.size_mb} MB)
            </div>
          )}
        </div>
      )}

      {/* Live Transcript / Speech-to-Text Box */}
      {showTranscript && (
        <div className="transcript-box-section" style={{ marginTop: 16 }}>
          <div className="row" style={{ justifyContent: 'space-between', marginBottom: 4 }}>
            <label htmlFor="rec-transcript" style={{ margin: 0 }}>
              Live Transcript {status === 'recording' && hasSpeechApi && '· Streaming Speech-to-Text'}
            </label>
            {!hasSpeechApi && (
              <span className="mono small muted" style={{ fontSize: 11 }}>
                Dictation API active in Chrome & Edge
              </span>
            )}
          </div>

          <div className="transcript-editor-wrapper">
            <textarea
              id="rec-transcript"
              rows={status === 'recording' ? 6 : 5}
              value={transcript}
              placeholder="Spoken words will format here into THERAPIST: and CLIENT: turns, or you can type directly..."
              onChange={(e) => setTranscript(e.target.value)}
            />
            {interimText && status === 'recording' && (
              <div className="interim-text-bubble">
                <span className="who">{speaker}:</span> {interimText}
              </div>
            )}
          </div>
          <p className="small muted" style={{ marginTop: 4, marginBottom: 0 }}>
            Speaker turns format automatically with <code className="mono">THERAPIST:</code> and <code className="mono">CLIENT:</code> labels ready for the scoring engine.
          </p>
        </div>
      )}
    </div>
  )
}
