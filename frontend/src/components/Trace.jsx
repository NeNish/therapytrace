import { useMemo, useState } from 'react'

/**
 * The Trace.
 *
 * The one element this product is meant to be remembered by. It draws the
 * client's Therapeutic Progress Index as an instrument readout: a corridor
 * band showing the client's own baseline +/- 1 SD, the trace itself, a tick
 * for every session, and an amber rule wherever a regime change was detected.
 *
 * Reading it: inside the band = talking the way they always have. Above it =
 * language has moved beyond their own norm. That is the whole idea of the
 * index expressed as one shape.
 */
export default function Trace({
  series = [],
  changePoint = null,
  measures = [],
  onSelect,
  selected = null,
  height = 300,
}) {
  const [hover, setHover] = useState(null)

  const W = 900
  const H = height
  const pad = { top: 26, right: 26, bottom: 40, left: 46 }
  const innerW = W - pad.left - pad.right
  const innerH = H - pad.top - pad.bottom

  const yMin = 25
  const yMax = 80

  const x = (i) =>
    pad.left + (series.length <= 1 ? innerW / 2 : (i * innerW) / (series.length - 1))
  const y = (v) =>
    pad.top + innerH - ((Math.max(yMin, Math.min(yMax, v)) - yMin) / (yMax - yMin)) * innerH

  const path = useMemo(
    () => series.map((v, i) => `${i === 0 ? 'M' : 'L'} ${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join(' '),
    [series, height]
  )

  // PHQ-9 style measures, inverted so "better" points the same way as the trace
  const measurePath = useMemo(() => {
    if (!measures.length) return null
    const vals = measures.map((m) => m.score)
    const lo = Math.min(...vals)
    const hi = Math.max(...vals)
    if (hi === lo) return null
    return measures
      .map((m, i) => {
        const norm = 1 - (m.score - lo) / (hi - lo)
        const yy = pad.top + innerH - norm * innerH
        const xx = x(Math.min(series.length - 1, m.session_number - 1))
        return `${i === 0 ? 'M' : 'L'} ${xx.toFixed(1)} ${yy.toFixed(1)}`
      })
      .join(' ')
  }, [measures, series.length, height])

  if (!series.length) {
    return (
      <div className="empty">
        <h3>No trace yet</h3>
        <p className="small muted">Add a session transcript and the trace starts drawing.</p>
      </div>
    )
  }

  const active = hover ?? selected

  return (
    <div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        width="100%"
        role="img"
        aria-label="Therapeutic Progress Index across sessions"
        style={{ display: 'block' }}
        onMouseLeave={() => setHover(null)}
      >
        {/* baseline corridor: the client's own norm, +/- 1 SD */}
        <rect
          x={pad.left}
          y={y(60)}
          width={innerW}
          height={y(40) - y(60)}
          fill="var(--rule-soft)"
          opacity="0.75"
        />
        <line
          x1={pad.left} x2={pad.left + innerW} y1={y(50)} y2={y(50)}
          stroke="var(--ink-3)" strokeWidth="1" strokeDasharray="1 4"
        />
        <text x={pad.left + 4} y={y(60) - 5} fontFamily="var(--data)" fontSize="9"
              fill="var(--ink-3)" letterSpacing="1.4">
          OWN BASELINE ±1 SD
        </text>

        {/* y axis */}
        {[30, 40, 50, 60, 70, 80].map((v) => (
          <g key={v}>
            <line x1={pad.left - 4} x2={pad.left} y1={y(v)} y2={y(v)} stroke="var(--rule)" />
            <text x={pad.left - 8} y={y(v) + 3} textAnchor="end" fontFamily="var(--data)"
                  fontSize="9" fill="var(--ink-3)">{v}</text>
          </g>
        ))}

        {/* change point */}
        {changePoint && (
          <g>
            <line
              x1={x(changePoint.index)}
              x2={x(changePoint.index)}
              y1={pad.top - 8}
              y2={pad.top + innerH}
              stroke="var(--mark)"
              strokeWidth="1.5"
              strokeDasharray="4 3"
            />
            <rect
              x={x(changePoint.index) + 3} y={pad.top - 18}
              width={128} height={15} fill="var(--mark-wash)" stroke="var(--mark)"
            />
            <text
              x={x(changePoint.index) + 8} y={pad.top - 7}
              fontFamily="var(--data)" fontSize="9" fill="var(--mark)" letterSpacing="0.6"
            >
              SHIFT {changePoint.shift > 0 ? '+' : ''}{changePoint.shift} PTS
            </text>
          </g>
        )}

        {/* self-report measure, inverted */}
        {measurePath && (
          <path d={measurePath} fill="none" stroke="var(--ink-3)" strokeWidth="1.25"
                strokeDasharray="5 4" opacity="0.8" />
        )}

        {/* the trace */}
        <path d={path} fill="none" stroke="var(--ink)" strokeWidth="2.25"
              strokeLinejoin="round" strokeLinecap="round" />

        {/* session ticks */}
        {series.map((v, i) => {
          const isActive = active === i
          const above = v >= 60
          const below = v <= 40
          const fill = above ? 'var(--gain)' : below ? 'var(--regress)' : 'var(--panel)'
          return (
            <g
              key={i}
              onMouseEnter={() => setHover(i)}
              onClick={() => onSelect && onSelect(i)}
              style={{ cursor: onSelect ? 'pointer' : 'default' }}
            >
              <rect x={x(i) - 14} y={pad.top} width={28} height={innerH} fill="transparent" />
              <line x1={x(i)} x2={x(i)} y1={pad.top + innerH} y2={pad.top + innerH + 5}
                    stroke="var(--rule)" />
              <rect
                x={x(i) - (isActive ? 5 : 3.5)}
                y={y(v) - (isActive ? 5 : 3.5)}
                width={isActive ? 10 : 7}
                height={isActive ? 10 : 7}
                fill={fill}
                stroke="var(--ink)"
                strokeWidth="1.5"
              />
              <text x={x(i)} y={pad.top + innerH + 18} textAnchor="middle"
                    fontFamily="var(--data)" fontSize="9"
                    fill={isActive ? 'var(--ink)' : 'var(--ink-3)'}>
                {i + 1}
              </text>
              {isActive && (
                <text x={x(i)} y={y(v) - 12} textAnchor="middle" fontFamily="var(--data)"
                      fontSize="11" fontWeight="600" fill="var(--ink)">
                  {v.toFixed(1)}
                </text>
              )}
            </g>
          )
        })}

        <text x={pad.left} y={H - 8} fontFamily="var(--data)" fontSize="9"
              fill="var(--ink-3)" letterSpacing="1.4">
          SESSION
        </text>
        {measurePath && (
          <text x={pad.left + innerW} y={H - 8} textAnchor="end" fontFamily="var(--data)"
                fontSize="9" fill="var(--ink-3)" letterSpacing="1">
            DASHED — SELF-REPORT, INVERTED
          </text>
        )}
      </svg>
    </div>
  )
}
