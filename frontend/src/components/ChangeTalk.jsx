import { useState } from 'react'

/**
 * Change-talk ratio across sessions.
 *
 * Sits under the Trace and shares its x-axis — one tick per session, same
 * order, same spacing — so the two can be read together. Deliberately NOT a
 * second line on the Trace itself: the TPI is a standardised score centred on
 * 50 and this is a proportion between 0 and 1, and putting two scales on one
 * chart invites people to read meaning into where the lines cross, which would
 * be an artefact of the axes rather than anything about the client.
 *
 * What it shows is the proportion of the client's directional turns that the
 * AnnoMI-trained classifier called change talk, after correcting for the
 * classifier's known bias. It is independent of the TPI: the TPI comes from the
 * lexicon, this comes from a model trained on expert annotations. Where the two
 * agree, that agreement is evidence rather than repetition.
 *
 * Three honesty rules are enforced here rather than left to the caller:
 *   - sessions with no directional turns break the line instead of reading zero
 *   - points whose credible interval is too wide to read as a point estimate
 *     are drawn hollow, and every point carries its interval as a vertical whisker
 *   - the number of turns behind each point is available on hover, since a
 *     ratio from four turns and one from forty should not look alike
 */
export default function ChangeTalk({ points = [], selected = null, onSelect, height = 170 }) {
  const [hover, setHover] = useState(null)

  const usable = points.filter((p) => p.ratio !== null && p.ratio !== undefined)
  if (usable.length < 2) return null

  const W = 900
  const H = height
  const pad = { top: 24, right: 26, bottom: 30, left: 46 }
  const innerW = W - pad.left - pad.right
  const innerH = H - pad.top - pad.bottom

  const n = points.length
  const x = (i) => pad.left + (n <= 1 ? innerW / 2 : (i * innerW) / (n - 1))
  const y = (v) => pad.top + innerH - v * innerH

  // Break the path wherever a session has no estimate, rather than joining
  // across the gap and inventing a trend through it.
  let path = ''
  let penDown = false
  points.forEach((p, i) => {
    if (p.ratio === null || p.ratio === undefined) { penDown = false; return }
    path += `${penDown ? 'L' : 'M'} ${x(i).toFixed(1)} ${y(p.ratio).toFixed(1)} `
    penDown = true
  })

  const active = hover ?? selected
  const activePoint = active !== null && active !== undefined ? points[active] : null

  return (
    <div style={{ marginTop: 14 }}>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        width="100%"
        role="img"
        aria-label="Change-talk ratio across sessions, corrected for classifier bias"
        style={{ display: 'block' }}
        onMouseLeave={() => setHover(null)}
      >
        <line
          x1={pad.left} x2={pad.left + innerW} y1={y(0.5)} y2={y(0.5)}
          stroke="var(--ink-3)" strokeWidth="1" strokeDasharray="1 4"
        />
        {[0, 0.5, 1].map((v) => (
          <g key={v}>
            <line x1={pad.left - 4} x2={pad.left} y1={y(v)} y2={y(v)} stroke="var(--rule)" />
            <text x={pad.left - 8} y={y(v) + 3} textAnchor="end" fontFamily="var(--data)"
                  fontSize="9" fill="var(--ink-3)">{v.toFixed(1)}</text>
          </g>
        ))}

        <text x={pad.left} y={14} fontFamily="var(--data)" fontSize="9"
              fill="var(--ink-3)" letterSpacing="1.4">
          CHANGE TALK — SHARE OF DIRECTIONAL TURNS, BIAS-CORRECTED
        </text>

        <path d={path.trim()} fill="none" stroke="var(--mark)" strokeWidth="2"
              strokeLinejoin="round" strokeLinecap="round" />

        {points.map((p, i) => {
          if (p.ratio === null || p.ratio === undefined) return null
          const isActive = active === i
          const r = isActive ? 5 : 3.5
          return (
            <g
              key={i}
              onMouseEnter={() => setHover(i)}
              onClick={() => onSelect && onSelect(i)}
              style={{ cursor: onSelect ? 'pointer' : 'default' }}
            >
              <rect x={x(i) - 14} y={pad.top} width={28} height={innerH} fill="transparent" />
              {p.lo !== null && p.lo !== undefined && p.hi !== null && p.hi !== undefined && (
                <line x1={x(i)} x2={x(i)} y1={y(p.hi)} y2={y(p.lo)}
                      stroke="var(--mark)" strokeWidth="1.25" opacity="0.45" />
              )}
              <circle
                cx={x(i)} cy={y(p.ratio)} r={r}
                /* hollow = interval too wide to read as a point estimate */
                fill={p.wide ? 'var(--panel)' : 'var(--mark)'}
                stroke="var(--mark)" strokeWidth="1.75"
              />
              <text x={x(i)} y={H - 10} textAnchor="middle" fontFamily="var(--data)"
                    fontSize="9" fill={isActive ? 'var(--ink)' : 'var(--ink-3)'}>
                {i + 1}
              </text>
            </g>
          )
        })}
      </svg>

      <p className="small muted" style={{ margin: '6px 0 0' }}>
        {activePoint && activePoint.ratio !== null && activePoint.ratio !== undefined ? (
          <>
            Session {activePoint.session_number}: <strong>{activePoint.ratio.toFixed(2)}</strong>{' '}
            from {activePoint.directional_turns} directional turn
            {activePoint.directional_turns === 1 ? '' : 's'}
            {activePoint.ratio_raw !== null && activePoint.ratio_raw !== undefined
              ? ` (uncorrected ${activePoint.ratio_raw.toFixed(2)})`
              : ''}
            {activePoint.lo !== null && activePoint.lo !== undefined
              ? ` — 80% interval ${activePoint.lo.toFixed(2)} to ${activePoint.hi.toFixed(2)}.`
              : '.'}
          </>
        ) : (
          <>
            Independent of the TPI: this comes from a classifier trained on expert
            annotations, the TPI from the lexicon. Whiskers are 80% credible intervals; hollow points are
            too uncertain to read as a single number. Read the direction over several sessions, not a single step.
          </>
        )}
      </p>
    </div>
  )
}
