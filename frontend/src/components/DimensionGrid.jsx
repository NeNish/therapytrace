import { DIMENSION_LABELS } from '../lib/api'

function Spark({ values, baseline }) {
  const W = 220
  const H = 54
  const pad = 6
  if (!values.length) return null
  const lo = Math.min(...values, baseline ?? Infinity) - 0.03
  const hi = Math.max(...values, baseline ?? -Infinity) + 0.03
  const span = hi - lo || 1
  const x = (i) => pad + (i * (W - 2 * pad)) / Math.max(1, values.length - 1)
  const y = (v) => H - pad - ((v - lo) / span) * (H - 2 * pad)
  const d = values.map((v, i) => `${i === 0 ? 'M' : 'L'} ${x(i)} ${y(v)}`).join(' ')
  const last = values[values.length - 1]
  const dir = baseline == null ? 0 : last - baseline

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} aria-hidden="true">
      {baseline != null && (
        <line x1={pad} x2={W - pad} y1={y(baseline)} y2={y(baseline)}
              stroke="var(--ink-3)" strokeDasharray="2 3" strokeWidth="1" />
      )}
      <path d={d} fill="none" strokeWidth="1.75"
            stroke={dir > 0.01 ? 'var(--gain)' : dir < -0.01 ? 'var(--regress)' : 'var(--hold)'} />
      <rect x={x(values.length - 1) - 2.5} y={y(last) - 2.5} width="5" height="5"
            fill={dir > 0.01 ? 'var(--gain)' : dir < -0.01 ? 'var(--regress)' : 'var(--hold)'} />
    </svg>
  )
}

export default function DimensionGrid({ dimensionSeries = {}, baseline = null }) {
  const keys = Object.keys(dimensionSeries)
  if (!keys.length) return null

  return (
    <div className="grid-3">
      {keys.map((k) => {
        const values = dimensionSeries[k] || []
        const base = baseline?.means?.[k] ?? null
        const last = values[values.length - 1]
        const delta = base == null ? null : last - base
        return (
          <div key={k} className="panel" style={{ marginBottom: 0 }}>
            <div className="eyebrow">{DIMENSION_LABELS[k] || k}</div>
            <Spark values={values} baseline={base} />
            <div className="small mono" style={{ marginTop: 4 }}>
              {last?.toFixed(3)}
              {delta != null && (
                <span
                  style={{
                    marginLeft: 8,
                    color: delta > 0 ? 'var(--gain)' : delta < 0 ? 'var(--regress)' : 'var(--hold)',
                  }}
                >
                  {delta > 0 ? '+' : ''}{delta.toFixed(3)} vs baseline
                </span>
              )}
            </div>
          </div>
        )
      })}
    </div>
  )
}
