/**
 * M26 — what this instrument can and cannot resolve.
 *
 * Every session produces a change-talk ratio. Whether that number can be read
 * is a different question, and it depends on how many directional turns the
 * session contained. At the lengths this corpus actually contains, the 80%
 * credible interval spans roughly two-thirds of the scale, so the midpoint
 * carries almost nothing about where the client sits.
 *
 * The old interface drew those points the same as any other. A reader with a
 * number in front of them will use it, whiskers or not — so this panel states
 * the verdict in words instead of leaving it to be inferred from the width of
 * an error bar.
 *
 * The numbers come from ml/artifacts/resolvability.json, measured by running
 * simulated sessions through the estimator that actually ships. They are not a
 * rule of thumb and not a fitted curve, and the panel says which reading the
 * evidence supports rather than implying the system has no opinion.
 */
export default function Resolvability({ summary, points = [] }) {
  if (!summary || !summary.n_sessions) return null

  const withTurns = points.filter((p) => (p.directional_turns || 0) > 0)
  if (!withTurns.length) return null

  const readable = withTurns.filter((p) => p.resolvability?.readable)
  const example = withTurns.find((p) => p.resolvability?.available && !p.resolvability?.readable)
  const need = example?.resolvability?.turns_for_readable_estimate
  const turns = withTurns.map((p) => p.directional_turns)
  const median = [...turns].sort((a, b) => a - b)[Math.floor(turns.length / 2)]
  const allShort = readable.length === 0

  return (
    <section className="panel resolvability" aria-labelledby="resolv-h">
      <header className="resolvability-head">
        <h2 id="resolv-h">What this can resolve</h2>
        <span className={`resolv-pill ${allShort ? 'is-trend' : 'is-single'}`}>
          {allShort ? 'Read the trend' : 'Single sessions readable'}
        </span>
      </header>

      <div className="resolv-stats">
        <div className="resolv-stat">
          <span className="k">Median turns / session</span>
          <span className="v">{median}</span>
        </div>
        <div className="resolv-stat">
          <span className="k">Needed to read one session</span>
          <span className="v">{need ?? '—'}</span>
        </div>
        <div className="resolv-stat">
          <span className="k">Sessions readable alone</span>
          <span className="v">
            {readable.length}<span className="of"> / {withTurns.length}</span>
          </span>
        </div>
      </div>

      <p className="resolv-verdict">
        {allShort ? (
          <>
            No session here is long enough for its ratio to be read as a single
            value. The direction across sessions is the reading this data
            supports — a point on the panel above is an estimate with an
            interval, not a measurement.
          </>
        ) : (
          <>
            {readable.length} of {withTurns.length} sessions are long enough to
            read on their own. The rest should be read as part of the trend.
          </>
        )}
      </p>

      <details className="resolv-more">
        <summary>How this was measured</summary>
        <p>
          Sessions at known true ratios were simulated and run through the same
          estimator the app uses. At {example?.directional_turns ?? median} directional
          turns the 80% credible interval is about{' '}
          {example?.resolvability?.ci80_width?.toFixed(2) ?? '—'} wide on a 0–1
          scale; a reading narrow enough to quote needs about {need ?? '—'} turns.
        </p>
        <p className="resolv-caveat">
          The simulation assumes turns are independent and that the classifier
          behaves on this session as it did on held-out data. Both are
          optimistic, so real sessions resolve no better than this.
        </p>
      </details>
    </section>
  )
}
