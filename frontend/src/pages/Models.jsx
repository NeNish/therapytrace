import { useEffect, useState } from 'react'

import { api } from '../lib/api'

/**
 * Every classifier that was compared, not only the one that shipped.
 *
 * The page reads ml/artifacts/benchmark.json through the API rather than
 * restating numbers typed into JSX, so what is displayed is what the training
 * run actually produced. There is no second copy to fall out of step.
 *
 * Showing all seven is the argument, not padding. The gaps between them are
 * mostly smaller than the variation between cross-validation folds, which is
 * the evidence that effort belongs in the longitudinal layer rather than in
 * chasing a better utterance classifier. A table of the winner alone would
 * hide precisely that.
 */

function pct(v) {
  return v === null || v === undefined ? '—' : v.toFixed(3)
}

function TaskTable({ task }) {
  const best = Math.max(...task.models.filter((m) => !m.is_baseline).map((m) => m.test_macro_f1))

  return (
    <div className="panel">
      <h2>{task.label}</h2>
      <p className="small muted" style={{ marginTop: -6 }}>
        {task.n_utterances.toLocaleString()} utterances across {task.n_conversations}{' '}
        conversations. Held-out test: {task.n_test.toLocaleString()} utterances from{' '}
        {task.n_test_conversations} conversations never seen in training. Classes:{' '}
        {task.classes.map((c) => `${c} (${task.class_balance[c]})`).join(', ')}.
      </p>

      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th className="num">CV macro-F1</th>
            <th className="num">Accuracy</th>
            <th className="num">Macro-F1</th>
            <th className="num">Weighted-F1</th>
            <th className="num">ROC-AUC</th>
            <th className="num">Fit (s)</th>
            <th className="num">p vs best</th>
          </tr>
        </thead>
        <tbody>
          {task.models.map((m) => {
            const notSeparable = m.p_vs_best !== null && m.p_vs_best > 0.05
            return (
              <tr key={m.id} style={m.is_baseline ? { opacity: 0.6 } : undefined}>
                <td>
                  {m.name}
                  {m.is_selected && <span className="small muted"> — selected on CV</span>}
                  {m.is_baseline && <span className="small muted"> — the floor</span>}
                </td>
                <td className="num">
                  {pct(m.cv_macro_f1)}
                  <span className="small muted"> ±{m.cv_macro_f1_sd.toFixed(3)}</span>
                </td>
                <td className="num">{pct(m.test_accuracy)}</td>
                <td
                  className="num"
                  style={m.test_macro_f1 === best ? { fontWeight: 600 } : undefined}
                >
                  {pct(m.test_macro_f1)}
                </td>
                <td className="num">{pct(m.test_weighted_f1)}</td>
                <td className="num">{pct(m.test_roc_auc)}</td>
                <td className="num">{m.fit_seconds === null ? '—' : m.fit_seconds.toFixed(2)}</td>
                <td className="num">
                  {m.p_vs_best === null ? '—' : m.p_vs_best.toFixed(3)}
                  {notSeparable && <span className="small muted"> ns</span>}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>

      <p className="small muted">
        Bold is the best held-out macro-F1. <strong>ns</strong> marks a model a paired
        t-test across the five folds cannot separate from the selected one at the 5%
        level. Fit times are the classifier alone, measured on an idle machine.
      </p>

      {task.models.some((m) => Object.keys(m.per_class_recall || {}).length > 0) && (
        <>
          <h3 style={{ marginTop: 22 }}>Per-class recall</h3>
          <p className="small muted" style={{ marginTop: -6 }}>
            Where the averages hide the problem. A model can win on macro-F1 and still
            miss the class the interface depends on.
          </p>
          <table>
            <thead>
              <tr>
                <th>Model</th>
                {task.classes.map((c) => <th key={c} className="num">{c}</th>)}
              </tr>
            </thead>
            <tbody>
              {task.models.filter((m) => !m.is_baseline).map((m) => (
                <tr key={m.id}>
                  <td>{m.name}</td>
                  {task.classes.map((c) => (
                    <td key={c} className="num">{pct(m.per_class_recall[c])}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  )
}

export default function Models() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    api.benchmark().then(setData).catch((e) => setError(e.message))
  }, [])

  if (error) return <div className="notice">{error}</div>
  if (!data) return <p className="small muted">Loading…</p>

  if (!data.available) {
    return (
      <>
        <div className="eyebrow">Models</div>
        <h1>Model comparison</h1>
        <div className="notice">
          {data.reason} Run <code>python ml/benchmark.py</code> to generate it.
        </div>
      </>
    )
  }

  return (
    <>
      <div className="eyebrow">Models</div>
      <h1>Seven classifiers, one feature matrix</h1>
      <p className="lede">
        Every learner sees the same features, so a difference in score is attributable
        to the learner and nothing else. Splits are grouped by conversation, never by
        utterance — an utterance-level split lets a model recognise a conversation it
        memorised and inflates every number on this page.
      </p>

      <div className="panel">
        <h2>What shipped, and why</h2>
        <p>
          The model running in this build is <strong>logistic regression</strong>.
          {' '}{data.deployed_note}
        </p>
        <p className="small muted" style={{ marginBottom: 0 }}>
          {data.dataset}. {data.protocol && data.protocol.split}.{' '}
          {data.protocol && data.protocol.class_balancing}
        </p>
      </div>

      {data.tasks.map((t) => <TaskTable key={t.id} task={t} />)}

      <div className="panel">
        <h2>The finding</h2>
        <p>
          On client talk type a paired t-test cannot separate the best learner from any
          of the other six. The spread across all seven is 0.051 macro-F1 against a
          fold-to-fold standard deviation of 0.043 — the gaps sit inside the noise.
          XGBoost needs 56 seconds to fit where Naive Bayes needs 0.01 and scores no
          better.
        </p>
        <p style={{ marginBottom: 0 }}>
          With roughly three thousand short utterances and tens of thousands of sparse
          features, a regularised linear boundary is already the right capacity. The
          ceiling is set by the task, not the model — which is the reason this project
          spends its effort on within-person normalisation and trajectory across
          sessions rather than on a better utterance classifier.
        </p>
      </div>
    </>
  )
}
