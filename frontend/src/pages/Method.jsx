export default function Method() {
  return (
    <>
      <div className="eyebrow">Method</div>
      <h1>How the index works</h1>
      <p className="lede">
        TherapyTrace does not try to work out what is wrong with anyone. It
        measures whether a person is talking differently than they used to, on
        five dimensions that therapy is supposed to move.
      </p>

      <div className="panel">
        <h2>The five dimensions</h2>
        <table>
          <thead>
            <tr><th>Dimension</th><th>The question it asks</th><th>What movement looks like</th></tr>
          </thead>
          <tbody>
            <tr>
              <td>Self-agency</td>
              <td>Does the client speak as someone who acts, or as someone acted upon?</td>
              <td>"it just happens to me" → "I decided to"</td>
            </tr>
            <tr>
              <td>Future orientation</td>
              <td>Is attention on what comes next, or looping on what already happened?</td>
              <td>replaying the argument → naming a plan for Tuesday</td>
            </tr>
            <tr>
              <td>Emotional granularity</td>
              <td>Is affect named precisely, or left as undifferentiated "bad"?</td>
              <td>"I feel bad" → "resentful, and a bit ashamed of that"</td>
            </tr>
            <tr>
              <td>Problem ownership</td>
              <td>Is the difficulty inside the client's reach, or entirely outside it?</td>
              <td>"he always" → "my part is that I go quiet then punish him for it"</td>
            </tr>
            <tr>
              <td>Reflection depth</td>
              <td>Is the client describing events, or working out why they happen?</td>
              <td>a narrated week → a noticed pattern with a cause attached</td>
            </tr>
          </tbody>
        </table>
        <p className="small muted" style={{ marginBottom: 0 }}>
          Ownership is deliberately not the same as self-blame. "It's all my
          fault, I ruin everything" is scored down, not up — collapsing into
          fault is a symptom, not insight.
        </p>
      </div>

      <div className="panel">
        <h2>Within-person, not between-person</h2>
        <p>
          Every dimension is standardised against the client's own first two
          sessions:
        </p>
        <pre className="mono small" style={{ background: '#fff', border: '1px solid var(--rule)', padding: 14 }}>
{`z_d(s) = ( x_d(s) - mu_d ) / sigma_d
TPI(s) = 50 + 10 * sum_d ( w_d * z_d(s) )`}
        </pre>
        <p className="small">
          mu and sigma come from that person's baseline, and sigma is shrunk
          toward a cohort value because two sessions give a terrible variance
          estimate on their own. A TPI of 50 means "talking the way they always
          have". 60 means one standard deviation above their own norm. Nobody is
          ever compared to anybody else.
        </p>
      </div>

      <div className="panel">
        <h2>What it cannot do</h2>
        <ul>
          <li>
            It cannot diagnose. Nothing in the pipeline maps to a disorder, and
            it should never be read as if it did.
          </li>
          <li>
            The therapist-contribution view is association, not causation.
            Therapists pick interventions in response to what was just said, so
            those numbers are confounded by indication by construction.
          </li>
          <li>
            Short sessions produce noisy scores. The confidence figure drops
            below 50% under about 300 client words, and it should be believed.
          </li>
          <li>
            The lexicon tier is English and culturally situated. Directness,
            emotion vocabulary, and how ownership is expressed all vary. Treat
            cross-cultural transfer as an open question, not a solved one.
          </li>
          <li>
            A falling trace is a prompt for a conversation in supervision. It is
            not evidence that a therapist is doing badly or that a client is
            failing.
          </li>
        </ul>
      </div>

      <div className="panel">
        <h2>Validation</h2>
        <p className="small">
          Record PHQ-9, GAD-7 or MADRS scores against sessions and the case page
          will report the within-person correlation with the trace, plus a
          lead-lag test of whether language moves before the questionnaire does.
          An index nobody has checked against an outcome is decoration.
        </p>
      </div>

      <p className="footnote">
        Transcripts are stored under pseudonymous case codes. Do not enter
        names, dates of birth, addresses, or any other identifying detail. Any
        deployment on real clinical material needs ethics approval, informed
        consent covering secondary analysis, and encryption at rest.
      </p>
    </>
  )
}
