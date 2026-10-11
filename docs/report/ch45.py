"""Chapters 4 and 5: System Design and Implementation."""

from rlib import bullets, chapter, figure, numbered, para, rich, section, subsection, table

F = "fig/"


def ch4(doc):
    chapter(doc, 4, "System Design")

    section(doc, "4.1", "Introduction")
    para(doc,
         "The system is organised as a layered pipeline. A transcript enters "
         "at the top, is parsed into speaker turns, is scored against the "
         "client’s own history, and leaves as a trajectory with supporting "
         "evidence. This chapter presents the architecture, the module "
         "decomposition, the behavioural views, the data model, and the "
         "scoring methodology that is the technical core of the work.")

    section(doc, "4.2", "System Architecture")
    para(doc,
         "Figure 4.1 shows the architecture. Three properties of the design "
         "are worth drawing attention to. First, the measurement layer is "
         "rule-based and deterministic, so a clinician can always be shown the "
         "exact phrase that moved a score. Second, the supervised layer is "
         "additive: the system functions without it, which matters because a "
         "fresh installation has no trained artifacts. Third, the optional "
         "modalities are bounded in their influence, so a noisy video cannot "
         "dominate a text-derived score.")
    figure(doc, F + "TherapyTrace_Architecture.png", "4.1",
           "System architecture. Layers run from ingestion through measurement "
           "and scoring to presentation; optional modalities enter as bounded "
           "modifiers rather than as primary signal.", width_cm=14.0)

    section(doc, "4.3", "Module Description")
    para(doc,
         "The system comprises twenty-five modules. Table 4.1 lists the "
         "principal ones with their responsibilities.")
    table(doc, "4.1", "Principal system modules",
          ["Module", "Responsibility"],
          [
              ["M1 Transcript parser", "Detect speaker labels, strip timestamps, order turns, infer roles when labels are absent"],
              ["M3 Feature extractor", "Score five process dimensions and five auxiliaries per client turn"],
              ["M4 Within-person scorer", "Build the personal baseline with shrinkage; compute the Therapeutic Progress Index"],
              ["M5 Trajectory", "OLS trend, recent-window momentum, binary-segmentation change points"],
              ["M6 Therapist attribution", "Pair each therapist turn with the client turns around it and estimate lift"],
              ["M7 Validation", "Correlate the index with self-report instruments; lead-lag testing"],
              ["M8 Persistence", "Store sessions and cached analyses; recompute a case when its baseline widens"],
              ["M10 Supervised models", "Client talk type and therapist behaviour classifiers trained on AnnoMI"],
              ["M11 Mechanism profile", "Order in which dimensions move; state-conditional response"],
              ["M12 Acoustic tier", "Pitch, pause structure and voice-quality descriptors from audio"],
              ["M19 Body tier", "Posture and movement descriptors from video landmarks; emits no emotion label"],
              ["M20 Narrative", "Plain-language session note with a numeric verification gate"],
              ["M21 Benchmark", "Seven-classifier comparison under conversation-grouped splits"],
              ["M22 Reliability", "Error-propagation measurement and prevalence correction"],
              ["M23 Interface integration", "Corrected change-talk series surfaced in the trajectory view"],
              ["M24 Hybrid tagging", "Behaviour model sets the family; lexicon names the subtype"],
          ],
          widths=[4.3, 10.3])

    section(doc, "4.4", "Behavioural Design")
    subsection(doc, "4.4.1", "Use Case View")
    para(doc,
         "Three actors interact with the system: the therapist, who registers "
         "cases and uploads sessions; the clinical supervisor, who reviews "
         "trajectories and therapist contribution; and the researcher, who "
         "bulk-loads corpora and inspects validation statistics.")
    figure(doc, F + "TherapyTrace_UseCase.png", "4.2",
           "Use case diagram. Therapist contribution is shown to supervisors "
           "as an association, never as a performance rating.", width_cm=14.0)

    subsection(doc, "4.4.2", "Activity View")
    para(doc,
         "Figure 4.3 traces one transcript from ingestion to index. The "
         "branch at the baseline step is the clinically important one: a case "
         "with fewer than two prior sessions has no stable personal reference, "
         "and the system reduces reported confidence rather than presenting a "
         "number as though it were grounded.")
    figure(doc, F + "TherapyTrace_Activity.png", "4.3",
           "Activity diagram for the end-to-end scoring pipeline.", width_cm=12.5)

    subsection(doc, "4.4.3", "Sequence View")
    figure(doc, F + "TherapyTrace_Sequence.png", "4.4",
           "Sequence diagram for adding a session, including the recompute "
           "that rescores the client's whole history.", width_cm=14.5)

    section(doc, "4.5", "Data Model")
    para(doc,
         "Four tables hold the persistent state. The design is deliberately "
         "minimal and pseudonymous: there is no field in which a name, date of "
         "birth or contact detail could be recorded even by accident.")
    figure(doc, F + "TherapyTrace_DataModel.png", "4.5",
           "Entity relationship diagram. Cases are identified only by a "
           "pseudonymous code.", width_cm=13.0)

    section(doc, "4.6", "Scoring Methodology")
    subsection(doc, "4.6.1", "The Five Process Dimensions")
    para(doc,
         "Each client turn is scored on five dimensions chosen because therapy "
         "is expected to move them, and because each is measurable from "
         "surface linguistic evidence.")
    table(doc, "4.2", "The five therapeutic process dimensions",
          ["Dimension", "The question it asks", "Movement looks like"],
          [
              ["Self-agency", "Does the client speak as one who acts, or one acted upon?", "“it just happens to me” → “I decided to”"],
              ["Future orientation", "Is attention on what comes next, or looping on what happened?", "replaying the argument → naming a plan"],
              ["Emotional granularity", "How precisely are feelings named?", "“bad” → “resentful, and a bit relieved”"],
              ["Problem ownership", "Is the difficulty located internally or externally?", "“he always does this” → “I shut down when”"],
              ["Reflection depth", "Is there causal or insight-bearing language?", "description → “I think it is because”"],
          ],
          widths=[3.2, 6.2, 5.2])
    para(doc,
         "Problem ownership deserves a note. Internal attribution is scored as "
         "ownership only when it is not self-blame; a client saying “it is all "
         "my fault, I ruin everything” is not demonstrating ownership but "
         "absolutist self-criticism, and the implementation penalises rather "
         "than rewards it.")

    subsection(doc, "4.6.2", "Within-Person Normalisation")
    para(doc,
         "The central methodological commitment of the system is that each "
         "session is scored against the client’s own earlier sessions. For "
         "dimension d in session s:")
    para(doc, "z_d(s) = ( x_d(s) − μ_d(own) ) / σ_d(own)",
         align="center", italic=True, after=8)
    para(doc,
         "where μ and σ are computed from that client’s earliest sessions. "
         "The index is then a weighted sum centred at fifty:")
    para(doc, "TPI(s) = 50 + 10 × Σ w_d · z_d(s)",
         align="center", italic=True, after=8)
    para(doc,
         "Fifty therefore means “this person talking the way this person "
         "usually talks”, and departures are departures from their own norm.")

    subsection(doc, "4.6.3", "Shrinkage for Small Samples")
    para(doc,
         "A standard deviation estimated from two or three sessions is "
         "unstable, and an unstable denominator produces wild index values. "
         "The implementation therefore shrinks the client’s own standard "
         "deviation toward a cohort prior in the manner of a James–Stein "
         "estimator:")
    para(doc, "σ_pooled = λ · σ_own + (1 − λ) · σ_cohort,   "
              "where λ = n / (n + k₀), k₀ = 3",
         align="center", italic=True, after=8)
    para(doc,
         "With one session the estimate leans almost entirely on the cohort "
         "prior; by eight or ten sessions it is dominated by the client’s own "
         "variability. This is what allows the index to be reported early in a "
         "case without being meaningless, while converging on a genuinely "
         "personal scale as evidence accumulates.")

    subsection(doc, "4.6.4", "Turn Aggregation")
    para(doc,
         "Session-level features aggregate turn-level scores weighted by the "
         "square root of word count. A long narrative turn should count for "
         "more than a two-word acknowledgement, but not forty times more, or a "
         "single monologue would determine the whole session. The square root "
         "produces that compromise. A short turn is also smoothed toward the "
         "midpoint rather than scored at zero, since two words are evidence of "
         "almost nothing.")

    subsection(doc, "4.6.5", "Trajectory Estimation")
    para(doc,
         "Three quantities are computed over the index series: an "
         "ordinary-least-squares trend with a significance test; a "
         "recent-window momentum classification over the last six sessions; "
         "and change points located by binary segmentation. The momentum "
         "window exists because a whole-series slope misclassifies the "
         "clinically important rise-then-relapse shape as steady improvement, "
         "a failure observed during development and corrected by corroborating "
         "the recent window against the full-series trend.")
    para(doc, "", after=0)


def ch5(doc):
    chapter(doc, 5, "Implementation")

    section(doc, "5.1", "Introduction")
    para(doc,
         "The system is implemented as a Python backend serving a React "
         "single-page application, with a separate offline package for model "
         "training and evaluation. This chapter describes the implementation "
         "of each layer and the decisions that shaped it.")

    section(doc, "5.2", "Implementation Environment")
    para(doc,
         "The backend is FastAPI over SQLAlchemy 2.0 with SQLite persistence, "
         "chosen so the system runs from a single command with no database "
         "server. The frontend is React 18 built with Vite. The machine "
         "learning package uses scikit-learn with XGBoost and LightGBM for the "
         "comparison study. The repository holds twenty-five modules, "
         "sixty-nine automated tests and a reproducible training pipeline.")

    section(doc, "5.3", "Measurement Layer")
    subsection(doc, "5.3.1", "Transcript Parsing")
    para(doc,
         "The parser detects speaker labels, strips leading timestamps in "
         "several formats, and orders turns. Where labels are anonymous it "
         "infers roles from question rate and mean turn length, and attaches a "
         "warning to the result so the inference is visible rather than "
         "silent. Timestamp stripping was added after testing revealed that "
         "timestamped transcripts were parsed as zero turns — a failure that "
         "produced no error, only an empty analysis.")

    subsection(doc, "5.3.2", "Feature Extraction")
    para(doc,
         "Features are computed from curated lexicons and syntactic patterns. "
         "Each returns both a score and the evidence phrases that produced it, "
         "so every number in the interface can be traced to specific words in "
         "the transcript. This traceability is a requirement rather than a "
         "convenience: a clinician asked to act on a score must be able to see "
         "what caused it.")

    section(doc, "5.4", "Supervised Layer")
    subsection(doc, "5.4.1", "Training Protocol")
    para(doc,
         "Two classifiers are trained on AnnoMI: client talk type (change, "
         "sustain, neutral) and therapist behaviour (question, reflection, "
         "therapist input, other). The single most consequential decision in "
         "the training code is that splits are grouped by transcript "
         "identifier and never by utterance. Utterances within a conversation "
         "share a client, a therapist, a topic and a speaking style; splitting "
         "at the utterance level allows a model to memorise a conversation and "
         "then recognise it at test time, which inflates every metric.")
    para(doc,
         "Twenty-five per cent of conversations are held out before any model "
         "is fitted. Model selection runs as five-fold grouped "
         "cross-validation over the remaining development conversations, and "
         "the held-out conversations are scored once.")

    subsection(doc, "5.4.2", "Equalised Class Balancing")
    para(doc,
         "The corpus is fifty-three per cent neutral on the client task, so a "
         "learner left to optimise raw accuracy under-predicts the minority "
         "classes and loses macro-F1 for a reason that has nothing to do with "
         "the algorithm. Every model is therefore balanced by whatever "
         "mechanism it offers: class weighting for the linear models and "
         "LightGBM, balanced subsampling for the random forest, uniform priors "
         "for naive Bayes, and injected per-row weights for XGBoost, which "
         "exposes no class weight parameter. The multilayer perceptron cannot "
         "be balanced in scikit-learn, and that asymmetry is reported rather "
         "than concealed.")

    subsection(doc, "5.4.3", "Hybrid Therapist Tagging")
    para(doc,
         "Therapist attribution combines the trained model with the rule-based "
         "taxonomy, assigning each the task it performs well. The model, "
         "trained on thousands of annotated turns, decides the coarse family: "
         "question, reflection, therapist input or other. The regular "
         "expression taxonomy then names the subtype within that family, "
         "distinguishing open from closed questions and simple from complex "
         "reflections, which the model has no notion of. Where no pattern "
         "matches inside the chosen family, a family default is used rather "
         "than a label from elsewhere.")
    para(doc,
         "A worked example shows the gain. The turn “Most people in that "
         "position would do the same. It is a common response to threat.” "
         "matches no pattern and falls through to “other” under the lexicon "
         "alone; the model identifies it as therapist input, and the label "
         "becomes psychoeducation. With no trained artifacts present the "
         "lexicon performs the whole task exactly as before.")

    section(doc, "5.5", "Bias Correction Before Display")
    para(doc,
         "The quantity the interface displays is the session-level change-talk "
         "ratio, change divided by the sum of change and sustain over "
         "directional turns. Counting predicted labels and dividing is a "
         "biased estimator whenever the classifier is imperfect, and the bias "
         "here is large because sustain talk is under-detected. Chapter 6 "
         "quantifies it.")
    para(doc,
         "The correction implemented is a Bayesian posterior over the true "
         "ratio under the known misclassification likelihood. If the "
         "classifier predicts change with probability")
    para(doc, "θ(p) = p · P(change | change) + (1 − p) · P(change | sustain)",
         align="center", italic=True, after=8)
    para(doc,
         "then the observed count is binomial in θ(p), and a flat prior over p "
         "with a grid evaluation yields a posterior whose mean lies inside the "
         "unit interval by construction and whose credible interval the "
         "interface displays.")
    para(doc,
         "The textbook alternative — inverting the misclassification matrix "
         "algebraically, known as adjusted classify-and-count — was "
         "implemented first and rejected on evidence. Real sessions here carry "
         "seven to fifteen directional turns, and at that size the algebraic "
         "solve lands outside the unit interval and must be clamped: eight of "
         "twelve demonstration sessions pinned to exactly zero or one, "
         "producing a step function between two extremes stated with total "
         "confidence. That is a worse failure than the bias it was intended to "
         "remove, and it was found by running the estimator on data rather "
         "than by reasoning about it.")

    section(doc, "5.6", "Web Application")
    para(doc,
         "The interface presents the trajectory as its central object. Figure "
         "5.1 shows the case view: the index across sessions with the client’s "
         "own baseline corridor, a detected regime change marked, and beneath "
         "it the corrected change-talk series sharing the same horizontal axis "
         "but drawn on its own scale.")
    figure(doc, F + "ui_trace.png", "5.1",
           "Case trajectory view. The upper panel is the Therapeutic Progress "
           "Index against the client's own baseline corridor; the lower panel "
           "is the bias-corrected change-talk ratio with credible intervals.",
           width_cm=13.0)
    para(doc,
         "The two series are deliberately drawn as separate panels rather than "
         "as two lines on one chart. The index is a standardised score centred "
         "at fifty and the ratio is a proportion between zero and one; placing "
         "both on a single pair of axes would invite the reader to interpret "
         "the point where the lines cross, which would be an artefact of the "
         "scaling and not a fact about the client.")
    para(doc,
         "Figure 5.2 shows the model comparison page, which reads the training "
         "artifact through the API rather than restating figures written into "
         "the interface code, so the numbers displayed cannot drift from those "
         "measured.")
    figure(doc, F + "ui_models.png", "5.2",
           "Model comparison page, served from the training artifact.",
           width_cm=13.0)

    section(doc, "5.7", "Testing")
    para(doc,
         "Sixty-nine automated tests cover parsing, scoring, trajectory, "
         "attribution, persistence, the API surface and the safety "
         "constraints. Several encode requirements that would otherwise be "
         "conventions: one asserts that the body-language module can never "
         "emit an emotion word; one asserts that the generated session note "
         "contains no number absent from its inputs; one asserts that the "
         "change-talk series reaches the trajectory endpoint, which is the "
         "regression that module exists to prevent.")
    para(doc, "", after=0)
