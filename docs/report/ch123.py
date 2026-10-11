"""Chapters 1-3: Introduction, Literature Survey, Requirements and Analysis."""

from rlib import bullets, chapter, figure, numbered, para, rich, section, subsection, table


def ch1(doc, first=False):
    # `first` suppresses the page break: the section break already made one.
    chapter(doc, 1, "Introduction", new_page=not first)

    section(doc, "1.1", "Introduction")
    para(doc,
         "Psychotherapy is one of the few clinical interventions whose primary "
         "medium is language, and one of the few whose progress is measured "
         "almost entirely without reference to it. A client attends a session "
         "once a week; every eight or twelve weeks a questionnaire such as the "
         "PHQ-9 or GAD-7 is administered, and the difference between two "
         "administrations stands in for everything that happened in between. "
         "The fifty minutes of speech that constitute the treatment itself — "
         "the material the therapist actually works with — leave no "
         "quantitative trace at all.")
    para(doc,
         "This is not an oversight so much as a practical limitation. "
         "Listening to recordings and coding them by hand is slow; the "
         "established coding schemes used in psychotherapy process research, "
         "such as the Motivational Interviewing Treatment Integrity code, "
         "require trained raters and hours of work per session. The result is "
         "that process measurement is confined to research settings and is "
         "absent from ordinary clinical practice.")
    para(doc,
         "Natural language processing offers an obvious remedy, and a great "
         "deal of recent work has applied it to therapy transcripts. Almost "
         "all of that work, however, classifies individual utterances: it asks "
         "what kind of thing a single sentence was. That is a different "
         "question from the one a clinician asks, which is whether this person "
         "is different from how they were a month ago. The unit of analysis in "
         "the literature is the utterance; the unit of clinical interest is "
         "the person over time.")
    para(doc,
         "TherapyTrace is built around that distinction. It measures five "
         "dimensions of therapeutic process in client speech, scores each "
         "session against the client’s own earlier sessions rather than "
         "against a population, and presents the result as a trajectory. The "
         "system is designed as decision support for reflection and clinical "
         "supervision. It is explicitly not a diagnostic device, and the "
         "software licence under which it is released states that restriction.")

    section(doc, "1.2", "Background and Motivation")
    para(doc,
         "Three observations motivate the design.")
    rich(doc, [("Language carries clinical signal. ", True, False),
               ("A substantial body of research links specific linguistic "
                "features to psychological state. First-person singular "
                "pronoun use rises with depression; absolutist vocabulary is "
                "elevated in anxiety, depression and suicidal ideation; "
                "emotional granularity — the precision with which a person "
                "names their feelings — predicts regulation capacity. These "
                "are not incidental correlations but measurable properties of "
                "how distressed people speak.", False, False)])
    rich(doc, [("Between-person norms are the wrong reference. ", True, False),
               ("A naturally reserved client and a naturally expansive one "
                "differ in vocabulary, sentence length and emotional "
                "explicitness for reasons that have nothing to do with "
                "therapy. Comparing either to a population average measures "
                "personality as much as progress. The clinically meaningful "
                "comparison is within the person: is this client speaking "
                "differently from how this client usually speaks?", False, False)])
    rich(doc, [("Direction matters more than level. ", True, False),
               ("A single session score answers almost nothing. Whether the "
                "score is rising, falling, plateauing or has recently changed "
                "regime is the question a supervisor asks, and answering it "
                "requires a series rather than a point.", False, False)])

    section(doc, "1.3", "Problem Statement")
    para(doc,
         "Existing computational work on psychotherapy transcripts classifies "
         "utterances in isolation and against population-level reference "
         "points. No available system measures change within an individual "
         "client across successive sessions, normalises that measurement "
         "against the client’s own baseline, and presents the result in a form "
         "a clinician or supervisor can act on. Further, where machine "
         "learning components are deployed in clinical interfaces, the effect "
         "of their classification error on the aggregate quantity actually "
         "displayed is seldom quantified, so a clinician may be shown a number "
         "whose bias is unknown.")

    section(doc, "1.4", "Objectives")
    numbered(doc, [
        "To define and implement a set of therapeutic process dimensions "
        "measurable from client speech in a session transcript.",
        "To design a within-person scoring method that normalises each "
        "session against the client’s own earlier sessions, with variance "
        "stabilisation appropriate to small samples.",
        "To estimate trajectory across sessions, including trend, momentum "
        "and change-point detection.",
        "To train and rigorously evaluate supervised classifiers on an "
        "expert-annotated corpus under a protocol that prevents conversational "
        "leakage between training and test data.",
        "To quantify how classifier error propagates into the session-level "
        "quantity presented to the user, and to correct it.",
        "To implement the whole as a working full-stack web application with "
        "an automated test suite.",
    ])

    section(doc, "1.5", "Scope of the Project")
    para(doc, "The following are within scope:")
    bullets(doc, [
        "Analysis of English-language therapy session transcripts supplied as "
        "text, with speaker labels.",
        "Within-person scoring, trajectory estimation and therapist-move "
        "attribution.",
        "Supervised classification of client talk type and therapist "
        "behaviour, trained on a publicly available expert-annotated corpus.",
        "Optional analysis of accompanying audio and video for paralinguistic "
        "and postural descriptors.",
        "A web interface for case management, session upload and trajectory "
        "review.",
    ])
    para(doc, "The following are explicitly outside scope:")
    bullets(doc, [
        "Diagnosis of any mental health condition. The system produces no "
        "diagnostic label and is not a medical device.",
        "Assessment or ranking of therapist competence. Therapist-move "
        "attribution is correlational and is presented as a prompt for "
        "supervision discussion, not as a performance rating.",
        "Inference of emotion from facial expression, which the evidence does "
        "not support at the required reliability.",
        "Real-time analysis during a live session.",
    ])

    section(doc, "1.6", "Organisation of the Report")
    para(doc,
         "Chapter 2 surveys the relevant literature across linguistic markers "
         "of mental state, motivational interviewing process coding, prior "
         "computational work on the corpus used here, and multimodal "
         "approaches. Chapter 3 states the functional and non-functional "
         "requirements and the feasibility analysis. Chapter 4 presents the "
         "system architecture, module decomposition, data model and the "
         "scoring methodology in detail. Chapter 5 describes the "
         "implementation. Chapter 6 reports results: the seven-model "
         "comparison, the error-propagation study and the system-level "
         "validation. Chapter 7 draws conclusions and sets out future work.",
         after=0)


def ch2(doc):
    chapter(doc, 2, "Literature Survey")

    section(doc, "2.1", "Introduction")
    para(doc,
         "The literature relevant to this project falls into four groups: "
         "work establishing that linguistic features carry psychological "
         "signal; work on process coding in motivational interviewing, from "
         "which the annotated corpus used here derives; prior computational "
         "work on that corpus specifically; and multimodal approaches. A fifth "
         "subsection draws out the gap this project addresses.")

    section(doc, "2.2", "Linguistic Markers of Psychological State")
    para(doc,
         "Pennebaker and colleagues established the foundational observation "
         "that function words — pronouns, articles, prepositions — carry "
         "reliable information about psychological state, and that first "
         "person singular pronoun use in particular rises with depression and "
         "emotional distress. The effect is small per text but consistent "
         "across corpora and languages.")
    para(doc,
         "Al-Mosaiwi and Johnstone extended this to absolutist vocabulary, "
         "showing that words expressing totality such as “always”, “never” "
         "and “completely” occur at significantly elevated rates in anxiety, "
         "depression and suicidal ideation forums, and that the elevation is "
         "larger than that for negative emotion words. This finding motivates "
         "the absolutism auxiliary feature in the present system.")
    para(doc,
         "Barrett and colleagues reviewed the evidence linking facial "
         "configurations to emotional states and concluded that the mapping is "
         "far weaker and more context-dependent than applied work assumes. "
         "This review is the explicit basis for the decision in this project "
         "not to infer emotion from video, and the corresponding module is "
         "constrained by an automated test asserting that it can never emit an "
         "emotion label.")
    para(doc,
         "Work on emotional granularity, principally by Barrett and by "
         "Kashdan and colleagues, shows that individuals who differentiate "
         "their emotional states more finely regulate them more effectively. "
         "Granularity is therefore treated here as a therapeutic process "
         "dimension rather than as a stylistic property.")

    section(doc, "2.3", "Process Coding in Motivational Interviewing")
    para(doc,
         "Motivational interviewing provides the most developed process-coding "
         "tradition in psychotherapy research. The Motivational Interviewing "
         "Skill Code and the later Motivational Interviewing Treatment "
         "Integrity instrument define categories for therapist behaviour — "
         "open and closed questions, simple and complex reflections, giving "
         "information — and for client speech, principally the distinction "
         "between change talk and sustain talk.")
    para(doc,
         "The relationship between these codes and outcome is not simple, and "
         "the detail matters for this project. The meta-analysis by Magill and "
         "colleagues synthesising fifty-eight MI process studies found that "
         "raw change-talk frequency showed no significant independent "
         "association with behavioural outcome, that sustain talk was "
         "associated with worse outcomes (r = 0.19), and that the "
         "proportion of change talk relative to total directional speech "
         "related to reduced risk behaviour (r = −0.16). The implication "
         "adopted here is that the quantity worth computing is a proportion, "
         "not a count.")

    section(doc, "2.4", "Prior Computational Work on AnnoMI")
    para(doc,
         "AnnoMI, released by Wu and colleagues, is a corpus of 133 "
         "expert-annotated motivational interviewing dialogues transcribed "
         "from publicly available demonstration videos. It is important to "
         "state precisely what the corpus is: these are performed "
         "demonstrations used for MI training, not recordings of clinical "
         "sessions, and twenty-three of the transcripts are deliberately "
         "staged examples of poor practice. Every result reported in this "
         "project is a claim about this corpus and nothing more.")
    para(doc,
         "A considerable body of work has since used it. Table 2.1 summarises "
         "the principal studies and what each does.")

    table(doc, "2.1", "Prior computational work using the AnnoMI corpus",
          ["Study", "Contribution", "Task type"],
          [
              ["Wu et al. (2022, 2023)", "Created and released the corpus; baseline utterance classifiers", "Classification"],
              ["Tavabi et al. (2021)", "Behaviour classification in motivational interviewing sessions", "Classification"],
              ["Cohen et al. (2022, 2024)", "RoBERTa classifiers; LLM-automated MITI and CLEAR scoring", "Classification"],
              ["Hoang, Rogers & Ross (2024)", "Client language coding including commitment strength", "Classification"],
              ["Kumar et al. (2024), IC-AnnoMI", "LLM-augmented corpus expansion and bias measurement", "Data augmentation"],
              ["M3TCM (2024)", "Multimodal multi-task context model for utterance classification", "Classification"],
              ["EMMI (2025)", "Empathic multimodal MI dataset and feature analysis", "Feature engineering"],
              ["VERVE (2023)", "Rewriting counsellor turns for improved reflection quality", "Generation"],
              ["Consistent Client Simulation (2025)", "Simulating clients for chatbot evaluation", "Simulation"],
          ],
          widths=[4.0, 7.5, 3.1])

    para(doc,
         "The pattern is consistent and is the basis of the gap claimed in "
         "Section 2.7: every one of these studies classifies, generates or "
         "simulates a single utterance. None asks whether a person is changing "
         "across sessions, none normalises against an individual’s own "
         "baseline, and none estimates whether a therapist move works "
         "differently depending on the state the client is already in.")
    para(doc,
         "A further methodological point deserves emphasis. Published figures "
         "on this corpus vary widely, and the reported split protocol is often "
         "not stated. Because utterances within one conversation share a "
         "speaker, a topic and a speaking style, a split made at the utterance "
         "level permits a model to recognise a conversation it has already "
         "seen, which inflates every metric. Results obtained under "
         "conversation-grouped splits are therefore not directly comparable "
         "with results whose protocol is unstated, and this report adopts the "
         "stricter protocol throughout.")

    section(doc, "2.5", "Summary of Reviewed Literature")
    para(doc,
         "Table 2.2 consolidates the principal works reviewed, with the "
         "contribution of each and the limitation that motivates the present "
         "project.")
    table(doc, "2.2", "Summary of reviewed literature",
          ["#", "Author (Year)", "Focus", "Finding relevant here", "Limitation"],
          [
              ["1", "Pennebaker et al. (2003)", "Function words and psychological state", "Pronoun use tracks distress reliably across corpora", "Between-person; no therapy context"],
              ["2", "Al-Mosaiwi & Johnstone (2018)", "Absolutist language", "Absolutist words elevated in anxiety, depression, suicidal ideation", "Forum text, cross-sectional"],
              ["3", "Barrett et al. (2019)", "Facial expression and emotion", "Expression-to-emotion mapping unreliable", "Motivates exclusion of emotion inference"],
              ["4", "Kashdan et al. (2015)", "Emotional granularity", "Finer differentiation predicts better regulation", "Not operationalised for transcripts"],
              ["5", "Boyd & Schwartz (2021)", "Verbal behaviour analysis", "Review of language-psychology methods", "No longitudinal clinical application"],
              ["6", "Miller & Rollnick (2013)", "Motivational interviewing", "Defines change talk and sustain talk", "Theoretical, not computational"],
              ["7", "Moyers et al. (2016)", "MITI 4 coding instrument", "Reliable therapist behaviour categories", "Manual coding, hours per session"],
              ["8", "Magill et al. (2018)", "MI process meta-analysis", "Proportion change talk relates to outcome; raw counts do not", "Establishes which quantity to compute"],
              ["9", "Wu et al. (2022, 2023)", "AnnoMI corpus", "133 expert-annotated MI dialogues released", "Performed demonstrations, one conversation per client"],
              ["10", "Tavabi et al. (2021)", "Multimodal MI coding", "Behaviour classification from speech and text", "Utterance-level only"],
              ["11", "Cohen et al. (2022, 2024)", "Transformer MI scoring", "RoBERTa and LLM automated MITI scoring", "Single utterance; no trajectory"],
              ["12", "Hoang et al. (2024)", "Client language coding", "Commitment strength added to talk type", "Cross-sectional"],
              ["13", "Kumar et al. (2024)", "IC-AnnoMI", "LLM-augmented corpus, bias measured", "Augmentation, not new measurement"],
              ["14", "M3TCM (2024)", "Multimodal context model", "Context improves utterance classification", "Split protocol unstated"],
              ["15", "EMMI (2025)", "Multimodal MI dataset", "Empathy features across modalities", "No within-person normalisation"],
              ["16", "Xiao et al. (2015)", "Automated empathy rating", "Speech and language predict therapist empathy", "Rates therapist, not client change"],
              ["17", "Tao et al. (2024)", "CUEMPATHY", "156 sessions, 39 dyads, four sessions each", "Cantonese; restricted access"],
              ["18", "Tsakalidis et al. (2022)", "CLPsych moments of change", "Within-person change detection over time", "Social media, not therapy"],
              ["19", "Saerens et al. (2002)", "Prior probability adjustment", "Correcting classifier outputs for prevalence shift", "Not applied to clinical interfaces"],
              ["20", "Forman (2008)", "Quantification learning", "Counting predicted labels is a biased prevalence estimator", "General ML; no clinical deployment"],
          ],
          widths=[0.8, 3.0, 2.9, 4.2, 3.7], size=8)

    section(doc, "2.6", "Multimodal and Longitudinal Approaches")
    para(doc,
         "Multimodal work on counselling data includes CUEMPATHY, a corpus of "
         "156 counselling sessions from 39 therapist-client dyads with aligned "
         "audio and three validated empathy scales; it is notable here because "
         "each dyad contributes four sessions recorded weekly, making it one "
         "of very few corpora with genuine repeated measurement on the same "
         "client. DAIC-WOZ provides 189 clinical interviews with audio, "
         "transcripts and PHQ-8 scores, but one session per participant. "
         "Larger transcript collections such as the Counseling and "
         "Psychotherapy Transcripts Series contain several thousand real "
         "sessions and permit following a client across sessions, but are "
         "available only by institutional subscription.")
    para(doc,
         "Longitudinal modelling of psychological state is more developed on "
         "social media data than on therapy data. The CLPsych shared tasks on "
         "capturing moments of change in longitudinal user posts address a "
         "structurally similar problem — detecting within-person shifts over "
         "time — on a very different substrate.")

    section(doc, "2.7", "Research Gap")
    para(doc,
         "Synthesising the above, three gaps are identified and addressed in "
         "this work.")
    bullets(doc, [
        ("Unit of analysis. ", "Prior computational work on therapy "
         "transcripts operates on the utterance. No reviewed system models the "
         "client as a subject measured repeatedly over time."),
        ("Reference point. ", "Scores are reported against population "
         "distributions or corpus-level class priors. No reviewed system "
         "normalises a client against their own earlier sessions."),
        ("Error propagation. ", "Where classifiers are embedded in clinical "
         "interfaces, the bias their error induces in the displayed aggregate "
         "is not quantified. This report measures that bias and corrects it."),
    ])
    para(doc, "", after=0)


def ch3(doc):
    chapter(doc, 3, "Requirements and System Analysis")

    section(doc, "3.1", "Introduction")
    para(doc,
         "This chapter states what the system must do, the qualities it must "
         "exhibit, and the analysis supporting the conclusion that it can be "
         "built within the constraints of the project.")

    section(doc, "3.2", "Functional Requirements")
    table(doc, "3.1", "Functional requirements",
          ["ID", "Requirement", "Priority"],
          [
              ["FR-1", "Register a case under a pseudonymous code with no identifying detail", "High"],
              ["FR-2", "Accept a session transcript as text or file upload and parse speaker turns", "High"],
              ["FR-3", "Extract five process dimensions and five auxiliary features per client turn", "High"],
              ["FR-4", "Score each session against the client's own baseline", "High"],
              ["FR-5", "Recompute all earlier sessions when the baseline widens", "High"],
              ["FR-6", "Estimate trend, momentum and change points across a case", "High"],
              ["FR-7", "Attribute client change to the preceding therapist move", "Medium"],
              ["FR-8", "Classify client talk type and therapist behaviour using trained models", "Medium"],
              ["FR-9", "Correct classifier bias before any aggregate is displayed", "High"],
              ["FR-10", "Record self-report outcome scores and correlate them with the index", "Medium"],
              ["FR-11", "Accept optional audio and video for paralinguistic and postural description", "Low"],
              ["FR-12", "Present model performance figures to the user within the application", "Medium"],
          ],
          widths=[1.6, 10.5, 2.5])

    section(doc, "3.3", "Non-Functional Requirements")
    bullets(doc, [
        ("Safety. ", "The system must not emit diagnostic labels, emotion "
         "attributions or therapist competence ratings. This is enforced by "
         "automated tests, not only by convention."),
        ("Privacy. ", "No identifying information is stored. Cases carry "
         "pseudonymous codes. Transcript directories and database files are "
         "excluded from version control."),
        ("Honesty of presentation. ", "Any displayed quantity derived from an "
         "imperfect model must carry its uncertainty. Point estimates from "
         "thin evidence must be visibly marked as such."),
        ("Graceful degradation. ", "The system must run with the machine "
         "learning stack absent, falling back to the rule-based layer, so that "
         "a fresh installation is usable before training artifacts exist."),
        ("Reproducibility. ", "Every reported figure must be regenerable from "
         "committed code and a publicly downloadable dataset."),
        ("Performance. ", "A session of ordinary length must be analysed in "
         "under five seconds on commodity hardware."),
    ])

    section(doc, "3.4", "Feasibility Analysis")
    subsection(doc, "3.4.1", "Technical Feasibility")
    para(doc,
         "All components are implementable with mature open-source libraries: "
         "scikit-learn for the supervised layer, SciPy for the statistical "
         "procedures, FastAPI and SQLAlchemy for the service layer, React for "
         "the interface, and MediaPipe and librosa for the optional modalities. "
         "No component requires specialised hardware; the trained models are "
         "linear and occupy under half a megabyte.")

    subsection(doc, "3.4.2", "Data Feasibility")
    para(doc,
         "The supervised layer requires expert annotation, which AnnoMI "
         "supplies and which is freely downloadable. The longitudinal layer "
         "requires repeated sessions from the same client, which no freely "
         "available corpus supplies. This is the principal constraint on the "
         "project and is addressed in two ways: by validating the longitudinal "
         "machinery against synthetic cases whose trajectory is known by "
         "construction, and by stating the limitation explicitly rather than "
         "implying clinical validation that has not been performed.")

    subsection(doc, "3.4.3", "Ethical Feasibility")
    para(doc,
         "The system processes no real clinical material in this project. The "
         "corpus used is public and consists of performed demonstrations. Any "
         "deployment on genuine session recordings would require institutional "
         "ethics approval, informed consent covering secondary analysis, and "
         "encryption at rest; the report states this requirement rather than "
         "treating it as a formality.")

    section(doc, "3.5", "Software and Hardware Requirements")
    table(doc, "3.2", "Development environment",
          ["Component", "Specification"],
          [
              ["Language (backend)", "Python 3.11 or later"],
              ["Web framework", "FastAPI with Uvicorn"],
              ["Persistence", "SQLAlchemy 2.0 over SQLite"],
              ["Machine learning", "scikit-learn, XGBoost, LightGBM, SciPy, NumPy, pandas"],
              ["Frontend", "React 18 with Vite"],
              ["Optional modalities", "MediaPipe, librosa, OpenCV"],
              ["Testing", "pytest"],
              ["Hardware", "Any x86-64 machine with 8 GB RAM; no GPU required"],
          ],
          widths=[4.5, 10.1])
    para(doc, "", after=0)
