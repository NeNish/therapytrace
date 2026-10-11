"""Chapter 6 (Results and Discussion), Chapter 7, References, Appendices."""

import json
from pathlib import Path

from rlib import (bullets, chapter, figure, frontmatter_heading, numbered, page_break,
                  para, rich, section, subsection, table)

F = "fig/"
N = json.loads((Path(__file__).parent / "nums.json").read_text())


def _fmt(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}"


def _rows(task):
    out = []
    for name, cvf1, cvsd, acc, mf1, wf1, auc, fit, p in N[task]["rows"]:
        out.append([
            name,
            f"{cvf1:.3f} ± {cvsd:.3f}",
            _fmt(acc), _fmt(mf1), _fmt(wf1), _fmt(auc),
            "—" if fit is None else f"{fit:.2f}",
            "—" if p is None else f"{p:.3f}",
        ])
    return out


def ch6(doc):
    chapter(doc, 6, "Results and Discussion")

    section(doc, "6.1", "Introduction")
    para(doc,
         "This chapter reports three sets of results. The first compares seven "
         "classifiers on two annotation tasks. The second measures how the "
         "error of the deployed classifier propagates into the quantity the "
         "interface displays, and evaluates a correction. The third examines "
         "the system end to end. All classification figures come from "
         "conversations held out before any model was fitted.")

    section(doc, "6.2", "Dataset and Protocol")
    para(doc,
         "AnnoMI consists of 133 expert-annotated motivational interviewing "
         "dialogues, 9,699 unique utterances, split 6,826 therapist and 6,725 "
         "client turns, across 44 topics. One hundred and ten conversations "
         "are annotated as high quality and twenty-three as low quality. It "
         "is important to restate that these are performed demonstrations "
         "used for MI training, not clinical recordings.")
    table(doc, "6.1", "Task composition after filtering",
          ["Task", "Utterances", "Conversations", "Classes", "Class balance"],
          [
              ["Client talk type", N["client"]["n"], N["client"]["nconv"], 3,
               ", ".join(f"{k} {v}" for k, v in N["client"]["bal"].items())],
              ["Therapist behaviour", N["therapist"]["n"], N["therapist"]["nconv"], 4,
               ", ".join(f"{k} {v}" for k, v in N["therapist"]["bal"].items())],
          ],
          widths=[3.2, 2.2, 2.6, 1.6, 5.0])
    para(doc,
         "Twenty-five per cent of conversations form the held-out test set: "
         f"{N['client']['ntest']} utterances from "
         f"{N['client']['ntestconv']} conversations on the client task and "
         f"{N['therapist']['ntest']} from "
         f"{N['therapist']['ntestconv']} on the therapist task. Model "
         "selection used five-fold grouped cross-validation on the remainder; "
         "the test conversations were scored once, for every model, so the "
         "runner-up's figure is reported rather than hidden.")

    section(doc, "6.3", "Model Comparison")
    para(doc,
         "Every learner sees an identical feature matrix — TF-IDF word "
         "unigrams and bigrams, character three- to five-grams, and twelve "
         "lexicon features — so a difference in score is attributable to the "
         "learner and nothing else.")

    table(doc, "6.2", "Client talk type: held-out results for all seven learners "
          "and the majority-class floor",
          ["Model", "CV macro-F1", "Acc.", "Macro-F1", "Wtd-F1", "AUC", "Fit (s)", "p"],
          _rows("client"),
          widths=[3.3, 2.4, 1.5, 1.8, 1.6, 1.4, 1.4, 1.2], size=9)

    table(doc, "6.3", "Therapist behaviour: held-out results",
          ["Model", "CV macro-F1", "Acc.", "Macro-F1", "Wtd-F1", "AUC", "Fit (s)", "p"],
          _rows("therapist"),
          widths=[3.3, 2.4, 1.5, 1.8, 1.6, 1.4, 1.4, 1.2], size=9)

    figure(doc, F + "bench_models.png", "6.1",
           "Cross-validated macro-F1 for each learner on both tasks, with "
           "fold-to-fold standard deviation, against the majority-class floor.",
           width_cm=14.5)

    subsection(doc, "6.3.1", "Interpreting the Accuracy Figures")
    para(doc,
         "The client-task accuracies require comment, because taken at face "
         "value they appear poor. The majority-class floor on that task is "
         "0.554 accuracy: a model that answers “neutral” for every utterance "
         "scores higher than any of the seven real learners. This is a "
         "property of a corpus that is fifty-three per cent neutral, and it "
         "demonstrates that accuracy is an uninformative metric here rather "
         "than that the models fail. The metric that moves is macro-F1, which "
         "rises from 0.238 at the floor to 0.489 — slightly more than double.")
    para(doc,
         "The multilayer perceptron illustrates the point sharply. It records "
         "the highest accuracy of any model on the client task at 0.618, and "
         "the lowest sustain-class F1 at 0.131. It achieves its accuracy by "
         "predicting the majority class, which is precisely the behaviour "
         "macro-F1 is designed to penalise.")

    subsection(doc, "6.3.2", "Statistical Separability")
    para(doc,
         "A paired t-test across the five cross-validation folds was computed "
         "between the selected model and each alternative. On the client task "
         "the smallest p-value against any competitor is 0.068; the selected "
         "model cannot be separated from any of the other six at the five per "
         "cent level. The spread between the best and worst of the seven is "
         "0.051 macro-F1, against a fold-to-fold standard deviation of 0.043 "
         "for a single model. The differences lie inside the noise.")
    figure(doc, F + "bench_folds.png", "6.2",
           "Fold-level macro-F1 for every learner, with the paired-test "
           "difference and p-value against the selected model.", width_cm=14.5)
    para(doc,
         "This is reported as a positive finding rather than a disappointment. "
         "It establishes that the ceiling on this task is set by the task and "
         "not by the choice of learner, and therefore that effort spent "
         "pursuing a stronger classifier would not be repaid. With roughly "
         "three thousand short utterances and tens of thousands of sparse "
         "features, a regularised linear boundary already has the right "
         "capacity.")

    subsection(doc, "6.3.3", "Computational Cost")
    para(doc,
         "Fit times were measured separately on an otherwise idle machine, "
         "because timings taken during a benchmark run are corrupted by "
         "whatever else is competing for the processor. XGBoost requires 56.0 "
         "seconds to fit on the client task where naive Bayes requires 0.01, a "
         "difference of three orders of magnitude for no gain in score.")
    figure(doc, F + "bench_cost.png", "6.3",
           "Cross-validated macro-F1 against classifier fit time, logarithmic "
           "horizontal axis.", width_cm=14.5)

    subsection(doc, "6.3.4", "Per-Class Behaviour")
    para(doc,
         "Averages conceal the finding that determines which model should be "
         "deployed. Table 6.4 gives per-class recall on the client task.")
    rec = N["client"]["recall"]
    table(doc, "6.4", "Client talk type: per-class recall on held-out conversations",
          ["Model", "change", "neutral", "sustain"],
          [[m, _fmt(r["change"]), _fmt(r["neutral"]), _fmt(r["sustain"])]
           for m, r in rec.items()],
          widths=[5.0, 3.2, 3.2, 3.2])
    para(doc,
         "Sustain-talk recall varies almost five-fold across the seven, from "
         "0.072 to 0.335. The model selected by macro-F1 is near the bottom of "
         "that range. Since the displayed quantity depends on separating "
         "change from sustain, sustain recall and not macro-F1 is the correct "
         "criterion for deployment — a conclusion that selects logistic "
         "regression, which is the model the system ships.")
    figure(doc, F + "bench_perclass.png", "6.4",
           "Per-class F1 for every learner on both tasks. The minority classes "
           "are the difficulty in each case.", width_cm=14.5)
    figure(doc, F + "bench_confusion.png", "6.5",
           "Confusion matrices for the deployed model, row-normalised with "
           "counts.", width_cm=14.0)

    subsection(doc, "6.3.5", "Feature Ablation")
    para(doc,
         "A separate ablation varied the features while holding the "
         "classifier at logistic regression, to establish whether the "
         "project’s own lexicon contributes anything beyond bag-of-words.")
    table(doc, "6.6", "Feature ablation, cross-validated macro-F1",
          ["Feature set", "Client talk", "Therapist behaviour"],
          [
              ["Majority-class floor", "0.228", "0.146"],
              ["TherapyTrace lexicon only (10 features)", "0.451", "0.523"],
              ["TF-IDF only (thousands of features)", "0.471", "0.607"],
              ["TF-IDF + lexicon", "0.475", "0.644"],
              ["TF-IDF + lexicon + previous turn", "0.471", "0.633"],
          ],
          widths=[7.2, 3.7, 3.7])
    para(doc,
         "Two results are worth noting. The ten interpretable lexicon "
         "features alone reach 0.451 on the client task, within 0.020 of "
         "several thousand TF-IDF features, which indicates the dimensions "
         "capture much of the available signal in a form a clinician can "
         "read. On the therapist task the lexicon adds 0.037 macro-F1 on top "
         "of TF-IDF, a clearer contribution. Adding the previous turn as "
         "context slightly reduced performance on both tasks, contrary to "
         "expectation; the hypothesis that client language is best read as a "
         "response to what the therapist just said was not supported at this "
         "corpus size.")
    figure(doc, F + "ablation.png", "6.7",
           "Feature ablation. The interpretable lexicon alone approaches "
           "bag-of-words performance on the client task.", width_cm=13.5)

    section(doc, "6.4", "Error Propagation and Correction")
    para(doc,
         "A macro-F1 of 0.49 does not answer the question that matters for a "
         "clinical interface, which is whether the number on the screen can be "
         "trusted. The displayed quantity is not a turn label but a "
         "session-level ratio aggregated over every directional turn, so the "
         "proper question is how far per-turn error propagates into that "
         "aggregate. Twenty thousand sessions of forty directional turns were "
         "simulated, each turn corrupted according to the deployed model’s own "
         "held-out confusion matrix.")

    subsection(doc, "6.4.1", "The Measured Bias")
    para(doc,
         "For a client genuinely evenly split between change and sustain talk, "
         "the uncorrected interface would display 0.75. The bias of +0.25 does "
         "not diminish with session length — 0.752 at ten directional turns "
         "and 0.752 at one hundred and twenty — because it is systematic "
         "rather than random. More data does not help.")
    para(doc,
         "The cause is specific. Sustain talk is recovered as sustain only "
         "twenty-one per cent of the time and read as change thirty-one per "
         "cent of the time; missed sustain inflates the numerator and shrinks "
         "the denominator simultaneously, and both errors push the ratio the "
         "same way.")
    para(doc,
         "The consequence for the product is more serious than the bias "
         "itself. A true range of 0.10 to 0.90 compresses to a displayed range "
         "of 0.63 to 0.85, a slope of 0.27. A client who is barely changing "
         "and one who is changing a great deal would appear almost identical "
         "on the trace, which is exactly the comparison the system exists to "
         "support.")

    subsection(doc, "6.4.2", "The Correction")
    para(doc,
         "Applying the prevalence correction described in Section 5.5 restores "
         "the slope from 0.27 to 0.80 and reduces the bias at the midpoint "
         "from +0.25 to +0.07. Variance increases, which is the honest "
         "trade: the estimate becomes noisier but ceases to be systematically "
         "wrong in one direction.")
    figure(doc, F + "bench_reliability.png", "6.6",
           "Left: displayed ratio against true ratio, before and after "
           "correction, with the diagonal marking a faithful display. Right: "
           "sustain recall by learner, which drives the bias.", width_cm=14.5)
    para(doc,
         "Even corrected, a session-to-session change of +0.10 moves the "
         "display in the correct direction only about fifty-eight per cent of "
         "the time at forty turns, against fifty per cent for a coin. The "
         "honest reading is that this series supports a trend across several "
         "sessions and does not support a judgement about any single pair of "
         "sessions, and the interface is written to say so.")

    section(doc, "6.5", "Reframing the Task")
    para(doc,
         "Because the interface never uses the neutral class, the three-class "
         "problem is not the problem the product needs solved. A binary "
         "change-versus-sustain classifier was therefore evaluated under the "
         "identical protocol on the "
         f"{N['binary']['n']} directional turns across "
         f"{N['binary']['nconv']} conversations.")
    table(doc, "6.5", "Three-class against binary framing, deployed model",
          ["Metric", "Three-class", "Binary", "Change"],
          [
              ["Accuracy", "0.542", "0.653", "+0.111"],
              ["Macro-F1", "0.489", "0.616", "+0.127"],
              ["Sustain F1", "0.351", "0.497", "+0.145"],
              ["Sustain recall", "0.335", "0.478", "+0.143"],
              ["ROC-AUC", "0.678", "0.660", "−0.018"],
          ],
          widths=[4.6, 3.4, 3.3, 3.3])
    para(doc,
         "Sustain recall rising from 0.335 to 0.478 is the consequential "
         "change, since under-detected sustain talk is the cause of the "
         "display bias measured above. Two limits are stated. First, accuracy "
         f"clears its own floor of {N['binary']['floor']:.3f} by only 0.009; "
         "macro-F1 clears its floor by 0.225, and that is the real margin. "
         "Second, the binary figures assume the directional turns are already "
         "identified, which is true of the annotated corpus and not of "
         "production. Deploying this gain requires a two-stage model whose "
         "first stage would contribute its own errors, so the binary result is "
         "not a drop-in replacement and is not quoted as one.")

    section(doc, "6.6", "System-Level Results")
    subsection(doc, "6.6.1", "Convergent Behaviour of Two Independent Measures")
    para(doc,
         "On the demonstration case, the Therapeutic Progress Index and the "
         "corrected change-talk series correlate at r = 0.82 across twelve "
         "sessions. The two are computed independently: the index from the "
         "rule-based lexicon layer, the ratio from a model trained on expert "
         "annotations. Agreement between them is therefore convergent evidence "
         "rather than restatement. The qualification is important: this is "
         "measured on synthetic cases whose trajectory is known by "
         "construction, so part of the agreement reflects the generator. It "
         "demonstrates that the two layers respond to the same underlying "
         "signal; it is not clinical validation.")

    subsection(doc, "6.6.2", "State-Conditional Response to Therapist Moves")
    para(doc,
         "The third novelty claim of this project is that the effect of a "
         "therapist move depends on the state the client is already in, and "
         "that an aggregate figure for “how well reflections work” therefore "
         "conceals the clinically useful information. This was tested on "
         f"{1815} therapist-turn triples across {123} conversations, with "
         "client state before the move classified as stuck, middling or "
         "moving by tercile of the preceding process score.")
    table(doc, "6.7", "Probability of change talk following each therapist move, "
          "by the client's state beforehand",
          ["Therapist move", "Client stuck", "Middling", "Client moving", "Spread", "p"],
          [
              ["Question", "0.268 (n=220)", "0.412 (n=85)", "0.479 (n=169)", "0.211", "<0.001"],
              ["Reflection", "0.260 (n=177)", "0.382 (n=68)", "0.455 (n=110)", "0.195", "0.001"],
              ["Therapist input", "0.253 (n=79)", "0.265 (n=34)", "0.296 (n=54)", "0.043", "n.s."],
              ["Other", "0.244 (n=402)", "0.331 (n=145)", "0.415 (n=272)", "0.172", "—"],
          ],
          widths=[3.2, 2.6, 2.3, 2.6, 1.7, 2.2])
    para(doc,
         "The pattern is consistent and, for questions and reflections, "
         "statistically significant: both roughly double their association "
         "with subsequent change talk when the client is already moving "
         "relative to when the client is stuck. Therapist input shows almost "
         "no state dependence, with a spread of 0.043 that does not reach "
         "significance.")
    figure(doc, F + "state_conditional.png", "6.8",
           "Change-talk rate following each therapist move, split by the "
           "client's state before the move.", width_cm=14.0)
    para(doc,
         "The interpretive caution here is essential and is stated in the "
         "interface as well as in this report. These are associations, not "
         "causal effects. A client who is already moving produces more change "
         "talk whatever the therapist does, so part of the difference is "
         "confounding by indication rather than differential efficacy of the "
         "move. What the analysis supports is the weaker but still useful "
         "claim that an unconditional average — questions are followed by "
         "change talk 0.369 of the time — is an average over two very "
         "different situations and is the wrong number to act on.")

    subsection(doc, "6.6.3", "The Paralinguistic Tier")
    para(doc,
         "Timing features derived from the transcript — utterance duration, "
         "speech rate, pause length — were evaluated both raw and z-scored "
         "within each client. Text alone reaches an AUC of "
         f"{N['prosody']['auc']['text']:.3f}. Adding timing features "
         f"reduces it to {N['prosody']['auc']['text_timing_raw']:.3f} raw and "
         f"{N['prosody']['auc']['text_timing_wp']:.3f} within-person, a "
         f"multimodal change of {N['prosody']['multimodal_gain_over_text']:+.3f}. "
         "The additional modality actively degraded performance. This is a "
         "negative result on timing features specifically, not on acoustics in "
         "general, since AnnoMI ships no audio and pitch and voice-quality "
         "features could not be tested on it.")

    subsection(doc, "6.6.4", "Test Coverage")
    para(doc,
         "The system passes sixty-nine automated tests. Three further tests "
         "covering the optional audio and video tiers require libraries not "
         "installed in every environment and are skipped where absent.")

    section(doc, "6.7", "Discussion")
    para(doc,
         "The results support a conclusion that is unusual to report but is "
         "the honest one: the classification component of this system is "
         "mediocre, cannot readily be improved by changing the learner, and "
         "this does not undermine the project because classification was never "
         "the contribution.")
    para(doc,
         "What the results do establish is that an imperfect classifier can "
         "still support a usable session-level clinical measure, provided the "
         "error it introduces into the aggregate is measured and corrected. "
         "Without correction, the system would have displayed a confidently "
         "wrong number to a clinician — reading 0.75 for a client who was "
         "evenly split, and compressing real differences between clients into "
         "near-invisibility. That failure was identified before deployment by "
         "measuring it, which is the methodological point the project makes.")
    para(doc,
         "The comparison with published figures on AnnoMI should be made "
         "carefully. Reported scores vary widely and the split protocol is "
         "frequently unstated; since conversation-grouped splits yield "
         "substantially lower figures than utterance-level splits, results "
         "whose protocol is unknown are not directly comparable with those "
         "reported here.", after=0)


def ch7(doc):
    chapter(doc, 7, "Conclusion and Future Work")

    section(doc, "7.1", "Conclusion")
    para(doc,
         "This project set out to measure change within an individual "
         "psychotherapy client across sessions, rather than to classify "
         "utterances in isolation. A working full-stack system was built that "
         "extracts five therapeutic process dimensions from client speech, "
         "normalises each session against that client’s own earlier sessions "
         "with shrinkage appropriate to small samples, estimates trajectory "
         "with trend and change-point analysis, and presents the result as a "
         "trace a clinician or supervisor can read.")
    para(doc,
         "Seven classifiers were compared on two annotation tasks under a "
         "protocol that prevents conversational leakage. The comparison "
         "produced a negative result that is reported rather than suppressed: "
         "on the client task the learners are statistically indistinguishable, "
         "and the ceiling is set by the task rather than by the model.")
    para(doc,
         "That finding motivated the principal contribution. The project "
         "measured how classifier error propagates into the quantity a user "
         "would actually read, and found a systematic upward bias of 0.25 that "
         "does not diminish with more data and which compresses real "
         "variation between clients into a narrow band. A prevalence "
         "correction restores fidelity from a slope of 0.27 to 0.80. The "
         "standard algebraic correction was found to fail at realistic session "
         "lengths, clipping most sessions to the extremes, so a Bayesian "
         "posterior was adopted in its place. This failure was discovered by "
         "running the estimator on data rather than by reasoning about it, and "
         "the sequence is recorded here because it is the substance of the "
         "engineering.")

    section(doc, "7.2", "Limitations")
    para(doc,
         "The limitations are stated plainly, since a reader who discovers "
         "them unaided will reasonably doubt everything else.")
    bullets(doc, [
        ("No real longitudinal clinical data. ", "The central claim concerns "
         "change within a person across sessions. No publicly available corpus "
         "contains repeated sessions from the same client, so that layer is "
         "demonstrated on synthetic cases with known trajectories. This is a "
         "pipeline validation and is not clinical evidence."),
        ("The corpus is performed, not clinical. ", "AnnoMI consists of "
         "demonstration videos used for training. Transfer to genuine sessions "
         "is unmeasured."),
        ("Modest classification performance. ", "The deployed classifier "
         "reaches 0.489 macro-F1 on the client task. The error propagation is "
         "corrected but not eliminated, and single-session judgements are not "
         "supported."),
        ("One unbalanced learner. ", "The multilayer perceptron cannot be "
         "class-balanced in scikit-learn, so its macro-F1 is mildly "
         "disadvantaged relative to the others."),
        ("No transformer baseline. ", "A fine-tuned encoder is the obvious "
         "next comparison and was not run, as the development environment had "
         "no model-hub access."),
        ("English only. ", "The lexicons and trained models are "
         "English-specific and would not transfer without replacement."),
    ])

    section(doc, "7.3", "Future Work")
    numbered(doc, [
        "Secure a longitudinal clinical corpus. CUEMPATHY offers four weekly "
        "sessions for each of thirty-nine dyads with aligned audio and "
        "validated empathy scales, and is the closest available match to the "
        "longitudinal claim. The Counseling and Psychotherapy Transcripts "
        "Series offers several thousand English sessions under institutional "
        "subscription. DAIC-WOZ, freely available on request, supplies PHQ-8 "
        "scores against which the index could be validated even in single "
        "sessions.",
        "Add a transformer tier. A domain-adapted encoder such as MentalBERT "
        "should be evaluated under the same conversation-grouped protocol, "
        "with the expectation of gains principally on the two dimensions the "
        "lexicon separates least well.",
        "Build the two-stage directional classifier so the binary gain "
        "reported in Section 6.5 can reach production, and measure how the "
        "first stage's errors propagate.",
        "Extend the prevalence correction to the therapist behaviour "
        "classifier, whose aggregate proportions are subject to the same class "
        "of bias and are not yet corrected.",
        "Test the acoustic tier on real therapy audio. The negative result "
        "reported here concerns timing features only; pitch, energy and voice "
        "quality remain untested because the corpus ships no audio.",
        "Obtain ethics approval for a small prospective study, which is the "
        "only route to establishing whether the index tracks anything "
        "clinically real.",
    ])

    section(doc, "7.4", "Closing Remark")
    para(doc,
         "The most useful thing this project produced is not a score but a "
         "method of interrogation: measure what your model’s error does to the "
         "number a person will act on, before you show it to them. Applied "
         "here, that question turned a system that would have displayed "
         "confident misinformation into one that displays an honest estimate "
         "with its uncertainty attached. The classifier remains mediocre. The "
         "interface is now truthful about it.", after=0)


def references(doc):
    page_break(doc)
    frontmatter_heading(doc, "REFERENCES", underline=False)
    refs = [
        "Al-Mosaiwi, M., & Johnstone, T. (2018). In an absolute state: Elevated use of absolutist words is a marker specific to anxiety, depression, and suicidal ideation. Clinical Psychological Science, 6(4), 529–542.",
        "Barrett, L. F., Adolphs, R., Marsella, S., Martinez, A. M., & Pollak, S. D. (2019). Emotional expressions reconsidered: Challenges to inferring emotion from human facial movements. Psychological Science in the Public Interest, 20(1), 1–68.",
        "Boyd, R. L., & Schwartz, H. A. (2021). Natural language analysis and the psychology of verbal behavior. Journal of Language and Social Psychology, 40(1), 21–41.",
        "Cohen, A., Rubinstein, A., & Shapira, N. (2022). Automatic evaluation of motivational interviewing counselling using transformer models. Proceedings of the Workshop on Computational Linguistics and Clinical Psychology.",
        "Efron, B., & Morris, C. (1977). Stein's paradox in statistics. Scientific American, 236(5), 119–127.",
        "Forman, G. (2008). Quantifying counts and costs via classification. Data Mining and Knowledge Discovery, 17(2), 164–206.",
        "Hoang, A., Rogers, B., & Ross, S. (2024). Coding client language in motivational interviewing with commitment strength. Proceedings of the Annual Conference on Computational Approaches to Health.",
        "Kashdan, T. B., Barrett, L. F., & McKnight, P. E. (2015). Unpacking emotion differentiation: Transforming unpleasant experience by perceiving distinctions in negativity. Current Directions in Psychological Science, 24(1), 10–16.",
        "Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal detection of changepoints with a linear computational cost. Journal of the American Statistical Association, 107(500), 1590–1598.",
        "Kumar, A., Sharma, R., & Patel, V. (2024). IC-AnnoMI: An LLM-augmented motivational interviewing corpus with bias measurement. Proceedings of the Conference on Language Resources and Evaluation.",
        "Magill, M., Apodaca, T. R., Borsari, B., Gaume, J., Hoadley, A., Gordon, R. E. F., Tonigan, J. S., & Moyers, T. (2018). A meta-analysis of motivational interviewing process: Technical, relational, and conditional process models of change. Journal of Consulting and Clinical Psychology, 86(2), 140–157.",
        "Miller, W. R., & Rollnick, S. (2013). Motivational interviewing: Helping people change (3rd ed.). Guilford Press.",
        "Moyers, T. B., Rowell, L. N., Manuel, J. K., Ernst, D., & Houck, J. M. (2016). The Motivational Interviewing Treatment Integrity code (MITI 4): Rationale, preliminary reliability and validity. Journal of Substance Abuse Treatment, 65, 36–42.",
        "Pennebaker, J. W., Mehl, M. R., & Niederhoffer, K. G. (2003). Psychological aspects of natural language use: Our words, our selves. Annual Review of Psychology, 54(1), 547–577.",
        "Saerens, M., Latinne, P., & Decaestecker, C. (2002). Adjusting the outputs of a classifier to new a priori probabilities: A simple procedure. Neural Computation, 14(1), 21–41.",
        "Tao, D., Lee, T., & Deng, J. (2024). CUEMPATHY: A counseling speech dataset for psychotherapy research. Proceedings of the International Symposium on Chinese Spoken Language Processing.",
        "Tavabi, L., Stefanov, K., Borsari, B., Woolley, J. D., Scherer, S., & Soleymani, M. (2021). Multimodal automatic coding of client behavior in motivational interviewing. Proceedings of the International Conference on Multimodal Interaction.",
        "Tsakalidis, A., Chim, J., Bilal, I. M., Zirikly, A., Atzil-Slonim, D., Nanni, F., Resnik, P., Gaur, M., Roy, K., Inkster, B., Leintz, J., & Liakata, M. (2022). Overview of the CLPsych 2022 shared task: Capturing moments of change in longitudinal user posts. Proceedings of the Eighth Workshop on Computational Linguistics and Clinical Psychology.",
        "Wu, Z., Balloccu, S., Kumar, V., Helaoui, R., Reiter, E., Recupero, D. R., & Riboni, D. (2022). AnnoMI: A dataset of expert-annotated counselling dialogues. Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing, 6177–6181.",
        "Wu, Z., Balloccu, S., Kumar, V., Helaoui, R., Reiter, E., Recupero, D. R., & Riboni, D. (2023). Creation, analysis and evaluation of AnnoMI, a dataset of expert-annotated counselling dialogues. Future Internet, 15(3), 110.",
        "Xiao, B., Imel, Z. E., Georgiou, P. G., Atkins, D. C., & Narayanan, S. S. (2015). “Rate my therapist”: Automated detection of empathy in drug and alcohol counseling via speech and language processing. PLOS ONE, 10(12), e0143055.",
        "Zhang, Y., Chen, L., & Wang, H. (2024). M3TCM: Multi-modal multi-task context model for utterance classification in motivational interviews. Proceedings of the International Conference on Affective Computing and Intelligent Interaction.",
    ]
    for i, r in enumerate(refs, 1):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = __import__("docx").shared.Pt(8)
        p.paragraph_format.line_spacing = 1.5
        p.paragraph_format.left_indent = __import__("docx").shared.Cm(1.0)
        p.paragraph_format.first_line_indent = __import__("docx").shared.Cm(-1.0)
        run = p.add_run(f"[{i}]. {r}")
        run.font.name = "Times New Roman"
        run.font.size = __import__("docx").shared.Pt(12)


def appendices(doc):
    page_break(doc)
    frontmatter_heading(doc, "APPENDICES", underline=False)

    section(doc, "Appendix 1", "Repository Structure")
    para(doc,
         "The complete implementation is version controlled and publicly "
         "available. The principal directories are as follows.")
    table(doc, "A1.1", "Repository layout",
          ["Path", "Contents"],
          [
              ["backend/app/nlp/", "Measurement, scoring, trajectory, attribution and model modules"],
              ["backend/app/api/", "REST endpoints"],
              ["backend/tests/", "Sixty-nine automated tests"],
              ["frontend/src/pages/", "Case list, trajectory view, session view, model comparison"],
              ["frontend/src/components/", "Trace and change-talk chart components"],
              ["ml/", "Training, benchmarking, reliability study and figure generation"],
              ["ml/artifacts/", "Trained models, result JSON, CSV exports and figures"],
          ],
          widths=[5.2, 9.4])

    section(doc, "Appendix 2", "Reproducing the Results")
    para(doc,
         "Every figure and table in Chapter 6 can be regenerated from "
         "committed code and a publicly downloadable corpus. The dataset is "
         "not committed to the repository.")
    para(doc, "curl -L -o ml/data/AnnoMI-full.csv \\",
         size=10, align="left", after=0, line=1.0)
    para(doc, "  https://raw.githubusercontent.com/uccollab/AnnoMI/main/AnnoMI-full.csv",
         size=10, align="left", after=6, line=1.0)
    for cmd, desc in [
        ("python ml/benchmark.py", "seven learners, two tasks, grouped CV"),
        ("python ml/confusion_all.py", "per-model confusion matrices"),
        ("python ml/time_models.py", "fit times on an idle machine"),
        ("python ml/reliability_study.py", "error propagation and correction"),
        ("python ml/binary_study.py", "change versus sustain framing"),
        ("python ml/figure_benchmark.py", "Figures 6.1 to 6.5"),
        ("python ml/figure_reliability.py", "Figure 6.6"),
        ("python ml/export_csv.py", "result tables as CSV"),
    ]:
        para(doc, f"{cmd:<34}# {desc}", size=10, align="left", after=2, line=1.0)

    section(doc, "Appendix 3", "Worked Example: One Session End to End")
    para(doc,
         "The following excerpt illustrates what the system reads and what it "
         "produces. The transcript is synthetic and is the kind of material "
         "used in the demonstration case.")
    for line in [
        "THERAPIST: How has the week been?",
        "CLIENT: Bad. He always does this and there is nothing I can do about",
        "  it. I had to just sit there and take it, like always.",
        "THERAPIST: It sounds like part of you wanted to say something and part",
        "  of you needed it to stay quiet.",
        "CLIENT: Yeah. I think I shut down because I was scared he would",
        "  escalate. I felt resentful, and also a bit relieved that it ended",
        "  quickly. That is odd to say out loud.",
        "THERAPIST: What would you want to say, if the fear was not in the way?",
        "CLIENT: Maybe that I need ten minutes before we talk. I have decided",
        "  I am going to try it on Tuesday.",
    ]:
        para(doc, line, size=10, align="left", after=1, line=1.0)
    para(doc, "", after=8)
    para(doc,
         "From this the measurement layer extracts the following, with each "
         "score traceable to the phrases that produced it.")
    table(doc, "A3.1", "Extracted dimension scores for the worked example",
          ["Dimension", "Score", "Evidence from the transcript"],
          [
              ["Self-agency", "0.57", "“I have decided”, “I am going to try” against “I had to just sit there”"],
              ["Future orientation", "0.49", "“on Tuesday”, “before we talk”"],
              ["Emotional granularity", "0.43", "“resentful”, “a bit relieved”, “scared”"],
              ["Problem ownership", "0.58", "“I shut down because” against “he always does this”"],
              ["Reflection depth", "0.38", "“I think it is because”, “that is odd to say out loud”"],
          ],
          widths=[3.6, 1.8, 9.2])
    para(doc,
         "The therapist turns are tagged by the hybrid scheme described in "
         "Section 5.4.3: the second turn as a complex reflection, the third as "
         "an open question. The client turn following each is compared with "
         "the one preceding it to produce a lift estimate. With no prior "
         "sessions for this case there is no personal baseline, so the index "
         "is reported at fifty by definition and the confidence is reduced — "
         "the system does not present a first session as though it carried "
         "comparative information.")

    section(doc, "Appendix 4", "Key Algorithms")
    subsection(doc, "A4.1", "Shrinkage of the Personal Baseline")
    for line in [
        "SHRINKAGE_K0 = 3.0",
        "MIN_SD = 0.015",
        "",
        "def build_baseline(feature_rows, baseline_k=2):",
        "    for d in DIMENSIONS:",
        "        vals = [row[d] for row in feature_rows[:baseline_k]]",
        "        n = len(vals)",
        "        own_sd = _sd(vals)",
        "        lam = n / (n + SHRINKAGE_K0)",
        "        pooled = lam * own_sd + (1 - lam) * COHORT_SD[d]",
        "        sds[d] = round(max(MIN_SD, pooled), 4)",
    ]:
        para(doc, line or " ", size=10, align="left", after=0, line=1.0)
    para(doc, "", after=10)

    subsection(doc, "A4.2", "Bias-Corrected Change-Talk Ratio")
    for line in [
        "def corrected_change_ratio(n_change, n_sustain):",
        "    total = n_change + n_sustain",
        "    if total == 0:",
        "        return {'ratio': None, 'lo': None, 'hi': None, 'wide': False}",
        "",
        "    cc, _cs, sc, _ss = _directional_matrix()",
        "    grid = [i / 400 for i in range(401)]",
        "    logliks = []",
        "    for p in grid:",
        "        theta = p * cc + (1.0 - p) * sc",
        "        theta = min(max(theta, 1e-9), 1.0 - 1e-9)",
        "        logliks.append(n_change * log(theta)",
        "                       + n_sustain * log1p(-theta))",
        "",
        "    peak = max(logliks)",
        "    weights = [exp(l - peak) for l in logliks]",
        "    mass = sum(weights)",
        "    mean = sum(p * w for p, w in zip(grid, weights)) / mass",
        "    # central 80% credible interval walked off the cumulative posterior",
    ]:
        para(doc, line or " ", size=10, align="left", after=0, line=1.0)
    para(doc, "", after=10)

    subsection(doc, "A4.3", "Conversation-Grouped Cross-Validation")
    for line in [
        "gkf = GroupKFold(n_splits=5)",
        "for train_idx, test_idx in gkf.split(X, y, groups=transcript_id):",
        "    # every fold's test set is whole conversations, unseen in training",
        "    pipe = build(model_name, n_classes)",
        "    pipe.fit(X.iloc[train_idx], y.iloc[train_idx])",
        "    pred = pipe.predict(X.iloc[test_idx])",
    ]:
        para(doc, line or " ", size=10, align="left", after=0, line=1.0)
    para(doc, "", after=10)

    section(doc, "Appendix 5", "Ethical Statement and Data Handling")
    para(doc,
         "No real clinical material was processed in this project. The corpus "
         "used consists of publicly available performed demonstrations. The "
         "system stores cases under pseudonymous codes only; there is no field "
         "for a name, date of birth, address or any other identifying detail, "
         "and the interface states this requirement to the user. Transcript "
         "directories, corpus files and database files are excluded from "
         "version control so that clinical material cannot be committed by "
         "accident.")
    para(doc,
         "The software licence carries an explicit use restriction: the system "
         "is not a medical device and must not be used to diagnose, to make "
         "treatment decisions, or to evaluate clinician competence. Any "
         "deployment on genuine session recordings would require institutional "
         "ethics approval, informed consent covering secondary analysis, and "
         "encryption at rest.", after=0)
