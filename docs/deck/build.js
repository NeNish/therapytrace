/**
 * TherapyTrace — final review deck.
 *
 * Palette is matched to the project's generated figures (paper F2F0EB,
 * ink 16181D, terracotta C9502A, green 2E7D5B). That is deliberate rather
 * than a default: every matplotlib figure in this project is rendered on
 * F2F0EB, so a white slide would show each one as a cream rectangle floating
 * on white. Matching the deck to the figures is what makes them look placed
 * rather than pasted.
 *
 * Charts that PowerPoint can draw are drawn natively rather than inserted as
 * pictures — they stay sharp at projector resolution and the panel can click
 * into the data.
 */

const pptxgen = require("pptxgenjs");
const path = require("path");
const fs = require("fs");

const FIG = path.join(__dirname, "fig");

const THEME = {
  name: "TherapyTrace",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "16181D",   // ink
    lt1: "F2F0EB",   // paper
    dk2: "3A3F47",   // muted ink
    lt2: "E4E0D7",   // paper, one step down
    accent1: "C9502A", // terracotta — the accent
    accent2: "2E7D5B", // green — "good" / verified
    accent3: "8E2C4E", // claret — problems
    accent4: "6B7C82",
    accent5: "B47C14",
    accent6: "4C6B8A",
    hlink: "C9502A",
    folHlink: "8E2C4E",
  },
};

const INK = "16181D", PAPER = "F2F0EB", MUTED = "5A6068";
const ACC = "C9502A", VER = "2E7D5B", BAD = "8E2C4E", SOFT = "E4E0D7";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";          // 13.3 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.author = "Nishant E S";
pres.title = "TherapyTrace - Final Review";

const C = pres.SchemeColor;

// ---------------------------------------------------------------- layouts

pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: INK },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.9, y: 2.3, w: 11.5, h: 1.9,
        fontSize: 40, bold: true, color: PAPER, fontFace: "Cambria", align: "left", valign: "bottom" } } },
    { placeholder: { options: { name: "body", type: "body", x: 0.9, y: 4.35, w: 11.5, h: 1.6,
        fontSize: 16, color: "C9C4BA", align: "left", valign: "top" } } },
  ],
});

pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: PAPER },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.75, y: 0.45, w: 11.8, h: 0.85,
        fontSize: 32, bold: true, color: INK, fontFace: "Cambria", align: "left", valign: "middle" } } },
    { placeholder: { options: { name: "body", type: "body", x: 0.75, y: 1.45, w: 11.8, h: 5.3,
        fontSize: 15, color: INK, valign: "top" } } },
  ],
  slideNumber: { x: 12.55, y: 6.95, fontSize: 10, color: MUTED, fontFace: "Calibri" },
});

pres.defineSlideMaster({
  title: "TITLE_ONLY",
  background: { color: PAPER },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.75, y: 0.45, w: 11.8, h: 0.85,
        fontSize: 32, bold: true, color: INK, fontFace: "Cambria", align: "left", valign: "middle" } } },
  ],
  slideNumber: { x: 12.55, y: 6.95, fontSize: 10, color: MUTED, fontFace: "Calibri" },
});

pres.defineSlideMaster({
  title: "DIVIDER",
  background: { color: INK },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.9, y: 2.9, w: 11.5, h: 1.4,
        fontSize: 36, bold: true, color: PAPER, fontFace: "Cambria", align: "left", valign: "middle" } } },
    { placeholder: { options: { name: "body", type: "body", x: 0.9, y: 4.3, w: 11.5, h: 1.2,
        fontSize: 16, color: "C9C4BA", valign: "top" } } },
  ],
});

// ------------------------------------------------------------- components

let fignum = 0;

/** Big number with a caption under it. */
function stat(slide, x, y, w, value, label, color) {
  slide.addText(value, {
    x, y, w, h: 0.95, isTextBox: true, margin: 0,
    fontSize: 46, bold: true, color: color || ACC, fontFace: "Cambria",
    align: "left", valign: "bottom", objectName: `stat-${label.slice(0, 18)}`,
  });
  slide.addText(label, {
    x, y: y + 0.95, w, h: 0.75, isTextBox: true, margin: 0,
    fontSize: 12.5, color: MUTED, align: "left", valign: "top",
  });
}

/** A tinted card. No edge stripes — a tint and a shadow set it apart. */
function card(slide, x, y, w, h, fill) {
  slide.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.06,
    fill: { color: fill || SOFT },
    line: { color: fill || SOFT, width: 0 },
    shadow: { type: "outer", angle: 90, blur: 7, offset: 1, color: "B9B3A6", opacity: 0.35 },
    objectName: "card",
  });
}

function bodyBullets(slide, items, opts = {}) {
  slide.addText(
    items.map((t, i) => ({
      text: typeof t === "string" ? t : t.text,
      options: {
        bullet: true, breakLine: i < items.length - 1,
        bold: typeof t === "object" && t.bold,
        color: (typeof t === "object" && t.color) || INK,
      },
    })),
    Object.assign({
      x: 0.85, y: 1.6, w: 11.6, h: 4.9, isTextBox: true,
      fontSize: 16, color: INK, paraSpaceAfter: 11, valign: "top",
    }, opts)
  );
}

function caption(slide, text, y) {
  slide.addText(text, {
    x: 0.85, y: y || 6.5, w: 11.6, h: 0.5, isTextBox: true, valign: "top", margin: 0,
    fontSize: 11.5, italic: true, color: MUTED, align: "left",
  });
}

function figure(slide, file, x, y, w, h) {
  slide.addImage({ path: path.join(FIG, file), x, y, w, h, objectName: file });
}

// =========================================================== 1. TITLE
pres.addSection({ title: "Opening" });
let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Opening" });
s.addText("TherapyTrace", {
  x: 0.9, y: 1.5, w: 11.5, h: 0.7, isTextBox: true, valign: "top", margin: 0,
  fontSize: 18, color: ACC, charSpacing: 3, bold: true,
});
s.addText("Measuring change in a person,\nnot classifying a sentence", { placeholder: "title" });
s.addText([
  { text: "Final Project Review  ·  BCSE497J", options: { breakLine: true, bold: true } },
  { text: "Nishant E S  ·  School of Computer Science and Engineering", options: { breakLine: true } },
  { text: "Vellore Institute of Technology, Chennai", options: {} },
], { placeholder: "body" });
s.addNotes("Frame the whole talk in one line: existing work classifies utterances, we measure a person over time. Everything else follows from that distinction.");

// =========================================================== 2. PROBLEM
pres.addSection({ title: "Problem" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Problem" });
s.addText("Therapy is made of language, and measured without it", { placeholder: "title" });
s.addText([
  { text: "A client attends weekly. Every eight to twelve weeks a questionnaire is administered, and the difference between two administrations stands in for everything that happened in between.", options: { breakLine: true } },
  { text: "The fifty minutes of speech that constitute the treatment leave no quantitative trace at all.", options: { bold: true } },
], { x: 0.85, y: 1.55, w: 7.1, h: 2.2, isTextBox: true, valign: "top", fontSize: 16.5, color: INK, paraSpaceAfter: 10 });

card(s, 8.35, 1.55, 4.2, 4.3);
s.addText("Why it persists", {
  x: 8.7, y: 1.8, w: 3.5, h: 0.4, isTextBox: true, valign: "top", margin: 0,
  fontSize: 14, bold: true, color: INK,
});
s.addText([
  { text: "Manual process coding (MITI) needs trained raters and hours per session", options: { bullet: true, breakLine: true } },
  { text: "So it stays inside research and never reaches clinical practice", options: { bullet: true, breakLine: true } },
  { text: "Between measurements the clinician has only memory and notes", options: { bullet: true } },
], { x: 8.7, y: 2.3, w: 3.5, h: 3.3, isTextBox: true, valign: "top", fontSize: 13, color: INK, paraSpaceAfter: 9 });

stat(s, 0.85, 3.95, 3.4, "8–12", "weeks between measurements");
stat(s, 4.3, 3.95, 3.4, "0", "measurements in between", BAD);
s.addNotes("Do not rush this slide. The whole project only makes sense once the panel accepts that the measurement gap is real and that language is the obvious thing to measure.");

// =========================================================== 3. GAP
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Problem" });
s.addText("Nine papers on our dataset. All of them classify utterances", { placeholder: "title" });
s.addText([
  "Wu et al. 2022/2023 — created the corpus, baseline classifiers",
  "Cohen et al. 2022/2024 — RoBERTa, LLM-automated MITI scoring",
  "Tavabi et al. 2021 — multimodal behaviour classification",
  "Hoang, Rogers & Ross 2024 — client coding with commitment strength",
  "Kumar et al. 2024 — IC-AnnoMI, LLM-augmented corpus",
  "M3TCM 2024 — multimodal context model",
  "EMMI 2025 · VERVE 2023 · Consistent Client Simulation 2025",
].map((t, i, a) => ({ text: t, options: { bullet: true, breakLine: i < a.length - 1 } })),
  { x: 0.85, y: 1.55, w: 7.0, h: 3.6, isTextBox: true, valign: "top", fontSize: 14, color: INK, paraSpaceAfter: 8 });

card(s, 8.2, 1.55, 4.35, 3.6, INK);
s.addText("The unit of analysis", {
  x: 8.55, y: 1.85, w: 3.7, h: 0.4, isTextBox: true, valign: "top", margin: 0,
  fontSize: 13, bold: true, color: ACC,
});
s.addText([
  { text: "Every one of them:\nthe utterance", options: { breakLine: true, bold: true, color: PAPER, fontSize: 19 } },
  { text: "\nWhat a clinician asks:\nthe person, over time", options: { bold: true, color: PAPER, fontSize: 19 } },
], { x: 8.55, y: 2.35, w: 3.7, h: 2.5, isTextBox: true, valign: "top", fontSize: 19, color: PAPER, paraSpaceAfter: 6 });

s.addText("None asks whether a person is changing across sessions. None normalises against an individual's own baseline. None estimates whether a therapist move works differently depending on the state the client is already in.",
  { x: 0.85, y: 5.4, w: 11.7, h: 1.0, isTextBox: true, valign: "top", fontSize: 15, bold: true, color: INK });
s.addNotes("This is the novelty slide. The claim is specific and checkable: the unit of analysis in the literature is the utterance. Do not overclaim beyond that.");

// =========================================================== 4. SYSTEM
pres.addSection({ title: "System" });
s = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: "System" });
s.addText("What we built", { placeholder: "title" });
figure(s, "TherapyTrace_Architecture.png", 1.55, 1.45, 10.2, 4.99);
caption(s, "Rule-based measurement is deterministic, so every score traces to specific phrases. The supervised layer is additive — the system runs without it. Optional modalities enter as bounded modifiers.", 6.55);
s.addNotes("Point at the three properties in the caption. The additive supervised layer matters because a fresh clone has no trained artifacts and must still work.");

// =========================================================== 5. CONTRIBUTIONS
s = pres.addSlide({ masterName: "DIVIDER", sectionTitle: "System" });
s.addText("Three contributions", { placeholder: "title" });
s.addText("Each is measured, and each has a limitation stated alongside it", { placeholder: "body" });
[["01", "Within-person scoring", "Each session normalised against that client's own earlier sessions, with shrinkage for small samples", 0.9],
 ["02", "An honest negative result", "Seven classifiers are statistically indistinguishable — the ceiling is the task, not the model", 5.05],
 ["03", "Error propagation measured", "Classifier error biases the displayed aggregate by +0.25; we quantify and correct it", 9.2],
].forEach(([n, t, d, x]) => {
  s.addText(n, { x, y: 5.35, w: 3.6, h: 0.45, isTextBox: true, valign: "top", margin: 0, fontSize: 13, bold: true, color: ACC, charSpacing: 2 });
  s.addText(t, { x, y: 5.75, w: 3.6, h: 0.4, isTextBox: true, valign: "top", margin: 0, fontSize: 15, bold: true, color: PAPER });
  s.addText(d, { x, y: 6.18, w: 3.6, h: 0.95, isTextBox: true, valign: "top", margin: 0, fontSize: 11.5, color: "A9A49A" });
});
s.addNotes("Signpost. The third is the one to dwell on later — it is the publishable contribution.");

// =========================================================== 6. METHOD
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "System" });
s.addText("Scored against yourself, not against a population", { placeholder: "title" });
s.addText([
  { text: "A reserved client and an expansive one differ in vocabulary and sentence length for reasons that have nothing to do with therapy. Comparing either to a population average measures personality as much as progress.", options: { breakLine: true } },
], { x: 0.85, y: 1.5, w: 11.6, h: 0.95, isTextBox: true, valign: "top", fontSize: 15.5, color: INK });

card(s, 0.85, 2.6, 5.6, 1.5);
s.addText("z  =  ( x − μ_own ) / σ_own", {
  x: 1.1, y: 2.85, w: 5.1, h: 0.45, isTextBox: true, valign: "top", margin: 0,
  fontSize: 19, bold: true, color: INK, fontFace: "Cambria",
});
s.addText("Each dimension, against this client's own mean and spread", {
  x: 1.1, y: 3.35, w: 5.1, h: 0.6, isTextBox: true, valign: "top", margin: 0, fontSize: 12, color: MUTED });

card(s, 6.9, 2.6, 5.65, 1.5);
s.addText("TPI  =  50 + 10 × Σ w·z", {
  x: 7.15, y: 2.85, w: 5.15, h: 0.45, isTextBox: true, valign: "top", margin: 0,
  fontSize: 19, bold: true, color: ACC, fontFace: "Cambria",
});
s.addText("50 means: this person talking the way this person usually talks", {
  x: 7.15, y: 3.35, w: 5.15, h: 0.6, isTextBox: true, valign: "top", margin: 0, fontSize: 12, color: MUTED });

s.addText("Five process dimensions", { x: 0.85, y: 4.35, w: 11.6, h: 0.4, isTextBox: true, valign: "top", margin: 0, fontSize: 14, bold: true, color: INK });
s.addTable([
  [{ text: "Self-agency", options: { bold: true } }, { text: "Future orientation", options: { bold: true } },
   { text: "Emotional granularity", options: { bold: true } }, { text: "Problem ownership", options: { bold: true } },
   { text: "Reflection depth", options: { bold: true } }],
  ["“it just happens to me”\n→ “I decided to”", "replaying the argument\n→ naming a plan",
   "“bad” → “resentful, and\na bit relieved”", "“he always does this”\n→ “I shut down when”",
   "description → “I think\nit is because”"],
], { x: 0.85, y: 4.8, w: 11.6, colW: [2.32, 2.32, 2.32, 2.32, 2.32], fontSize: 11, color: INK,
     fill: { color: PAPER }, border: { type: "solid", color: "CFC9BD", pt: 0.75 }, valign: "top", margin: 6 });
s.addNotes("Shrinkage: sigma is pooled toward a cohort prior with lambda = n/(n+3), so an early session is reportable without being unstable. Mention if asked, do not spend a slide on it.");

// =========================================================== 7. DATA
pres.addSection({ title: "Evaluation" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Evaluation" });
s.addText("The dataset, described accurately", { placeholder: "title" });
stat(s, 0.85, 1.55, 2.6, "133", "annotated conversations");
stat(s, 3.65, 1.55, 2.6, "9,699", "unique utterances");
stat(s, 6.45, 1.55, 2.6, "44", "presenting topics");
stat(s, 9.25, 1.55, 3.3, "23", "deliberately poor practice", BAD);

card(s, 0.85, 3.85, 11.7, 1.55, INK);
s.addText([
  { text: "AnnoMI is performed demonstration material, not clinical recordings.", options: { bold: true, color: ACC, breakLine: true, fontSize: 17 } },
  { text: "It is expert-annotated, and it is MI training video. Every number in this talk is a claim about this corpus and nothing more.", options: { color: PAPER, fontSize: 14 } },
], { x: 1.2, y: 4.15, w: 11.0, h: 1.0, isTextBox: true, valign: "top", paraSpaceAfter: 6 });

s.addText("Say this before anyone asks. A panel member who knows the corpus will catch “real therapy data” immediately, and then doubt everything else.",
  { x: 0.85, y: 5.65, w: 11.7, h: 0.6, isTextBox: true, valign: "top", fontSize: 13, italic: true, color: MUTED });
s.addNotes("This slide exists to protect every other claim. Deliver it as confidence, not apology.");

// =========================================================== 8. PROTOCOL
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Evaluation" });
s.addText("The one methodological decision that matters most", { placeholder: "title" });
s.addText("Splits are grouped by conversation, never by utterance.",
  { x: 0.85, y: 1.5, w: 11.6, h: 0.5, isTextBox: true, valign: "top", fontSize: 20, bold: true, color: ACC });
s.addText("Utterances inside one conversation share a client, a therapist, a topic and a speaking style. Split at the utterance level and a model can memorise a conversation in training and recognise it at test time — which inflates every metric reported.",
  { x: 0.85, y: 2.1, w: 11.6, h: 1.0, isTextBox: true, valign: "top", fontSize: 15, color: INK });

card(s, 0.85, 3.3, 5.6, 2.5, SOFT);
s.addText("What we do", { x: 1.15, y: 3.55, w: 5.0, h: 0.35, isTextBox: true, valign: "top", margin: 0, fontSize: 13, bold: true, color: VER });
s.addText([
  "25% of conversations held out before any model is fitted",
  "5-fold GroupKFold on transcript_id for selection",
  "Held-out conversations scored once, for every model",
].map((t, i, a) => ({ text: t, options: { bullet: true, breakLine: i < a.length - 1 } })),
  { x: 1.15, y: 3.95, w: 5.0, h: 1.7, isTextBox: true, valign: "top", fontSize: 13, color: INK, paraSpaceAfter: 8 });

card(s, 6.9, 3.3, 5.65, 2.5, SOFT);
s.addText("Why it limits comparison", { x: 7.2, y: 3.55, w: 5.05, h: 0.35, isTextBox: true, valign: "top", margin: 0, fontSize: 13, bold: true, color: BAD });
s.addText("Published figures on this corpus vary widely and the split protocol is frequently unstated. Results under conversation-grouped splits are not directly comparable with results whose protocol is unknown — in either direction.",
  { x: 7.2, y: 3.95, w: 5.05, h: 1.7, isTextBox: true, valign: "top", fontSize: 13, color: INK });
s.addNotes("If asked why our numbers look lower than a published 0.8: this is the answer, and give it without claiming the other work cheated. We cannot verify their protocol.");

// =========================================================== 9. MODELS (native chart)
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Evaluation" });
s.addText("Seven classifiers, one identical feature matrix", { placeholder: "title" });
s.addChart(pres.ChartType.bar, [{
  name: "Held-out macro-F1",
  labels: ["Majority class", "Naive Bayes", "Logistic Regression", "Linear SVM",
           "Random Forest", "XGBoost", "LightGBM", "MLP (128)"],
  values: [0.238, 0.457, 0.489, 0.493, 0.500, 0.485, 0.425, 0.462],
}], {
  x: 0.85, y: 1.5, w: 7.3, h: 4.75,
  barDir: "bar", chartColors: [ACC],
  showTitle: false, showLegend: false,
  showValue: true, dataLabelPosition: "outEnd", dataLabelColor: INK,
  dataLabelFontSize: 11, dataLabelFontFace: "+mn-lt", dataLabelFormatCode: "0.000",
  catAxisLabelColor: INK, catAxisLabelFontSize: 11.5, catAxisLabelFontFace: "+mn-lt",
  valAxisLabelColor: MUTED, valAxisLabelFontSize: 10, valAxisLabelFontFace: "+mn-lt",
  valAxisMinVal: 0, valAxisMaxVal: 0.62, valGridLine: { color: "DAD5CA", size: 0.75 },
  catGridLine: { style: "none" }, barGapWidthPct: 45,
  objectName: "client-macro-f1",
});
card(s, 8.55, 1.5, 4.0, 4.75, SOFT);
s.addText("Client talk type", { x: 8.85, y: 1.75, w: 3.4, h: 0.35, isTextBox: true, valign: "top", margin: 0, fontSize: 13, bold: true, color: INK });
s.addText([
  { text: "1,656 utterances from 34 conversations the models never saw.", options: { breakLine: true } },
  { text: "\nEvery learner sees the same features: TF-IDF word 1–2 grams, character 3–5 grams, and 12 lexicon features.", options: { breakLine: true } },
  { text: "\nSo a difference in score is attributable to the learner and nothing else.", options: { bold: true } },
], { x: 8.85, y: 2.2, w: 3.4, h: 3.8, isTextBox: true, valign: "top", fontSize: 12.5, color: INK, paraSpaceAfter: 4 });
s.addNotes("The majority-class floor is 0.238 macro-F1 but 0.554 ACCURACY. If anyone says 54% is low, that is the answer: a model that does nothing scores 55%.");

// =========================================================== 10. NEGATIVE RESULT
s = pres.addSlide({ masterName: "DIVIDER", sectionTitle: "Evaluation" });
s.addText("The gaps are inside the noise", { placeholder: "title" });
s.addText("On client talk type, a paired t-test cannot separate the best learner from any of the other six", { placeholder: "body" });
stat(s, 0.9, 5.3, 3.6, "0.051", "spread across all seven");
stat(s, 4.75, 5.3, 3.6, "0.043", "fold-to-fold SD, one model");
stat(s, 8.6, 5.3, 3.9, "p ≥ 0.068", "smallest against any rival");
[0, 1, 2].forEach(i => {
  s.addShape(pres.ShapeType.rect, { x: 0.9 + i * 3.85, y: 5.22, w: 3.6, h: 0.03,
    fill: { color: ACC }, line: { width: 0 }, objectName: `rule-${i}` });
});
s.addNotes("Report this as a finding, not a disappointment. It establishes that the ceiling is the task and licenses everything that follows.");

// =========================================================== 11. PER-CLASS (native chart)
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Evaluation" });
s.addText("Where the averages hide the problem", { placeholder: "title" });
s.addChart(pres.ChartType.bar, [{
  name: "Sustain-talk recall",
  labels: ["Logistic Regression", "Linear SVM", "XGBoost", "Naive Bayes",
           "Random Forest", "LightGBM", "MLP (128)"],
  values: [0.335, 0.290, 0.276, 0.210, 0.159, 0.117, 0.072],
}], {
  x: 0.85, y: 1.5, w: 7.3, h: 4.4,
  barDir: "bar", chartColors: [ACC],
  showTitle: false, showLegend: false,
  showValue: true, dataLabelPosition: "outEnd", dataLabelColor: INK,
  dataLabelFontSize: 11, dataLabelFontFace: "+mn-lt", dataLabelFormatCode: "0.000",
  catAxisLabelColor: INK, catAxisLabelFontSize: 11.5, catAxisLabelFontFace: "+mn-lt",
  valAxisLabelColor: MUTED, valAxisLabelFontSize: 10, valAxisLabelFontFace: "+mn-lt",
  valAxisMinVal: 0, valAxisMaxVal: 0.42, valGridLine: { color: "DAD5CA", size: 0.75 },
  catGridLine: { style: "none" }, barGapWidthPct: 45,
  objectName: "sustain-recall",
});
card(s, 8.55, 1.5, 4.0, 4.4, SOFT);
s.addText("Sustain talk is the weak class", { x: 8.85, y: 1.75, w: 3.4, h: 0.6, isTextBox: true, valign: "top", margin: 0, fontSize: 13, bold: true, color: INK });
s.addText([
  { text: "Recall varies almost five-fold across the seven.", options: { breakLine: true } },
  { text: "\nThe model that macro-F1 selected sits near the bottom of that range.", options: { breakLine: true } },
  { text: "\nSo the deployment criterion is sustain recall, not macro-F1 — which selects logistic regression, the model we ship.", options: { bold: true } },
], { x: 8.85, y: 2.4, w: 3.4, h: 3.3, isTextBox: true, valign: "top", fontSize: 12.5, color: INK, paraSpaceAfter: 4 });
s.addNotes("This is the bridge into the key finding. Under-detected sustain talk is exactly what biases the display.");

// =========================================================== 12. KEY FINDING
pres.addSection({ title: "Key finding" });
s = pres.addSlide({ masterName: "DIVIDER", sectionTitle: "Key finding" });
s.addText("What a weak classifier does to the number on the screen", { placeholder: "title" });
s.addText("The user never sees a label. They see an aggregate — and its bias was nobody's number", { placeholder: "body" });
card(s, 0.9, 5.25, 5.6, 1.6, "23262D");
s.addText([
  { text: "For a client genuinely 50/50\nbetween change and sustain talk,", options: { breakLine: true, color: "A9A49A", fontSize: 13 } },
  { text: "the screen would read 0.75", options: { bold: true, color: ACC, fontSize: 21 } },
], { x: 1.2, y: 5.5, w: 5.0, h: 1.1, isTextBox: true, valign: "top", paraSpaceAfter: 4 });
card(s, 6.9, 5.25, 5.6, 1.6, "23262D");
s.addText([
  { text: "And the bias does not shrink\nwith longer sessions —", options: { breakLine: true, color: "A9A49A", fontSize: 13 } },
  { text: "0.752 at 10 turns, 0.752 at 120", options: { bold: true, color: PAPER, fontSize: 18 } },
], { x: 7.2, y: 5.5, w: 5.0, h: 1.1, isTextBox: true, valign: "top", paraSpaceAfter: 4 });
s.addNotes("Pause here. This is the most important slide in the deck. It is systematic, not random — that is why more data does not help.");

// =========================================================== 13. ATTENUATION (native chart)
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Key finding" });
s.addText("A client who changes a lot looks like one who barely does", { placeholder: "title" });
// One line chart with three series, not a combo of three single-series
// charts: in a combo, chartColors applies per chart spec rather than per
// series, so all three lines came out the same grey — which destroyed the
// only thing this slide is for.
const RATIO_X = ["0.10", "0.25", "0.40", "0.50", "0.60", "0.75", "0.90"];
s.addChart(
  pres.ChartType.line,
  [
    { name: "Perfect display", labels: RATIO_X,
      values: [0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90] },
    { name: "As built (uncorrected)", labels: RATIO_X,
      values: [0.632, 0.681, 0.725, 0.751, 0.777, 0.813, 0.845] },
    { name: "With bias correction", labels: RATIO_X,
      values: [0.229, 0.352, 0.481, 0.572, 0.654, 0.768, 0.858] },
  ],
  {
    x: 0.85, y: 1.5, w: 7.6, h: 4.5,
    chartColors: ["9AA0A6", ACC, "4C6B8A"],
    showTitle: false, showLegend: true, legendPos: "b", legendColor: INK, legendFontSize: 11,
    legendFontFace: "+mn-lt",
    lineSize: 3, lineSmooth: false, showValue: false,
    catAxisTitle: "the client's true change-talk ratio", showCatAxisTitle: true,
    catAxisTitleColor: MUTED, catAxisTitleFontSize: 11, catAxisTitleFontFace: "+mn-lt",
    valAxisTitle: "what the website displays", showValAxisTitle: true,
    valAxisTitleColor: MUTED, valAxisTitleFontSize: 11, valAxisTitleFontFace: "+mn-lt",
    catAxisLabelColor: MUTED, catAxisLabelFontSize: 10, catAxisLabelFontFace: "+mn-lt",
    valAxisLabelColor: MUTED, valAxisLabelFontSize: 10, valAxisLabelFontFace: "+mn-lt",
    valAxisMinVal: 0, valAxisMaxVal: 1,
    valGridLine: { color: "DAD5CA", size: 0.75 }, catGridLine: { style: "none" },
    objectName: "attenuation",
  }
);
card(s, 8.8, 1.5, 3.75, 4.5, SOFT);
s.addText("Slope", { x: 9.1, y: 1.8, w: 3.15, h: 0.35, isTextBox: true, valign: "top", margin: 0, fontSize: 13, bold: true, color: INK });
s.addText([
  { text: "0.27", options: { bold: true, color: ACC, fontSize: 34, breakLine: true } },
  { text: "uncorrected\n", options: { color: MUTED, fontSize: 12, breakLine: true } },
  { text: "\n0.80", options: { bold: true, color: "4C6B8A", fontSize: 34, breakLine: true } },
  { text: "with correction", options: { color: MUTED, fontSize: 12 } },
], { x: 9.1, y: 2.2, w: 3.15, h: 2.6, isTextBox: true, valign: "top", paraSpaceAfter: 2 });
s.addText("A true range of 0.10–0.90 compresses to 0.63–0.85 on screen — the exact comparison the system exists to make.",
  { x: 9.1, y: 4.85, w: 3.15, h: 1.0, isTextBox: true, valign: "top", margin: 0, fontSize: 11.5, color: INK });
s.addNotes("Simulated over 20,000 sessions of 40 directional turns, corrupting each turn by the deployed model's own held-out confusion matrix.");

// =========================================================== 14. CORRECTION
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Key finding" });
s.addText("The textbook correction failed, and the data said so", { placeholder: "title" });

card(s, 0.85, 1.5, 5.6, 2.45, SOFT);
s.addText("Adjusted classify-and-count", { x: 1.15, y: 1.75, w: 5.0, h: 0.35, isTextBox: true, valign: "top", margin: 0, fontSize: 14, bold: true, color: BAD });
s.addText("Invert the misclassification matrix algebraically. Correct in principle, and fine on long sequences.",
  { x: 1.15, y: 2.15, w: 5.0, h: 0.7, isTextBox: true, valign: "top", margin: 0, fontSize: 13, color: INK });
s.addText("Our sessions carry 7–15 directional turns. At that size the solve lands outside [0,1] and must be clamped: 8 of 12 demo sessions pinned to exactly 0.00 or 1.00.",
  { x: 1.15, y: 2.85, w: 5.0, h: 0.95, isTextBox: true, valign: "top", margin: 0, fontSize: 13, bold: true, color: INK });

card(s, 6.9, 1.5, 5.65, 2.45, SOFT);
s.addText("Bayesian posterior", { x: 7.2, y: 1.75, w: 5.05, h: 0.35, isTextBox: true, valign: "top", margin: 0, fontSize: 14, bold: true, color: VER });
s.addText("Treat the observed count as binomial in θ(p) and compute a posterior over the true ratio under a flat prior.",
  { x: 7.2, y: 2.15, w: 5.05, h: 0.7, isTextBox: true, valign: "top", margin: 0, fontSize: 13, color: INK });
s.addText("Inside [0,1] by construction, shrinks toward the centre when evidence is thin, and yields a credible interval the interface shows.",
  { x: 7.2, y: 2.85, w: 5.05, h: 0.95, isTextBox: true, valign: "top", margin: 0, fontSize: 13, bold: true, color: INK });

card(s, 0.85, 4.2, 11.7, 1.35, INK);
s.addText([
  { text: "A step function between two extremes, asserted with total confidence, is a worse failure than the bias it removes.", options: { bold: true, color: PAPER, breakLine: true, fontSize: 15 } },
  { text: "Found by running the estimator on real sessions, not by reasoning about its definition.", options: { color: ACC, fontSize: 13.5 } },
], { x: 1.2, y: 4.45, w: 11.0, h: 0.95, isTextBox: true, valign: "top", paraSpaceAfter: 5 });

s.addText("Even corrected: a +0.10 session-to-session change moves the display the right way 58% of the time at 40 turns, against 50% for a coin. The series supports a trend, not a single-pair judgement — and the interface says so.",
  { x: 0.85, y: 5.75, w: 11.7, h: 0.8, isTextBox: true, valign: "top", fontSize: 12.5, italic: true, color: MUTED });
s.addNotes("The honesty of the last line is the point. We did not oversell the fix.");

// =========================================================== 15. STATE-CONDITIONAL
pres.addSection({ title: "Further results" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Further results" });
s.addText("The same therapist move, two different effects", { placeholder: "title" });
s.addTable([
  [{ text: "Therapist move", options: { bold: true } }, { text: "Client stuck", options: { bold: true } },
   { text: "Middling", options: { bold: true } }, { text: "Client moving", options: { bold: true } },
   { text: "p", options: { bold: true } }],
  ["Question", "0.268", "0.412", "0.479", "< 0.001"],
  ["Reflection", "0.260", "0.382", "0.455", "0.001"],
  ["Therapist input", "0.253", "0.265", "0.296", "n.s."],
  ["Other", "0.244", "0.331", "0.415", "—"],
], { x: 0.85, y: 1.55, w: 7.5, colW: [2.3, 1.3, 1.3, 1.3, 1.3], fontSize: 13, color: INK,
     fill: { color: PAPER }, border: { type: "solid", color: "CFC9BD", pt: 0.75 },
     align: "center", valign: "middle", margin: 7, rowH: 0.42 });
s.addText("Probability of change talk following each move, by the client's state beforehand. 1,815 triples across 123 conversations.",
  { x: 0.85, y: 4.0, w: 7.5, h: 0.6, isTextBox: true, valign: "top", fontSize: 11.5, italic: true, color: MUTED });

card(s, 8.75, 1.55, 3.8, 4.3, SOFT);
s.addText("What it means", { x: 9.05, y: 1.8, w: 3.2, h: 0.35, isTextBox: true, valign: "top", margin: 0, fontSize: 13, bold: true, color: INK });
s.addText([
  { text: "Questions and reflections roughly double their association when the client is already moving. Therapist input barely shifts.", options: { breakLine: true } },
  { text: "\nThese are associations, not causal effects. A client already moving produces more change talk whatever the therapist does — part of this is confounding by indication.", options: { breakLine: true, color: BAD } },
  { text: "\nWhat it does support: the unconditional average of 0.369 averages over two very different situations, and is the wrong number to act on.", options: { bold: true } },
], { x: 9.05, y: 2.25, w: 3.2, h: 3.4, isTextBox: true, valign: "top", fontSize: 11.5, color: INK, paraSpaceAfter: 4 });
s.addNotes("Volunteer the confounding before the panel raises it. The weaker claim still stands and is still useful.");

// =========================================================== 16. THE SYSTEM
s = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: "Further results" });
s.addText("The model reaches the screen, corrected", { placeholder: "title" });
figure(s, "ui_trace_crop.png", 0.85, 1.4, 6.3, 4.69);
card(s, 7.5, 1.4, 5.05, 4.69, SOFT);
s.addText("Two independent measures", { x: 7.8, y: 1.65, w: 4.45, h: 0.35, isTextBox: true, valign: "top", margin: 0, fontSize: 13, bold: true, color: INK });
s.addText([
  { text: "Upper: the progress index against this client's own baseline corridor, with a detected regime change marked.", options: { breakLine: true } },
  { text: "\nLower: the bias-corrected change-talk ratio with credible intervals, sharing the x-axis but on its own scale.", options: { breakLine: true } },
  { text: "\nThey agree at r = 0.82 — one from the lexicon, one from a model trained on expert annotations.", options: { bold: true, breakLine: true } },
  { text: "\nOn demo data, so part of that agreement reflects the generator.", options: { italic: true, color: MUTED } },
], { x: 7.8, y: 2.1, w: 4.45, h: 3.85, isTextBox: true, valign: "top", fontSize: 12, color: INK, paraSpaceAfter: 4 });
s.addNotes("Deliberately two panels, not two lines on one chart: the index is centred at 50 and the ratio is 0 to 1, so a crossing point would be an artefact of the axes.");

// =========================================================== 17. BUILD
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Further results" });
s.addText("What was actually built", { placeholder: "title" });
stat(s, 0.85, 1.6, 2.7, "25", "modules");
stat(s, 3.75, 1.6, 2.7, "69", "automated tests", VER);
stat(s, 6.65, 1.6, 2.7, "7", "classifiers compared");
stat(s, 9.55, 1.6, 3.0, "10", "result figures");

s.addText("Tests that encode requirements, not just behaviour", { x: 0.85, y: 3.5, w: 11.7, h: 0.4, isTextBox: true, valign: "top", margin: 0, fontSize: 15, bold: true, color: INK });
bodyBullets(s, [
  "The body-language module can never emit an emotion word — expression-to-emotion mapping is not reliable (Barrett et al. 2019)",
  "The generated session note contains no number absent from its inputs",
  "The change-talk series must reach the trajectory endpoint — the regression that whole module exists to prevent",
  "The system runs with the ML stack absent, falling back to the rule-based layer",
], { y: 3.95, h: 2.3, fontSize: 14 });
s.addText("FastAPI · SQLAlchemy 2.0 · React 18 + Vite · scikit-learn, XGBoost, LightGBM · MediaPipe, librosa",
  { x: 0.85, y: 6.3, w: 11.7, h: 0.45, isTextBox: true, valign: "top", fontSize: 12, italic: true, color: MUTED });
s.addNotes("The safety tests are worth naming individually. They show the constraints are enforced rather than promised.");

// =========================================================== 18. LIMITATIONS
pres.addSection({ title: "Closing" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Closing" });
s.addText("Limitations, stated before anyone asks", { placeholder: "title" });
const lims = [
  ["No real longitudinal clinical data", "The central claim is about change across sessions. No public corpus has repeated sessions per client, so that layer runs on synthetic cases with known trajectories. Pipeline validation, not clinical evidence."],
  ["The corpus is performed, not clinical", "AnnoMI is MI training video. Transfer to genuine sessions is unmeasured."],
  ["Modest classification performance", "0.489 macro-F1 on the client task. Corrected, but not eliminated — single-session judgements are not supported."],
  ["No transformer baseline", "A fine-tuned encoder is the obvious next comparison; no model-hub access in the build environment."],
];
lims.forEach(([t, d], i) => {
  const y = 1.55 + i * 1.28;
  card(s, 0.85, y, 11.7, 1.12, SOFT);
  s.addText(t, { x: 1.15, y: y + 0.13, w: 11.1, h: 0.33, isTextBox: true, valign: "top", margin: 0, fontSize: 14, bold: true, color: BAD });
  s.addText(d, { x: 1.15, y: y + 0.48, w: 11.1, h: 0.58, isTextBox: true, valign: "top", margin: 0, fontSize: 12.5, color: INK });
});
s.addNotes("Deliver these as confidence. A reviewer who finds a limitation unaided doubts everything else; one who hears it from you trusts the rest more.");

// =========================================================== 19. FUTURE + PUBLICATION
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Closing" });
s.addText("Next steps, and where this can be published", { placeholder: "title" });

card(s, 0.85, 1.55, 5.6, 4.6, SOFT);
s.addText("Future work", { x: 1.15, y: 1.82, w: 5.0, h: 0.35, isTextBox: true, valign: "top", margin: 0, fontSize: 14, bold: true, color: INK });
s.addText([
  "Secure a longitudinal corpus — CUEMPATHY (4 sessions × 39 dyads), the Alexander Street collection, DAIC-WOZ for PHQ-8 outcomes",
  "Add a transformer tier under the same grouped protocol",
  "Build the two-stage directional classifier so the binary gain reaches production",
  "Extend the prevalence correction to therapist behaviour",
  "Ethics approval for a small prospective study",
].map((t, i, a) => ({ text: t, options: { bullet: true, breakLine: i < a.length - 1 } })),
  { x: 1.15, y: 2.25, w: 5.0, h: 3.7, isTextBox: true, valign: "top", fontSize: 12.5, color: INK, paraSpaceAfter: 9 });

card(s, 6.9, 1.55, 5.65, 4.6, SOFT);
s.addText("Publication", { x: 7.2, y: 1.82, w: 5.05, h: 0.35, isTextBox: true, valign: "top", margin: 0, fontSize: 14, bold: true, color: INK });
s.addText([
  { text: "Classification on AnnoMI is crowded — nine-plus papers. Even at 0.80 we would be the tenth, so the contribution cannot be a classifier.", options: { breakLine: true } },
  { text: "\nThe measurement finding is publishable, and it needs an imperfect classifier to be interesting:", options: { breakLine: true } },
  { text: "\n“Weak classifiers can support session-level clinical measures, but only after prevalence correction.”", options: { bold: true, color: ACC, breakLine: true } },
  { text: "\nRealistic: a CLPsych-style workshop, or an IEEE conference. A clinical journal needs the real corpus first.", options: {} },
], { x: 7.2, y: 2.25, w: 5.05, h: 3.7, isTextBox: true, valign: "top", fontSize: 12.5, color: INK, paraSpaceAfter: 4 });
s.addNotes("An IEEE-format manuscript of the measurement contribution is already drafted.");

// =========================================================== 20. CLOSING
s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Closing" });
s.addText("The take-away", {
  x: 0.9, y: 1.5, w: 11.5, h: 0.6, isTextBox: true, valign: "top", margin: 0,
  fontSize: 18, color: ACC, charSpacing: 3, bold: true,
});
s.addText("Measure what your model's error does\nto the number a person will act on —\nbefore you show it to them", { placeholder: "title" });
s.addText([
  { text: "Applied here, that question turned a system that would have displayed confident misinformation into one that shows an honest estimate with its uncertainty attached.", options: { breakLine: true } },
  { text: "\nThe classifier is still mediocre. The interface is now truthful about it.", options: { bold: true, color: PAPER } },
], { x: 0.9, y: 4.6, w: 10.5, h: 1.6, isTextBox: true, valign: "top", fontSize: 15, color: "C9C4BA", paraSpaceAfter: 6 });
s.addText("Thank you  ·  Questions", {
  x: 0.9, y: 6.4, w: 11.5, h: 0.5, isTextBox: true, valign: "top", margin: 0,
  fontSize: 14, bold: true, color: ACC, charSpacing: 2,
});
s.addNotes("Close on the methodological point rather than on the system. It is the thing that generalises beyond this project.");

// ---------------------------------------------------------------- write
//
// The majority-class floor on slide 9 is a reference point, not an eighth
// competitor, so it must not wear the models' colour. pptxgenjs applies
// `chartColors` per SERIES only — a single-series bar chart gets one fill for
// every bar — so the per-point override goes in as <c:dPt>, which is native
// OOXML the library simply does not expose.
//
// Schema order inside <c:ser> is idx, order, tx, spPr, invertIfNegative,
// pictureOptions, dPt*, dLbls, … so it is inserted immediately after
// <c:invertIfNegative/>, and the element that follows is asserted.
const DPT_FLOOR =
  '<c:dPt><c:idx val="0"/><c:invertIfNegative val="0"/><c:bubble3D val="0"/>' +
  '<c:spPr><a:solidFill><a:srgbClr val="9AA0A6"/></a:solidFill><a:effectLst/></c:spPr>' +
  "</c:dPt>";

async function greyTheFloorBar(pptxPath) {
  const JSZip = require(require.resolve("jszip", { paths: [require.resolve("pptxgenjs")] }));
  const zip = await JSZip.loadAsync(fs.readFileSync(pptxPath));
  const names = Object.keys(zip.files).filter((n) => /^ppt\/charts\/chart\d+\.xml$/.test(n));
  let patched = 0;
  for (const n of names) {
    const xml = await zip.file(n).async("string");
    if (!xml.includes("Majority class")) continue;      // only the macro-F1 chart
    const anchor = '<c:invertIfNegative val="0"/>';
    const at = xml.indexOf(anchor);
    if (at === -1) throw new Error(`${n}: <c:invertIfNegative/> not found`);
    const cut = at + anchor.length;
    if (!xml.slice(cut).startsWith("<c:dLbls>")) {
      throw new Error(`${n}: expected <c:dLbls> after invertIfNegative, found ${xml.slice(cut, cut + 24)}`);
    }
    zip.file(n, xml.slice(0, cut) + DPT_FLOOR + xml.slice(cut));
    patched++;
  }
  if (patched !== 1) throw new Error(`expected to patch exactly 1 chart, patched ${patched}`);
  fs.writeFileSync(pptxPath, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}

(async () => {
  const out = path.join(__dirname, "TherapyTrace_Final_Review.pptx");
  await pres.writeFile({ fileName: out });
  const { applyTheme } = require(
    "/root/.claude/skills/synced/4dff1370-2256-4c45-a8f0-39cdaad81460_b8862463-0927-4928-81e9-5a70c9825586/pptx/scripts/apply_theme.js"
  );
  await applyTheme(out, THEME);
  await greyTheFloorBar(out);
  console.log("wrote " + out);
})();
