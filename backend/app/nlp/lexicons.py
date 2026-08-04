"""
Lexical resources for TherapyTrace.

Everything here is deliberately transparent: a clinician or reviewer can read
these lists and understand exactly why a session scored the way it did. The
transformer tier (see `encoders.py`) is optional and layered on top; the
lexicon tier alone is enough to run the whole system.

Sources of inspiration (re-implemented, not copied):
  - LIWC-style pronoun / cognitive-mechanism categories
  - Motivational Interviewing change-talk vs sustain-talk markers
  - Emotion-differentiation work on granularity of affect labels
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# 1. SELF-AGENCY
#    Does the client present themself as an actor or as something acted upon?
# --------------------------------------------------------------------------

AGENCY_ACTIVE = {
    "i chose", "i choose", "i decided", "i decide", "i made", "i make",
    "i started", "i start", "i stopped", "i stop", "i told", "i asked",
    "i said no", "i set", "i planned", "i plan", "i will", "i'll",
    "i can", "i could", "i want to", "i'm going to", "i am going to",
    "i tried", "i try", "i managed", "i handled", "i took", "i decided to",
    "i pushed", "i reached out", "i signed up", "i booked", "i left",
    "i stood up", "i let myself", "i allowed myself", "i gave myself",
    "i chose to", "i decided that", "my decision", "my choice", "up to me",
    "i took responsibility", "i decided against",
}

AGENCY_PASSIVE = {
    "i had to", "i have to", "i must", "made me", "makes me", "making me",
    "forced me", "forces me", "i can't", "i cannot", "i couldn't",
    "there's nothing i", "there is nothing i", "no choice", "no option",
    "it just happened", "it just happens", "happened to me", "happens to me",
    "i ended up", "i got dragged", "out of my hands", "out of my control",
    "i was told", "they wouldn't let", "he wouldn't let", "she wouldn't let",
    "i'm stuck", "i am stuck", "i have no", "it's not up to me",
    "nothing i can do", "i'm not allowed", "i just react", "i can't help it",
}

# --------------------------------------------------------------------------
# 2. FUTURE ORIENTATION
#    Is the client oriented toward what comes next, or looping on what was?
# --------------------------------------------------------------------------

FUTURE_MARKERS = {
    "will", "i'll", "we'll", "going to", "gonna", "plan", "plans",
    "planning", "next week", "next time", "next month", "tomorrow",
    "soon", "upcoming", "intend", "intending", "aim to", "hope to",
    "looking forward", "by friday", "by monday", "eventually",
    "from now on", "starting", "the plan is", "i'd like to", "i would like to",
    "goal", "goals", "step", "steps", "try to", "want to",
}

PAST_MARKERS = {
    "was", "were", "had", "did", "used to", "back then", "last week",
    "last year", "last month", "yesterday", "when i was", "always did",
    "never did", "growing up", "as a child", "years ago", "ago",
    "kept", "would always", "at the time", "before that", "had been",
}

# Rumination markers: past-oriented AND repetitive/counterfactual
RUMINATION_MARKERS = {
    "should have", "shouldn't have", "if only", "why did i", "why didn't i",
    "over and over", "keep thinking", "kept thinking", "can't stop thinking",
    "again and again", "replay", "replaying", "go over it", "went over it",
    "what if i had", "i keep going back",
}

# --------------------------------------------------------------------------
# 3. EMOTIONAL GRANULARITY
#    Tiered by specificity. Tier 1 is undifferentiated affect ("bad"),
#    tier 3 is a precisely named blend ("ambivalent", "resentful").
# --------------------------------------------------------------------------

EMOTION_TIER_1 = {
    "bad", "good", "fine", "okay", "ok", "upset", "down", "off", "weird",
    "stressed", "meh", "blah", "low", "rough", "heavy", "numb", "fed up",
    "not great", "not good", "shit", "awful", "terrible", "great",
}

EMOTION_TIER_2 = {
    "sad", "angry", "happy", "scared", "afraid", "anxious", "worried",
    "tired", "lonely", "hurt", "annoyed", "excited", "calm", "nervous",
    "guilty", "ashamed", "embarrassed", "frustrated", "relieved", "proud",
    "hopeful", "disappointed", "jealous", "confused", "overwhelmed", "bored",
}

EMOTION_TIER_3 = {
    "resentful", "resentment", "ambivalent", "ambivalence", "wistful",
    "indignant", "contemptuous", "defensive", "dejected", "despondent",
    "apprehensive", "exasperated", "humiliated", "mortified", "remorseful",
    "vindicated", "validated", "invalidated", "dismissed", "unseen",
    "unheard", "protective", "possessive", "envious", "contempt", "dread",
    "grief", "bereft", "yearning", "longing", "tenderness", "affection",
    "compassion", "self-compassion", "reluctant", "wary", "guarded",
    "bittersweet", "conflicted", "torn", "grateful", "gratitude", "regret",
    "loathing", "disgust", "revulsion", "betrayed", "abandoned", "smothered",
    "suffocated", "crowded", "restless", "agitated", "hollow", "raw",
    "brittle", "tender", "exposed", "vulnerable", "safe", "settled",
    "steady", "grounded", "unmoored", "adrift",
}

# Hedged / mixed emotion constructions are a strong granularity signal
MIXED_EMOTION_PATTERNS = [
    r"\b(part of me|a part of me)\b.{0,60}\b(but|and|while|another part)\b",
    r"\b(both)\b.{0,40}\b(and)\b",
    r"\b(at the same time)\b",
    r"\b(angry|sad|happy|scared|relieved|proud)\b.{0,30}\b(but also|and also|and yet)\b",
]

# --------------------------------------------------------------------------
# 4. PROBLEM OWNERSHIP
#    Internal attribution vs external blame. Note: high ownership is NOT the
#    same as self-blame — see SELF_BLAME below, which is scored separately and
#    penalised, because "it's all my fault" is not therapeutic ownership.
# --------------------------------------------------------------------------

OWNERSHIP_INTERNAL = {
    "my part in", "my part of", "i contributed", "i played a part",
    "i realise i", "i realize i", "i notice i", "i noticed i",
    "i tend to", "i have a habit", "my pattern", "my reaction",
    "the way i react", "the way i respond", "i shut down", "i withdraw",
    "i lash out", "i avoid", "i push people", "i do this thing where",
    "i can see how i", "i own that", "that's on me", "i was defensive",
    "i wasn't listening", "i escalated", "my responsibility",
}

OWNERSHIP_EXTERNAL = {
    "he always", "she always", "they always", "he never", "she never",
    "they never", "because of him", "because of her", "because of them",
    "it's his fault", "it's her fault", "it's their fault", "if he would",
    "if she would", "if they would", "he makes", "she makes", "they make",
    "he won't", "she won't", "they won't", "everyone else", "nobody ever",
    "no one ever", "the problem is he", "the problem is she",
    "the problem is they", "it's not me", "i didn't do anything",
}

SELF_BLAME = {
    "it's all my fault", "i ruin", "i ruined everything", "i'm the problem",
    "i am the problem", "i always mess", "i always screw", "i'm useless",
    "i am useless", "i'm worthless", "i am worthless", "i'm broken",
    "i am broken", "i deserve this", "i'm a failure", "i am a failure",
    "there's something wrong with me", "i hate myself",
}

# --------------------------------------------------------------------------
# 5. REFLECTION DEPTH
#    Insight / causal reasoning / tentativeness / self-distancing.
# --------------------------------------------------------------------------

INSIGHT_TERMS = {
    "realise", "realised", "realize", "realized", "understand", "understood",
    "notice", "noticed", "noticing", "see now", "makes sense", "aware",
    "awareness", "pattern", "patterns", "connection", "connected", "link",
    "linked", "reminds me of", "similar to when", "the same thing as",
    "underneath", "beneath", "at the root", "what's really", "actually about",
    "occurred to me", "dawned on me", "figured out", "insight", "perspective",
    "stepping back", "step back", "looking at it differently", "reframe",
}

CAUSAL_TERMS = {
    "because", "since", "therefore", "so that", "which is why", "the reason",
    "leads to", "led to", "causes", "caused", "results in", "resulted in",
    "as a result", "consequently", "that's why", "due to", "makes sense that",
    "explains", "effect", "impact",
}

TENTATIVE_TERMS = {
    "maybe", "perhaps", "might", "may be", "possibly", "i wonder",
    "i think", "i guess", "sort of", "kind of", "somewhat", "probably",
    "it seems", "seemed", "i suppose", "could be", "one way of looking",
    "not sure but", "i'd say",
}

ABSOLUTIST_TERMS = {
    "always", "never", "everyone", "nobody", "no one", "everything",
    "nothing", "completely", "totally", "absolutely", "entirely",
    "constantly", "every single time", "all the time", "definitely",
    "must", "impossible", "forever",
}

# --------------------------------------------------------------------------
# AUXILIARY SIGNALS (reported alongside, not part of the composite index)
# --------------------------------------------------------------------------

HOPELESSNESS_TERMS = {
    "hopeless", "pointless", "no point", "what's the point", "give up",
    "gave up", "giving up", "never get better", "never change",
    "won't change", "nothing works", "nothing helps", "trapped", "stuck forever",
    "can't see a way", "no way out", "burden", "why bother", "too late",
}

SOLUTION_FOCUS_TERMS = {
    "i could try", "what if i", "one option", "another way", "instead of",
    "next time i", "i'm going to try", "a small step", "start with",
    "practise", "practice", "experiment", "test it", "give it a go",
    "work on", "working on", "strategy", "approach", "manage it",
}

# Self-distancing: talking about oneself in second/third person or by name
SELF_DISTANCING_PATTERNS = [
    r"\byou (?:just|kind of|sort of)? ?(?:start|get|feel|end up|find yourself)\b",
    r"\bpart of me\b",
    r"\bthe part of me that\b",
    r"\bthat side of me\b",
    r"\bwhen someone\b",
    r"\bif a friend of mine\b",
]

FIRST_PERSON_SINGULAR = {"i", "me", "my", "mine", "myself", "i'm", "i'll", "i've", "i'd"}
OTHER_PERSON = {
    "he", "she", "they", "him", "her", "them", "his", "their", "we", "us",
    "our", "you", "your",
}

# --------------------------------------------------------------------------
# THERAPIST INTERVENTION TAXONOMY
#    Ordered: the first pattern that matches wins, so put the specific ones
#    above the general ones.
# --------------------------------------------------------------------------

THERAPIST_INTERVENTIONS: list[tuple[str, list[str]]] = [
    ("complex_reflection", [
        r"^(?:so|it sounds like|what i'm hearing|i'm hearing|it seems like|"
        r"there's a sense|part of you|you're caught between|underneath that)",
        r"\bpart of you\b.{0,60}\band part of you\b",
        r"\bas if\b.{0,40}\b(you|it)\b",
    ]),
    ("simple_reflection", [
        r"^(?:you feel|you felt|you're feeling|you said|you're saying|"
        r"you were|that was|that sounds)\b",
        r"^(?:mm|mhm|right),? (?:so )?you\b",
    ]),
    ("open_question", [
        r"^(?:what|how|tell me|say more|walk me through|describe|"
        r"i'm curious|help me understand|in what way|where does)\b.*\??",
    ]),
    ("closed_question", [
        r"^(?:did|do|does|is|are|was|were|have|has|had|can|could|would|"
        r"will|should|any|and you|so you)\b.*\?",
    ]),
    ("affirmation", [
        r"\b(that took|that's a lot|it makes sense that you|"
        r"i can see how hard|you've been carrying|that's real work|"
        r"you noticed that yourself|you did that)\b",
    ]),
    ("challenge", [
        r"\b(and yet|but earlier you said|how does that fit with|"
        r"i notice you|last week you told me|what stops you|"
        r"i want to push back|is that the whole picture)\b",
    ]),
    ("psychoeducation", [
        r"\b(what often happens|research|the body|nervous system|"
        r"a lot of people|this is called|one model|the way anxiety works|"
        r"what we know about)\b",
    ]),
    ("directive", [
        r"^(?:try|do|write|practise|practice|notice|between now and|"
        r"i'd like you to|your homework|let's agree|i want you to)\b",
        r"\b(before next session|this week i'd like)\b",
    ]),
    ("summary", [
        r"^(?:so to pull|let me pull|to recap|if i sum up|"
        r"we've covered|bringing that together|so today)\b",
    ]),
    ("minimal_encourager", [
        r"^(?:mm+|mhm+|hmm+|right|okay|ok|i see|go on|yeah)[.\s]*$",
    ]),
]
