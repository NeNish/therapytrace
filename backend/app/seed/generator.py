"""
Synthetic longitudinal cases.

Why this exists: essentially no public corpus contains many sessions from the
*same* client, which is exactly what a within-person index needs. So the
demo — and the first validation experiment — runs on generated cases where the
true trajectory is known by construction.

Each case has a latent "process level" p in [0, 1] that follows a programmed
shape (improving, plateauing, relapsing, late-breakthrough). Client turns are
sampled from stage-tagged pools weighted by p, and a synthetic PHQ-9 is emitted
as 24 - 18p + noise so the validation endpoint has something to correlate
against.

This data is clearly marked synthetic everywhere it surfaces. It is for
pipeline development and sanity checks, never for clinical claims.
"""

from __future__ import annotations

import random

# --------------------------------------------------------------------------
# client utterance pools, tagged by process stage
# --------------------------------------------------------------------------

LOW = [
    "Bad, I guess. Same as always. He always does this and there is nothing I can do about it.",
    "I don't know. It just happens. One minute it's fine and then it isn't and I'm stuck in it again.",
    "Honestly, what's the point. Nothing works. I've tried everything and nothing helps.",
    "I had to go. I have no choice, I never do. Everyone else decides and I just deal with it.",
    "It's all my fault anyway. I ruin everything I touch. I'm the problem, I always have been.",
    "I feel bad. Just bad, all the time. I can't describe it any better than that.",
    "She never listens. Not once. If she would just stop, everything would be fine.",
    "I keep going over it again and again. Why did I say that. I should have said something else.",
    "Work was terrible and then home was terrible. I was upset. That's it really.",
    "Nothing changed. Nothing ever changes. It's pointless talking about it.",
]

MID = [
    "It was okay. I was annoyed on Tuesday, and I think I was also a bit relieved when it was over.",
    "I noticed I went quiet when he raised his voice. I'm not sure why I do that.",
    "Maybe I avoid it because it's easier. I don't know if that's the whole picture though.",
    "I was frustrated, and underneath that I think I was scared he'd leave. That surprised me.",
    "I tried to say something on Thursday. It didn't come out right but I said it.",
    "Part of me wanted to go and part of me wanted to stay home. I couldn't decide.",
    "I think I shut down because I was overwhelmed. It reminds me of how it was at home growing up.",
    "I could try telling her earlier in the day instead of at night when we're both tired.",
    "I felt guilty afterwards, which seems like a lot for something that small.",
    "It's not only him. I can see how I escalate it when I get defensive.",
]

HIGH = [
    "I decided not to go, and I told her why. I felt anxious saying it and also steadier afterwards.",
    "I noticed the pattern this time. When I feel dismissed I withdraw, and then I read the silence as proof I was right.",
    "I was resentful, and honestly a bit ashamed of being resentful. Both at once. That's new for me to say out loud.",
    "My part in it is that I wait until I'm furious and then I deliver it like an accusation. That's on me.",
    "Next week I'm going to ask for the ten minutes before we talk. I've decided that.",
    "I realised the thing I'm actually afraid of isn't the argument, it's what it would mean if she agreed with me.",
    "I chose to stay in the conversation even though I wanted to leave. It was uncomfortable and I stayed.",
    "I set a boundary about Sundays. She wasn't happy, and I didn't collapse, which I think is the point.",
    "I can see now that the exhaustion and the irritability are the same thing wearing different clothes.",
    "I've been practising the thing we talked about. It works about half the time, which is more than none.",
]

THERAPIST_TURNS = [
    "How has the week been?",
    "It sounds like part of you wanted to speak and part of you needed it to stay quiet.",
    "What was that like for you?",
    "Say more about the moment it turned.",
    "You noticed that yourself, which is not a small thing.",
    "And yet last week you told me you never say anything. How does that fit?",
    "What often happens is that the body reacts before the thought arrives.",
    "Between now and next week, I'd like you to notice when it starts, not fix it.",
    "So to pull that together, the pattern shows up when you feel dismissed.",
    "Mm.",
    "Did that surprise you?",
    "You feel caught.",
    "Where does that go in your body?",
    "What stops you from saying it earlier?",
    "Tell me about Thursday.",
]


def _trajectory(shape: str, n: int) -> list[float]:
    if shape == "improving":
        return [0.25 + 0.55 * (i / (n - 1)) for i in range(n)]
    if shape == "plateauing":
        return [0.30 + 0.35 * min(1.0, i / 3) for i in range(n)]
    if shape == "relapsing":
        return [0.30 + 0.40 * (i / 4) if i <= 4 else 0.70 - 0.35 * ((i - 4) / (n - 5)) for i in range(n)]
    if shape == "late_breakthrough":
        return [0.30 if i < n * 0.6 else 0.30 + 0.50 * ((i - n * 0.6) / (n * 0.4)) for i in range(n)]
    return [0.5] * n


def _sample_client_turn(p: float, rng: random.Random) -> str:
    r = rng.random()
    if p < 0.4:
        pool = LOW if r < 0.7 else MID
    elif p < 0.65:
        pool = MID if r < 0.6 else (LOW if r < 0.85 else HIGH)
    else:
        pool = HIGH if r < 0.65 else (MID if r < 0.92 else LOW)
    return rng.choice(pool)


def build_session_transcript(p: float, rng: random.Random, n_exchanges: int = 20) -> str:
    lines = ["THERAPIST: How has the week been?"]
    for i in range(n_exchanges):
        turn = _sample_client_turn(p, rng)
        if rng.random() < 0.35:
            turn += " " + _sample_client_turn(p, rng)
        lines.append(f"CLIENT: {turn}")
        if i < n_exchanges - 1:
            lines.append(f"THERAPIST: {rng.choice(THERAPIST_TURNS)}")
    lines.append("THERAPIST: So to pull that together, we'll pick this up next week.")
    lines.append(f"CLIENT: {_sample_client_turn(p, rng)}")
    return "\n".join(lines)


DEMO_CASES = [
    ("CL-014", "improving", 12, "Relationship conflict, low assertiveness", "CBT", "TH-02"),
    ("CL-027", "plateauing", 10, "Generalised anxiety", "CBT", "TH-02"),
    ("CL-031", "relapsing", 11, "Recurrent depression", "Person-centred", "TH-05"),
    ("CL-046", "late_breakthrough", 12, "Grief, avoidance", "Integrative", "TH-05"),
]


def generate_case(shape: str, n_sessions: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    levels = _trajectory(shape, n_sessions)
    out = []
    for i, p in enumerate(levels, start=1):
        jitter = rng.uniform(-0.045, 0.045)
        p_noisy = max(0.05, min(0.95, p + jitter))
        phq9 = max(0.0, min(27.0, 24 - 18 * p_noisy + rng.uniform(-2, 2)))
        out.append({
            "session_number": i,
            "transcript": build_session_transcript(p_noisy, rng),
            "latent_level": round(p_noisy, 3),
            "phq9": round(phq9, 1),
        })
    return out
