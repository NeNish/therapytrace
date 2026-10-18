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
# Per-persona utterance pools — each case uses different language so the
# NLP evidence lexicon produces case-specific signals and the recommended
# questions are genuinely different per patient.
# --------------------------------------------------------------------------

# CL-014  Relationship conflict, low assertiveness  (CBT / TH-02)
CL014_LOW = [
    "He spoke over me again and I just went silent. I never know what to say so I end up saying nothing.",
    "She made the decision without asking me. She always does that. I don't have any say in it.",
    "I couldn't speak up at the time. I froze. It's like I'm not even allowed to have an opinion.",
    "They expect me to just agree with everything. If I push back, there's an atmosphere for days.",
    "I never set limits with anyone. I can't. People will think I'm difficult and then I'll be alone.",
    "The meeting happened without me being consulted. I was furious but I just nodded and walked away.",
    "I know what I want to say and then when the moment comes I swallow it. Every single time.",
    "They talked over my head and I let them. I always let them. I don't know why I never defend myself.",
]
CL014_MID = [
    "I did say something this time. It came out differently than I intended, but I said it.",
    "I noticed I was holding my breath when she raised her voice. I'm starting to see that pattern.",
    "Maybe I wait until I'm too angry and then the words come out all wrong. That might be mine to own.",
    "I tried to say what I needed before it escalated. It felt weird to speak first but I did it.",
    "I think I avoid confrontation because I'm more scared of the silence after than the argument itself.",
    "I'm starting to separate what's mine from what's hers. It's blurry but less blurry than before.",
    "I noticed I didn't apologise when I hadn't done anything wrong. That's different for me.",
]
CL014_HIGH = [
    "I set a boundary about the weekends. She didn't like it. I held it anyway and I didn't collapse.",
    "I said what I needed without waiting for permission. It felt strange and also right.",
    "I realised the thing I've been afraid of isn't disagreement—it's losing the relationship over it.",
    "I told him directly that I need time to think before I answer. He paused. That was new.",
    "I chose not to fill the silence with an apology. I let it sit. And it was fine.",
    "I can see now that assertiveness isn't aggression. I've been treating them as the same thing.",
    "I asked for what I needed this week, three separate times. I kept count. That's a record.",
]

# CL-027  Generalised anxiety  (CBT / TH-02)
CL027_LOW = [
    "I spent the whole week catastrophising. Every email felt like something was about to go wrong.",
    "I can't stop thinking about what might happen. The what-ifs just go around and around and never stop.",
    "My body was tense the whole week. My shoulders, my chest. It didn't let up at all.",
    "I cancelled the dinner because I was too anxious to go. I knew I'd ruin it with my worry.",
    "I checked my phone constantly in case something bad had happened. I couldn't stop even when I wanted to.",
    "Nothing felt safe. I kept scanning for the next problem. My mind won't switch off.",
    "I didn't sleep properly. I kept waking up with my heart racing and the same thought loops starting again.",
    "Every time I started to relax, a new worry came in. What if this breaks, what if they don't come back.",
]
CL027_MID = [
    "I noticed when the worry started — there was a trigger, a message I misread. That was new to notice.",
    "I caught myself catastrophising on Tuesday and I actually paused. I didn't stop it but I saw it.",
    "I was anxious at the meeting but I stayed in the room and it passed. That surprised me.",
    "I tried the breathing thing. It didn't cure anything but the physical tension did ease a little.",
    "I could see that the worst-case scenario I was building hadn't happened. Most of it never does.",
    "I pushed back on one of the anxious thoughts this week. It felt awkward but I tried.",
    "I told a colleague I was feeling overwhelmed instead of pretending I was fine. That was hard.",
]
CL027_HIGH = [
    "I caught the worry loop early and chose not to feed it. I did something physical instead and it worked.",
    "I went to the event even though I was anxious. I stayed the whole time. I decided that mattered.",
    "I can see the anxiety is a prediction machine and most of its predictions are wrong. That's useful.",
    "I set one boundary around checking emails after nine. It held for four days, which is progress.",
    "I told myself the uncertainty is uncomfortable but it isn't dangerous. And I believed it slightly more this time.",
    "I noticed my body relaxing in a situation that usually tightens me up. I stayed curious about that.",
    "I'm practising sitting with the discomfort instead of escaping it. Half the time it's working.",
]

# CL-031  Recurrent depression  (Person-centred / TH-05)
CL031_LOW = [
    "I didn't get out of bed until noon. There's just no point. Nothing is going to change.",
    "I'm worthless. I can't do anything right. I've always been like this and I always will be.",
    "Everything is flat. Food doesn't taste of anything. I watch television and feel nothing.",
    "I don't see the point of doing the things I used to enjoy. They just feel empty now.",
    "I've been through this before and nothing helped long-term. It always comes back. It always will.",
    "I'm exhausted before I even start the day. The weight of it is enormous and I don't know how to carry it.",
    "I cancelled everything this week. I told them I was ill. It was easier than explaining.",
    "I feel like a burden. I'm tired of being a problem that other people have to manage.",
]
CL031_MID = [
    "I got outside on Wednesday. Not far, but outside. I noticed the air which is something.",
    "I'm tired, but a different kind of tired. Not quite as hollow as before.",
    "I reached out to one person this week. I didn't tell them everything but I said I was struggling.",
    "I can see the depression is doing the thinking when I believe I'm worthless. They're not the same thing.",
    "I cooked something for myself. It felt unnecessary but I did it anyway and ate it.",
    "Part of me knows this has lifted before. I'm trying to hold onto that even when it doesn't feel true.",
    "I managed one thing on the list. The rest stayed undone but one thing happened.",
]
CL031_HIGH = [
    "I did the thing I've been putting off for two weeks. It took twenty minutes and the dread was bigger than the task.",
    "I noticed a moment of genuine pleasure on Thursday. I wrote it down because I didn't want to forget.",
    "I can separate my thoughts from facts now more often than not. That distinction is becoming real.",
    "I told my sister what's been happening. She didn't try to fix it. That was what I needed.",
    "I went for a walk every morning this week except one. That's different from how it's been.",
    "I chose to let myself feel sad without deciding it meant something catastrophic. That's new.",
    "The episode was shorter this time. I came back faster. I think the work we're doing is part of that.",
]

# CL-046  Grief, avoidance  (Integrative / TH-05)
CL046_LOW = [
    "I haven't touched any of his things. I can't go into that room. It's just too much.",
    "I don't feel anything anymore. I think I've gone numb. It's easier than the alternative.",
    "I keep busy so I don't have to think about it. If I stop moving, it all comes in.",
    "People say time heals it. They say that because they don't know what else to say.",
    "I'm not grieving, I'm just managing. There's a difference. Managing means not breaking down.",
    "I took a different route home so I didn't have to pass the spot. I'll do that as long as I need to.",
    "I cancelled the memorial gathering. I said I wasn't ready but the truth is I never want to be ready for it.",
    "I look at photographs and feel nothing. That frightens me more than the sadness would.",
]
CL046_MID = [
    "I looked at one photograph this week. Just one. I cried and then I put it away. That felt like progress.",
    "I went into the room for a few minutes. I didn't do anything. But I went in.",
    "I let myself feel it for about ten minutes on Tuesday instead of turning the television on.",
    "A friend talked about him and instead of changing the subject I let the conversation happen.",
    "I'm starting to see that avoiding it isn't the same as surviving it. There's something I'm not letting myself do.",
    "I opened the box but didn't look inside. I sat with it for a while and then put it back. That's different.",
    "I feel guilty for moments when I forget. And then I feel guilty for the guilt. It's exhausting.",
]
CL046_HIGH = [
    "I went through some of his things. I kept what mattered and let go of what didn't. I made that decision.",
    "I let myself grieve for an hour without trying to stop it or make it productive. That was new.",
    "I talked about him at dinner. I said what I remembered. People cried. It felt more real and also more bearable.",
    "I can hold both things now — that he's gone and that I'm still here. I couldn't do that before.",
    "I went past the spot. I didn't stop. But I didn't avoid it either. That was a choice I made.",
    "The numbness has been coming less. In its place there's sadness, which is actually harder and also more honest.",
    "I'm starting to understand that grieving him doesn't mean I'm leaving him behind. That changes something.",
]

# Fallback generic pools
LOW = CL014_LOW
MID = CL014_MID
HIGH = CL014_HIGH

# Map case code to persona-specific pools
PERSONA_POOLS: dict[str, tuple[list, list, list]] = {
    "CL-014": (CL014_LOW, CL014_MID, CL014_HIGH),
    "CL-027": (CL027_LOW, CL027_MID, CL027_HIGH),
    "CL-031": (CL031_LOW, CL031_MID, CL031_HIGH),
    "CL-046": (CL046_LOW, CL046_MID, CL046_HIGH),
}

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


def _sample_client_turn(
    p: float, rng: random.Random,
    low_pool: list | None = None,
    mid_pool: list | None = None,
    high_pool: list | None = None,
) -> str:
    """Sample one client turn from persona-specific pools (fallback to defaults)."""
    pool_low = low_pool or LOW
    pool_mid = mid_pool or MID
    pool_high = high_pool or HIGH
    r = rng.random()
    if p < 0.4:
        pool = pool_low if r < 0.7 else pool_mid
    elif p < 0.65:
        pool = pool_mid if r < 0.6 else (pool_low if r < 0.85 else pool_high)
    else:
        pool = pool_high if r < 0.65 else (pool_mid if r < 0.92 else pool_low)
    return rng.choice(pool)


def build_session_transcript(
    p: float, rng: random.Random, n_exchanges: int = 20,
    low_pool: list | None = None,
    mid_pool: list | None = None,
    high_pool: list | None = None,
) -> str:
    lines = ["THERAPIST: How has the week been?"]
    for i in range(n_exchanges):
        turn = _sample_client_turn(p, rng, low_pool, mid_pool, high_pool)
        if rng.random() < 0.35:
            turn += " " + _sample_client_turn(p, rng, low_pool, mid_pool, high_pool)
        lines.append(f"CLIENT: {turn}")
        if i < n_exchanges - 1:
            lines.append(f"THERAPIST: {rng.choice(THERAPIST_TURNS)}")
    lines.append("THERAPIST: So to pull that together, we'll pick this up next week.")
    lines.append(f"CLIENT: {_sample_client_turn(p, rng, low_pool, mid_pool, high_pool)}")
    return "\n".join(lines)


DEMO_CASES = [
    ("CL-014", "improving", 12, "Relationship conflict, low assertiveness", "CBT", "TH-02"),
    ("CL-027", "plateauing", 10, "Generalised anxiety", "CBT", "TH-02"),
    ("CL-031", "relapsing", 11, "Recurrent depression", "Person-centred", "TH-05"),
    ("CL-046", "late_breakthrough", 12, "Grief, avoidance", "Integrative", "TH-05"),
]


def generate_case(shape: str, n_sessions: int, seed: int, case_code: str = "") -> list[dict]:
    rng = random.Random(seed)
    levels = _trajectory(shape, n_sessions)
    # Look up persona-specific pools; fall back to generic if unknown case
    pools = PERSONA_POOLS.get(case_code, (LOW, MID, HIGH))
    pool_low, pool_mid, pool_high = pools
    out = []
    for i, p in enumerate(levels, start=1):
        jitter = rng.uniform(-0.045, 0.045)
        p_noisy = max(0.05, min(0.95, p + jitter))
        phq9 = max(0.0, min(27.0, 24 - 18 * p_noisy + rng.uniform(-2, 2)))
        out.append({
            "session_number": i,
            "transcript": build_session_transcript(
                p_noisy, rng,
                low_pool=pool_low, mid_pool=pool_mid, high_pool=pool_high,
            ),
            "latent_level": round(p_noisy, 3),
            "phq9": round(phq9, 1),
        })
    return out
