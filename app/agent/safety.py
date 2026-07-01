"""Safety monitor: rule-based detection for gibberish and indecent language.

Pure regex + heuristics — no LLM calls. Runs synchronously in the agent pipeline
before OBSERVE/THINK/EXECUTE so that bad input never contaminates the main
research model's context or the extracted `signals`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

# ---------------------------------------------------------------------------
# Reminder / termination templates (pre-written, not LLM-generated)
# ---------------------------------------------------------------------------

# Tiered reminders — index = (consecutive_reminders - 1). The 1st strike gets
# a gentler nudge; subsequent strikes use firmer wording. The list length must
# be at least CONSECUTIVE_REMINDER_LIMIT - 1 (the limit-th strike terminates
# instead of reminding).
GIBBERISH_REMINDERS = [
    # Strike 1 — gentle, assumes accident or confusion
    "I'm having a little trouble following that. Could you share your thought "
    "in a full sentence?",
    # Strike 2 — firmer, suggests closing if needed
    "I'm still having trouble following. Could you put your answer in a clear "
    "sentence? If now isn't a good time to continue, you can close the chat.",
    # Strike 3 — invites rephrasing
    "I want to make sure I understand you — could you try rephrasing that as "
    "a complete thought?",
    # Strike 4 — patient but direct
    "I'm having difficulty making sense of your message. Please take a moment "
    "and share what's on your mind in a full sentence.",
    # Strike 5+ — open-ended, no pressure
    "I haven't been able to follow your last few messages. Whenever you're "
    "ready to continue, I'm here — just send a clear sentence and we'll pick "
    "up from there.",
]

INDECENT_REMINDERS = [
    # Strike 1 — polite redirect
    "This is a research conversation, so please keep the language polite and "
    "respectful. Could you rephrase that?",
    # Strike 2 — firmer, suggests closing if needed
    "Another reminder, please use polite language here. If you'd rather not "
    "continue, you can close the chat.",
    # Strike 3 — keeps door open
    "I'd like to keep this conversation going, but I do need you to use "
    "respectful language. Could you try again?",
    # Strike 4 — frames it as a shared space
    "We're having a research conversation and I want to make sure it stays a "
    "comfortable space. Please keep the language courteous.",
    # Strike 5+ — patient, no ultimatum
    "I'm still here and happy to continue — whenever you're ready to engage "
    "respectfully, just send your next message.",
]

# Backward-compatible singular aliases — point at the strike-1 wording.
GIBBERISH_REMINDER = GIBBERISH_REMINDERS[0]
INDECENT_REMINDER = INDECENT_REMINDERS[0]

# ---------------------------------------------------------------------------
# Threshold
# ---------------------------------------------------------------------------

# Log a warning at this consecutive-reminder streak count so operators can
# monitor persistently disruptive sessions. Does not trigger termination —
# only the session time limit ends the conversation.
CONSECUTIVE_REMINDER_LIMIT = 3


# ---------------------------------------------------------------------------
# Indecent-language regex patterns
# Leet-substitution and spacing tolerance; word-boundary anchored to avoid
# false positives on embedded substrings ("class", "assassin", "Scunthorpe").
# ---------------------------------------------------------------------------

_I = r"[i1!|]"
_O = r"[o0]"
_S = r"[s$5]"
_A = r"[a@4]"
_E = r"[e3]"
_U = r"[u\*]"
_SEP = r"[\s\W_]*"  # allows "f u c k", "f.u.c.k", "f_u_c_k"


def _pat(body: str) -> re.Pattern[str]:
    """Compile a profanity pattern with word-boundary anchors and ignore-case."""
    return re.compile(rf"\b{body}\b", re.IGNORECASE)


# Conjugation suffixes used after a profanity stem. Constrained set to avoid
# greedy matching into innocuous words (e.g. "Dickinson" — see tests).
_SUF = r"(?:s|es|ed|er|ers|ing|ings|y|ies|ish)?"

# Patterns are ordered from most-used to least.
# - Words commonly used in compounds (fuck, shit) drop the leading \b so that
#   "motherfucker", "bullshit", "dipshit" are caught.
# - Words prone to substring false positives (cunt→Scunthorpe, dick→Dickinson)
#   keep the leading \b and use a constrained suffix list.
INDECENT_PATTERNS: list[re.Pattern[str]] = [
    # fuck, fucking, fucker, motherfucker, fck, f*ck, f u c k.
    # No leading \b — compounds are intentional. Trailing \b required.
    re.compile(rf"f{_SEP}{_U}{_SEP}c{_SEP}k{_SUF}\b", re.IGNORECASE),
    # "fck" missing-vowel variant
    re.compile(rf"\bf{_SEP}c{_SEP}k{_SUF}\b", re.IGNORECASE),
    # shit, shitty, bullshit, sh!t, sh1t. Compounds intentional → no leading \b.
    re.compile(rf"{_S}h{_SEP}{_I}{_SEP}t{_SUF}\b", re.IGNORECASE),
    # asshole, a$$hole — leading \b to avoid "ass" substring traps
    _pat(rf"{_A}{_SEP}{_S}{_SEP}{_S}{_SEP}h{_SEP}{_O}{_SEP}l{_SEP}{_E}{_SUF}"),
    # bitch, bitches, biatch, b!tch
    _pat(rf"b{_SEP}{_I}{_SEP}{_A}?{_SEP}t{_SEP}c{_SEP}h{_SUF}"),
    # cunt — keep \b to avoid Scunthorpe
    _pat(rf"c{_SEP}{_U}{_SEP}n{_SEP}t{_SUF}"),
    # dick, dicks, dicked, dickhead, d!ck — keep \b and constrained suffixes
    # so "Dickinson" does NOT match
    _pat(rf"d{_SEP}{_I}{_SEP}c{_SEP}k(?:s|ed|head|heads|ish|wad|wads)?"),
    # pussy (as insult)
    _pat(rf"p{_SEP}{_U}{_SEP}{_S}{_SEP}{_S}{_SEP}y"),
    # bastard, bastards
    _pat(rf"b{_SEP}{_A}{_SEP}{_S}{_SEP}t{_SEP}{_A}{_SEP}r{_SEP}d{_SUF}"),
    # whore
    _pat(rf"wh{_SEP}{_O}{_SEP}r{_SEP}{_E}{_SUF}"),
    # Slurs — curated conservatively, leading \b required.
    _pat(rf"n{_SEP}{_I}{_SEP}g{_SEP}g{_SEP}[e3ae]r?s?"),
    _pat(rf"f{_SEP}{_A}{_SEP}g(?:got|gots|s)?"),
    _pat(rf"r{_SEP}{_E}{_SEP}t{_SEP}{_A}{_SEP}r{_SEP}d(?:ed|s)?"),
]


def contains_indecent(msg: str) -> bool:
    """Return True if the message matches any INDECENT_PATTERNS entry."""
    return any(p.search(msg) for p in INDECENT_PATTERNS)


# ---------------------------------------------------------------------------
# Gibberish heuristics
# ---------------------------------------------------------------------------

# Allowlist: short-but-valid replies that would otherwise be flagged.
_SHORT_VALID_REPLIES = {
    "ok",
    "okay",
    "yes",
    "no",
    "yep",
    "nope",
    "yeah",
    "nah",
    "sure",
    "idk",
    "lol",
    "hmm",
    "huh",
    "maybe",
    "skip",
    "continue",
    "next",
    "true",
    "false",
    "right",
    "wrong",
    "fine",
    "good",
    "bad",
    "same",
    "agree",
    "republican",
    "democrat",
    "democratic",
    "gop",
    # Greetings — short but valid openers
    "hi",
    "hello",
    "hey",
    "heya",
    "hiya",
    "howdy",
    "greetings",
    "yo",
    "sup",
    "morning",
    "afternoon",
    "evening",
}

# Tiny built-in English wordlist — enough to recognize at least one common
# word in virtually any coherent English message. Intentionally small and
# auditable; for richer coverage, swap in a file-loaded wordlist later.
_COMMON_WORDS = frozenset(
    {
        # articles, pronouns, prepositions
        "the",
        "a",
        "an",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "its",
        "our",
        "their",
        "this",
        "that",
        "these",
        "those",
        "what",
        "which",
        "who",
        "whom",
        "of",
        "in",
        "on",
        "at",
        "by",
        "to",
        "for",
        "with",
        "from",
        "as",
        "into",
        "about",
        "over",
        "under",
        "through",
        "between",
        "before",
        "after",
        "during",
        "against",
        "without",
        # be / have / do / modal
        "am",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "done",
        "will",
        "would",
        "shall",
        "should",
        "can",
        "could",
        "may",
        "might",
        "must",
        # common verbs
        "go",
        "going",
        "went",
        "come",
        "came",
        "get",
        "got",
        "make",
        "made",
        "think",
        "thought",
        "see",
        "saw",
        "know",
        "knew",
        "feel",
        "felt",
        "say",
        "said",
        "tell",
        "told",
        "want",
        "need",
        "like",
        "love",
        "hate",
        "believe",
        "understand",
        "mean",
        "try",
        "use",
        "work",
        "find",
        "give",
        "take",
        "put",
        "keep",
        "let",
        "help",
        "talk",
        "speak",
        "ask",
        "answer",
        "live",
        "seem",
        "become",
        "leave",
        "happen",
        # political / study vocabulary
        "politics",
        "political",
        "party",
        "republican",
        "democrat",
        "democratic",
        "gop",
        "vote",
        "voted",
        "voter",
        "voters",
        "election",
        "elections",
        "government",
        "country",
        "nation",
        "american",
        "americans",
        "liberal",
        "conservative",
        "moderate",
        "independent",
        "president",
        "senator",
        "congress",
        "media",
        "news",
        "social",
        "people",
        "person",
        "side",
        "sides",
        "issue",
        "issues",
        # emotion / opinion
        "feel",
        "feeling",
        "feelings",
        "frustrated",
        "frustrating",
        "tired",
        "exhausted",
        "angry",
        "sad",
        "happy",
        "scared",
        "worried",
        "hopeful",
        "interesting",
        "surprised",
        "surprising",
        # conjunctions / common connectors
        "and",
        "or",
        "but",
        "not",
        "so",
        "because",
        "if",
        "then",
        "than",
        "though",
        "although",
        "while",
        "also",
        "too",
        "either",
        "neither",
        "just",
        "only",
        "really",
        "very",
        "much",
        "more",
        "most",
        "less",
        "some",
        "any",
        "all",
        "every",
        "each",
        "both",
        "many",
        "few",
        # time / place
        "now",
        "then",
        "today",
        "yesterday",
        "tomorrow",
        "year",
        "years",
        "day",
        "days",
        "time",
        "times",
        "here",
        "there",
        "where",
        "when",
        "why",
        "how",
        "because",
        # misc common
        "thing",
        "things",
        "something",
        "someone",
        "anything",
        "anyone",
        "everything",
        "everyone",
        "nothing",
        "nobody",
        "back",
        "down",
        "up",
        "out",
        "off",
        "still",
        "even",
        "again",
        "yes",
        "no",
        "maybe",
        "sure",
        "okay",
        # greetings
        "hi",
        "hello",
        "hey",
        "heya",
        "hiya",
        "howdy",
        "greetings",
        "yo",
        "sup",
        "morning",
        "afternoon",
        "evening",
        "hola",
        # everyday wellbeing vocabulary (control condition check-ins skew toward
        # these, and a single-word/short reply like "stressed" or "fine" has no
        # other token to fall back on)
        "fine",
        "busy",
        "stressed",
        "stress",
        "anxious",
        "anxiety",
        "overwhelmed",
        "overwhelming",
        "insomnia",
        "sleep",
        "sleeping",
        "sleepy",
        "lonely",
        "loneliness",
        "calm",
        "relax",
        "relaxed",
        "relaxing",
        "rest",
        "rested",
        "restless",
        "depressed",
        "depression",
        "motivated",
        "unmotivated",
        "burnout",
        "burned",
        "drained",
        "draining",
        "energy",
        "mood",
        "family",
        "friend",
        "friends",
        "relationship",
        "relationships",
        "school",
        "college",
        "job",
        "jobs",
        "deadline",
        "deadlines",
        "money",
        "bills",
        "bill",
        "rent",
        "groceries",
        "debt",
        "budget",
        # broader political topics (control_politics is open-ended, not just
        # the common_identity/personal_narrative vocabulary above)
        "economy",
        "economic",
        "immigration",
        "immigrant",
        "immigrants",
        "abortion",
        "healthcare",
        "health",
        "taxes",
        "tax",
        "inflation",
        "guns",
        "gun",
        "climate",
        "crime",
        "education",
        "policy",
        "policies",
        "law",
        "laws",
        "rights",
        "freedom",
        "justice",
        "war",
        "trade",
        "unemployment",
        "welfare",
        "security",
        # personality traits (personal_narrative asks users to describe a
        # specific person — single-trait answers like "stubborn" are the
        # expected response, not an edge case)
        "stubborn",
        "caring",
        "kind",
        "mean",
        "generous",
        "selfish",
        "honest",
        "dishonest",
        "loyal",
        "supportive",
        "judgmental",
        "passionate",
        "patient",
        "impatient",
        "smart",
        "hardworking",
        "lazy",
        "funny",
        "serious",
        "warm",
        "cold",
        "stubbornness",
        # relationship / family roles (the person personal_narrative asks
        # about is usually introduced with just a label like this)
        "uncle",
        "aunt",
        "cousin",
        "brother",
        "sister",
        "mom",
        "dad",
        "mother",
        "father",
        "grandma",
        "grandpa",
        "grandmother",
        "grandfather",
        "neighbor",
        "coworker",
        "boss",
        "classmate",
        "roommate",
        "husband",
        "wife",
        "spouse",
        "partner",
        "church",
        "thanksgiving",
        "holiday",
        "holidays",
        # media / common_identity vocabulary
        "fox",
        "cnn",
        "msnbc",
        "twitter",
        "facebook",
        "instagram",
        "tiktok",
        "youtube",
        "algorithm",
        "clickbait",
        "outrage",
        "biased",
        "unbiased",
        "bias",
        "misinformation",
        "echo",
        "chamber",
        "headline",
        "headlines",
        "coverage",
        "outlet",
        "outlets",
        "anchor",
        "pundit",
        "exhausting",
    }
)


def _looks_like_a_name(msg: str) -> bool:
    """True if every alphabetic token is capitalized (e.g. "Sarah", "My uncle Bob").

    Names can't be enumerated in `_COMMON_WORDS`, so a short reply where every
    word starts with a capital letter is treated as a likely proper noun
    rather than gibberish. Checked against the ORIGINAL casing, before
    `_normalize` lowercases everything.
    """
    words = re.findall(r"[A-Za-z]+", msg)
    if not words:
        return False
    return all(w[0].isupper() for w in words)


def _normalize(msg: str) -> str:
    return msg.strip().lower()


def _tokens(msg: str) -> list[str]:
    """Lowercase alphanumeric tokens."""
    return re.findall(r"[a-z0-9']+", msg.lower())


def _has_recognized_word(msg: str) -> bool:
    return any(t in _COMMON_WORDS for t in _tokens(msg))


def _has_long_char_run(msg: str) -> bool:
    """True if the message contains 5+ of the same character in a row."""
    return bool(re.search(r"(.)\1{4,}", msg))


def _is_repeating_substring(msg: str) -> bool:
    """True if the message is dominated by a single repeating short substring.

    Detects patterns like 'asdfasdf', 'lololol', 'abcabcabc' where a 2–4 char
    unit repeats to fill >=70% of the non-space message.
    """
    stripped = re.sub(r"\s+", "", msg)
    if len(stripped) < 6:
        return False
    for unit_len in (2, 3, 4):
        for start in range(max(1, len(stripped) - unit_len * 3 + 1)):
            unit = stripped[start : start + unit_len]
            if not unit or len(unit) < unit_len:
                continue
            # How many times does this unit repeat consecutively starting here?
            count = 1
            pos = start + unit_len
            while stripped[pos : pos + unit_len] == unit:
                count += 1
                pos += unit_len
            if count >= 3 and count * unit_len >= 0.7 * len(stripped):
                return True
    return False


def _has_no_vowels(msg: str) -> bool:
    """True if a sufficiently long alphabetic message contains no vowels.

    Catches random consonant sequences like 'qwrtp', 'zxcvbn'. Uses only the
    alphabetic subset so punctuation doesn't throw it off.
    """
    letters = re.sub(r"[^a-zA-Z]", "", msg)
    if len(letters) < 4:
        return False
    return not re.search(r"[aeiouyAEIOUY]", letters)


def is_gibberish(msg: str, quiz_mode: bool = False) -> bool:
    """Heuristic gibberish detection — no LLM, all regex/token rules.

    quiz_mode=True skips the dictionary soft-signal: in misperception_correction
    stage_2, valid Likert reasoning ("doesn't sound right", "feels wrong") often
    contains zero words from our small _COMMON_WORDS list, so the soft-signal's
    false-positive rate is too high. Hard signals (length, char-runs, repeats,
    vowel-less) still fire.
    """
    normalized = _normalize(msg)

    # Empty / whitespace-only is invalid
    if not normalized:
        return True

    # Hard signals — any one is sufficient. Char-run check runs first so
    # "!!!!!!!" is caught before the "no letters" allowance below.
    if _has_long_char_run(normalized):
        return True
    if _is_repeating_substring(normalized):
        return True

    # Messages with no letters at all (e.g. "...", "?", "!?") are not really
    # "gibberish" — they're ambiguous reactions. Let them pass; OBSERVE will
    # interpret them. Hard signals already handled punctuation spam above.
    if not re.search(r"[a-zA-Z]", normalized):
        return False

    # Single-letter alphabetic messages are too short to interpret.
    if len(normalized) < 2:
        return True

    if normalized in _SHORT_VALID_REPLIES:
        return False

    if _has_no_vowels(normalized):
        return True

    # Indecent language is profanity, not gibberish — don't double-flag.
    # (evaluate_message will route it to the indecent category.)
    if contains_indecent(normalized):
        return False

    if quiz_mode:
        return False

    # A capitalized word/phrase (checked against ORIGINAL casing, before
    # normalization) reads as a likely proper noun — e.g. a person's name in
    # personal_narrative ("Sarah", "My uncle Bob"). Names can't be enumerated
    # in _COMMON_WORDS, so the soft signal below would otherwise flag every
    # name it hasn't seen.
    if _looks_like_a_name(msg):
        return False

    # Soft signal — require both short-ish length AND no recognized word
    if len(normalized) <= 40 and not _has_recognized_word(normalized):
        return True

    return False


def is_exact_repeat(msg: str, previous: str | None) -> bool:
    """True if the normalized message matches the previous user message."""
    if not previous:
        return False
    return _normalize(msg) == _normalize(previous)


# ---------------------------------------------------------------------------
# Verdict + top-level evaluation
# ---------------------------------------------------------------------------

Action = Literal["clean", "reminder", "terminate"]
Category = Literal["gibberish", "indecent", "both", "clean", "time_limit"]


@dataclass
class SafetyVerdict:
    action: Action
    category: Category
    reason: str
    reminder_text: str = ""
    termination_text: str = ""
    user_message_excerpt: str = ""
    consecutive_reminders: int = 0  # post-application streak value
    indecent_count: int = 0  # lifetime, logging only
    invalid_count: int = 0  # lifetime, logging only
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "category": self.category,
            "reason": self.reason,
            "user_message_excerpt": self.user_message_excerpt,
            "consecutive_reminders": self.consecutive_reminders,
            "indecent_count": self.indecent_count,
            "invalid_count": self.invalid_count,
            "timestamp": self.timestamp,
        }


def evaluate_message(
    user_message: str,
    previous_user_message: str | None,
    consecutive_reminders: int,
    indecent_count: int,
    invalid_count: int,
    quiz_mode: bool = False,
) -> SafetyVerdict:
    """Classify the message and decide action given current session counters.

    Gibberish and indecent language never terminate the conversation — the
    action is always "clean" or "reminder". Termination is handled externally
    by the session time limit. A single clean message resets the streak to zero.
    Once the streak exceeds the reminder tier list length, the last tier is
    repeated indefinitely.

    This function is pure (no state mutation). The caller is responsible for
    applying the verdict to session state.
    """
    excerpt = user_message[:200]

    gibberish = is_gibberish(user_message, quiz_mode=quiz_mode) or is_exact_repeat(
        user_message, previous_user_message
    )
    indecent = contains_indecent(user_message)

    if not gibberish and not indecent:
        # Clean message — streak resets.
        return SafetyVerdict(
            action="clean",
            category="clean",
            reason="message passed all checks",
            user_message_excerpt=excerpt,
            consecutive_reminders=0,
            indecent_count=indecent_count,
            invalid_count=invalid_count,
        )

    category: Category
    if gibberish and indecent:
        category = "both"
    elif gibberish:
        category = "gibberish"
    else:
        category = "indecent"

    # Lifetime tallies (for logging only — do not drive termination).
    new_invalid_count = invalid_count + 1 if gibberish else invalid_count
    new_indecent_count = indecent_count + 1 if indecent else indecent_count

    # Send a reminder. Prefer the indecent reminder when both apply (more
    # specific feedback than the generic gibberish prompt). Cap the tier at the
    # last available entry so the firmest reminder repeats beyond that.
    prospective_streak = consecutive_reminders + 1
    tier = min(prospective_streak - 1, len(INDECENT_REMINDERS) - 1)
    if indecent:
        reminder_text = INDECENT_REMINDERS[tier]
        reason = "indecent language detected"
    else:
        reminder_text = GIBBERISH_REMINDERS[tier]
        reason = "gibberish / invalid input detected"

    return SafetyVerdict(
        action="reminder",
        category=category,
        reason=reason,
        reminder_text=reminder_text,
        user_message_excerpt=excerpt,
        consecutive_reminders=prospective_streak,
        indecent_count=new_indecent_count,
        invalid_count=new_invalid_count,
    )
