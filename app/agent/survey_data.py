"""
Survey findings and data card content for study conditions.

COMMON_IDENTITY_DATA_CARD
  Text shown to participants in the common_identity condition after the
  exhausted majority concept has been introduced. Replace with the exact
  survey citation when final data is available.
"""

# Placeholder — replace with actual survey citation and wording when available
COMMON_IDENTITY_DATA_CARD = (
    "In a recent national survey, most Americans — across party lines — "
    "said they feel exhausted with political division and don't feel "
    "represented by the most extreme voices."
)

"""
Survey findings for the misperception_correction condition.

Fill in actual values from the source document when available.
Each entry corresponds to one quiz question, in order.

Fields:
  id              — question key used in session signals ("q1"–"q8")
  label           — short description of the action (party-neutral, for display)
  survey_average  — the survey finding for this action, stated as a fact sentence.
                    "{party}" is replaced at runtime (via _get_opposing_party)
                    with the opposing-party adjective ("Republican" or
                    "Democratic"), the same label the quiz questions use.

The fact wording is derived from the survey average on the 1–4 scale:
  1 = Never        → "Most {party} supporters would never support this action."
  2 = Probably not → "Most {party} supporters would probably not support this action."
  3 = Probably     → "Most {party} supporters would probably support this action."
  4 = Definitely   → "Most {party} supporters would definitely support this action."
"""

QUIZ_QUESTIONS: list[dict] = [
    {
        "id": "q1",
        "label": "Banning FAR-LEFT group rallies in the state capital",
        "survey_average": "Most {party} supporters would probably not support this action.",
    },
    {
        "id": "q2",
        "label": "Prosecuting journalists who accuse opposing-party politicians of misconduct",
        "survey_average": "Most {party} supporters would never support this action.",
    },
    {
        "id": "q3",
        "label": "Reinterpreting the Constitution to block the other party's policies",
        "survey_average": "Most {party} supporters would never support this action.",
    },
    {
        "id": "q4",
        "label": "Using violence to block major laws passed by the other party",
        "survey_average": "Most {party} supporters would never support this action.",
    },
    {
        "id": "q5",
        "label": "Reducing voting stations in areas that lean toward the other party",
        "survey_average": "Most {party} supporters would never support this action.",
    },
    {
        "id": "q6",
        "label": "Ignoring court rulings issued by the other party's judges",
        "survey_average": "Most {party} supporters would probably not support this action.",
    },
    {
        "id": "q7",
        "label": "Not accepting the results of a presidential election they lost",
        "survey_average": "Most {party} supporters would never support this action.",
    },
    {
        "id": "q8",
        "label": "Laws making it easier for their party (and harder for the other party) to win elections",
        "survey_average": "Most {party} supporters would never support this action.",
    },
]
