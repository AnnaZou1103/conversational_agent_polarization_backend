"""Canonical survey question order, mirrored from
src/config/surveyConfig.tsx in conversational_agent_polarization.

Keep in sync with that file when questions are added, removed, or reordered:
a question `name` missing here still gets saved, just pushed after all the
known ones instead of appearing in its actual survey position.

Example: adding a new "petOwnership" question right after "AIFrequency" in
preSurveyPages in surveyConfig.tsx also means inserting "petOwnership" right
after "AIFrequency" in PRE_SURVEY_ORDER below.
"""

# commonEnemy/threatPerception are currently commented out in
# surveyConfig.tsx but kept here for older responses that still have them.
PRE_SURVEY_ORDER = [
    "AIFrequency",
    "partyIdentification",
    "strongRepublican",
    "strongDemocrat",
    "closerParty",
    "similarityToOutgroup",
    "partyImportance",
    "angerTowardOutgroup",
    "empathyTowardOutgroup",
    "commonEnemy",
    "threatPerception",
    "rateRepublicansPre",
    "rateDemocratsPre",
    "preSurveyAttentionCheck",
]

# rateRepublicans/rateDemocrats and oqAttitudeChange are the pre-rename/
# pre-split keys older responses may still have (see "Rebrand header logo,
# revise participant-facing task copy, and expand post-survey questions" in
# conversational_agent_polarization) — kept so those older records still
# sort into a sensible position instead of falling to the end.
POST_SURVEY_ORDER = [
    "emotionEnthusiastic",
    "emotionHappy",
    "emotionCalm",
    "emotionDull",
    "emotionAngry",
    "emotionSad",
    "completedTask",
    "rateRepublicans",
    "rateDemocrats",
    "rateRepublicansPost",
    "rateDemocratsPost",
    "giveCents",
    "reducePollStations",
    "ignoreUnfavorableJudges",
    "prosecuteJournalists",
    "acceptElectionResults",
    "postSurveyAttentionCheck1",
    "sendThreateningMessages",
    "publicHarass",
    "violenceForGoals",
    "violenceForElection",
    "ceEasyToUnderstand",
    "ceClearCommunication",
    "ceKeptContext",
    "postSurveyAttentionCheck2",
    "impressionFakeNatural",
    "impressionMachinelikeHumanlike",
    "impressionUnconsciousConscious",
    "fhListened",
    "fhInterested",
    "fhUnderstoodPerspective",
    "fhUnderstanding",
    "fhResponsive",
    "taSuspicious",
    "taWary",
    "taConfident",
    "taIntegrity",
    "taDependable",
    "taReliable",
    "taTrust",
    "maeChangingViews",
    "maeHiddenAgenda",
    "maeHonestIntentions",
    "maeWillingToDiscuss",
    "maeMoreOpen",
    "maeMoreConversations",
    "weWillingFuture",
    "oqWillingFutureWhy",
    "saSatisfiedOverall",
    "oqSatisfiedWhy",
    "oqAttitudeChange",
    "attitudeChangeSelf",
    "oqAttitudeChangeSelfWhy",
    "attitudeChangeOthers",
    "oqImprove",
]


def apply_order_prefix(responses: dict, order: list) -> dict:
    """Rebuild `responses` with each key prefixed by a zero-padded sequence
    number matching `order`, so that alphabetically-sorting viewers (Compass
    table view, Atlas UI, CSV export) display keys in survey order.

    Keys not present in `order` (e.g. a newly added question not yet
    reflected here) are kept, appended after the known ones in their
    original relative order, so nothing is silently dropped.
    """
    index = {key: i for i, key in enumerate(order)}
    known = sorted((k for k in responses if k in index), key=lambda k: index[k])
    unknown = [k for k in responses if k not in index]
    ordered_keys = known + unknown
    width = max(2, len(str(len(ordered_keys))))

    return {
        f"{i:0{width}d}_{key}": responses[key]
        for i, key in enumerate(ordered_keys, start=1)
    }
