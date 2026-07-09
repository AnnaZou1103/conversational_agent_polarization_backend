"""Verifies that pre_survey/post_survey responses are stored with a
sequence-number prefix matching survey order (see app/survey_order.py),
so MongoDB/CSV exports stop alphabetizing question keys.

No real MongoDB connection is made — app.db.documents is stubbed at import
time (other test files in this suite do the same, and share process-wide
sys.modules state, so whichever runs first "wins"). To stay correct
regardless of import order, save_pre_survey/save_post_survey are exercised
by patching app.db.survey.user_docs directly per test rather than relying
on a specific stub instance having been installed.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

sys.modules.setdefault("app.db.documents", MagicMock(user_docs=MagicMock()))

from app.db.survey import save_pre_survey, save_post_survey
from app.schema import SurveyResponses
from app.survey_order import apply_order_prefix, PRE_SURVEY_ORDER, POST_SURVEY_ORDER


def _set_payload(mock_user_docs, field: str) -> dict:
    """Pull the dict passed as {"$set": {field: ...}} from a mocked
    user_docs.update_one call."""
    args, kwargs = mock_user_docs.update_one.call_args
    update = args[1] if len(args) > 1 else kwargs["update"]
    return update["$set"][field]


def test_apply_order_prefix_orders_known_keys():
    responses = {
        "partyIdentification": "Democrat",
        "AIFrequency": "5 - Daily or almost daily",
    }
    ordered = apply_order_prefix(responses, PRE_SURVEY_ORDER)
    assert list(ordered.keys()) == ["01_AIFrequency", "02_partyIdentification"]
    assert ordered["01_AIFrequency"] == "5 - Daily or almost daily"
    assert ordered["02_partyIdentification"] == "Democrat"


def test_apply_order_prefix_appends_unknown_keys_last():
    responses = {"zzzNotInList": "x", "AIFrequency": "5"}
    ordered = apply_order_prefix(responses, PRE_SURVEY_ORDER)
    assert list(ordered.keys()) == ["01_AIFrequency", "02_zzzNotInList"]


def test_apply_order_prefix_empty_input():
    assert apply_order_prefix({}, PRE_SURVEY_ORDER) == {}


def test_pre_survey_order_includes_feeling_thermometer_questions():
    # Added when the pre-survey gained its own feeling-thermometer page
    # (rateRepublicansPre/rateDemocratsPre), right before the attention check.
    assert PRE_SURVEY_ORDER.index("rateRepublicansPre") < PRE_SURVEY_ORDER.index("preSurveyAttentionCheck")
    assert PRE_SURVEY_ORDER.index("rateDemocratsPre") < PRE_SURVEY_ORDER.index("preSurveyAttentionCheck")


def test_post_survey_order_covers_renamed_and_legacy_thermometer_keys():
    # rateRepublicans/rateDemocrats (pre-rename) and their *Post replacements
    # must both be recognized so old and new records sort correctly.
    for key in ("rateRepublicans", "rateDemocrats", "rateRepublicansPost", "rateDemocratsPost"):
        assert key in POST_SURVEY_ORDER


def test_post_survey_order_covers_attitude_change_split():
    # oqAttitudeChange (legacy single field) was split into a rating +
    # open-ended pair, plus a new attitudeChangeOthers rating.
    for key in ("oqAttitudeChange", "attitudeChangeSelf", "oqAttitudeChangeSelfWhy", "attitudeChangeOthers"):
        assert key in POST_SURVEY_ORDER
    assert POST_SURVEY_ORDER.index("attitudeChangeSelf") < POST_SURVEY_ORDER.index("oqImprove")


@patch("app.db.survey.user_docs")
def test_save_pre_survey_stores_prefixed_keys_in_survey_order(mock_user_docs):
    payload = SurveyResponses(responses={
        "empathyTowardOutgroup": "60",
        "AIFrequency": "5 - Daily or almost daily",
        "partyIdentification": "Democrat",
    })
    save_pre_survey(study_id="__test__", survey_responses=payload)

    stored = _set_payload(mock_user_docs, "pre_survey")
    # Dense numbering over the keys actually present, in canonical relative
    # order (AIFrequency < partyIdentification < empathyTowardOutgroup) —
    # not the questions' absolute position in the full survey.
    assert list(stored.keys()) == [
        "01_AIFrequency",
        "02_partyIdentification",
        "03_empathyTowardOutgroup",
    ]
    assert stored["01_AIFrequency"] == "5 - Daily or almost daily"


@patch("app.db.survey.user_docs")
def test_save_post_survey_stores_prefixed_keys_in_survey_order(mock_user_docs):
    payload = SurveyResponses(responses={
        "oqImprove": "n/a",
        "emotionHappy": "6",
        "rateRepublicans": "40",
    })
    save_post_survey(study_id="__test__", survey_responses=payload)

    stored = _set_payload(mock_user_docs, "post_survey")
    # Dense numbering over the keys actually present, in canonical relative
    # order (emotionHappy < rateRepublicans < oqImprove).
    assert list(stored.keys()) == [
        "01_emotionHappy",
        "02_rateRepublicans",
        "03_oqImprove",
    ]
