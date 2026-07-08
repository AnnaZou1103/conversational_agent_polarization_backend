from app.db.documents import user_docs
from app.schema import SurveyResponses
from app.survey_order import PRE_SURVEY_ORDER, POST_SURVEY_ORDER, apply_order_prefix


def save_pre_survey(study_id: str, survey_responses: SurveyResponses):
    ordered_responses = apply_order_prefix(survey_responses.responses, PRE_SURVEY_ORDER)
    user_docs.update_one(
        {"study_id": study_id},
        {
            "$set": {"pre_survey": ordered_responses},
            "$currentDate": {"updated_at": True},
        },
    )


def save_post_survey(study_id: str, survey_responses: SurveyResponses):
    ordered_responses = apply_order_prefix(survey_responses.responses, POST_SURVEY_ORDER)
    user_docs.update_one(
        {"study_id": study_id},
        {
            "$set": {"post_survey": ordered_responses},
            "$currentDate": {"updated_at": True},
        },
    )
