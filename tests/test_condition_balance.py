"""create_study_user's balanced round-robin must ignore `source: "manual"`
records — bare /survey opens (researcher testing, link-preview crawlers, stray
reloads) that mint a document but never yield data. Counting them permanently
shifts condition assignment for real participants.

Uses a hand-rolled collection stub rather than a live Mongo, following the
sys.modules-stubbing convention in test_safety_persistence.py.
"""
import sys
from unittest.mock import MagicMock

from app.agent.strategies import Strategy


class FakeCollection:
    """Just enough of a pymongo collection for create_study_user/generate_users.

    count_documents implements exact-match and `$ne`, including MongoDB's rule
    that `{"f": {"$ne": v}}` also matches documents where `f` is absent — the
    behaviour the backward-compatible filter relies on for pre-`source` records.
    """

    def __init__(self):
        self.docs = []

    def insert_one(self, doc):
        self.docs.append(doc)

    def insert_many(self, docs):
        self.docs.extend(docs)

    @staticmethod
    def _matches(doc, query):
        for field, cond in query.items():
            if isinstance(cond, dict) and "$ne" in cond:
                if doc.get(field, None) == cond["$ne"]:
                    return False
            elif doc.get(field, None) != cond:
                return False
        return True

    def count_documents(self, query):
        return sum(1 for d in self.docs if self._matches(d, query))

    def find(self, query, projection=None):
        return [d for d in self.docs if self._matches(d, query)]


def _fresh_admin_module():
    """Import app.db.admin against a fake collection, isolated per test."""
    for mod in ("app.db.admin", "app.db.documents"):
        sys.modules.pop(mod, None)
    col = FakeCollection()
    stub = MagicMock()
    stub.user_docs = col
    stub.conversation_docs = MagicMock()
    stub.conversations_archive = MagicMock()
    sys.modules["app.db.documents"] = stub
    import app.db.admin as admin

    return admin, col


ALL = [s.value for s in Strategy]


def _counts(col):
    return {s: sum(1 for d in col.docs if d["strategy"] == s) for s in ALL}


def test_manual_records_are_excluded_from_balancing():
    admin, col = _fresh_admin_module()

    # Five bare /survey opens, all landing on the same condition the tie-break
    # favours. Under the old counting these would push real participants away
    # from common_identity for the next five assignments.
    for _ in range(5):
        admin.create_study_user()
    assert {d["source"] for d in col.docs} == {"manual"}
    assert _counts(col)[Strategy.COMMON_IDENTITY.value] == 5

    # Real participants must still be assigned a clean round-robin.
    for i in range(5):
        admin.create_study_user(participant_id=f"PID{i:029d}")

    real = [d for d in col.docs if d["source"] == "cloudresearch"]
    assert len(real) == 5
    assert sorted(d["strategy"] for d in real) == sorted(ALL), (
        f"real participants were not balanced: {[d['strategy'] for d in real]}"
    )


def test_cloudresearch_records_are_counted():
    admin, col = _fresh_admin_module()

    for i in range(10):
        admin.create_study_user(participant_id=f"PID{i:029d}")

    counts = _counts(col)
    assert set(counts.values()) == {2}, counts


def test_admin_generated_records_are_counted():
    admin, col = _fresh_admin_module()

    # Pre-generated links are meant to be handed to real participants, so they
    # must occupy a slot: three extra control links means the next real
    # participants skip control until the rest catch up.
    admin.generate_users_by_agent_strategy(Strategy.CONTROL.value, 3)
    assert [d["source"] for d in col.docs] == ["admin"] * 3

    for i in range(4):
        admin.create_study_user(participant_id=f"PID{i:029d}")

    real = [d["strategy"] for d in col.docs if d.get("source") == "cloudresearch"]
    assert Strategy.CONTROL.value not in real, real
    assert sorted(real) == sorted(s for s in ALL if s != Strategy.CONTROL.value)


def test_generate_users_tags_every_condition_as_admin():
    admin, col = _fresh_admin_module()

    admin.generate_users(2)

    assert len(col.docs) == 2 * len(ALL)
    assert {d["source"] for d in col.docs} == {"admin"}
    assert _counts(col) == {s: 2 for s in ALL}


def test_records_predating_the_source_field_are_still_counted():
    admin, col = _fresh_admin_module()

    # Legacy documents have no `source` key at all. `$ne: "manual"` must match
    # them, otherwise a missed backfill silently zeroes the counter.
    col.docs.extend(
        {"type": "study", "strategy": Strategy.CONTROL.value, "state": "complete"}
        for _ in range(3)
    )

    for i in range(4):
        admin.create_study_user(participant_id=f"PID{i:029d}")

    real = [d["strategy"] for d in col.docs if d.get("source") == "cloudresearch"]
    assert Strategy.CONTROL.value not in real, real


def test_returning_participant_keeps_their_study_id_and_condition():
    admin, col = _fresh_admin_module()

    pid = "PID" + "0" * 29
    first = admin.create_study_user(participant_id=pid)
    condition = col.docs[0]["strategy"]

    # Re-opening the CloudResearch link / navigating back to /survey.
    for _ in range(3):
        assert admin.create_study_user(participant_id=pid) == first

    assert len(col.docs) == 1, [d["study_id"] for d in col.docs]
    assert col.docs[0]["strategy"] == condition


def test_returning_participant_does_not_consume_extra_slots():
    admin, col = _fresh_admin_module()

    # One participant refreshing five times must not push the next four
    # participants off the round-robin.
    pid = "PID" + "0" * 29
    admin.create_study_user(participant_id=pid)
    for _ in range(5):
        admin.create_study_user(participant_id=pid)

    for i in range(1, 5):
        admin.create_study_user(participant_id=f"PID{i:029d}")

    assert len(col.docs) == 5
    assert sorted(d["strategy"] for d in col.docs) == sorted(ALL)


def test_screened_out_participant_cannot_reroll_into_another_condition():
    admin, col = _fresh_admin_module()

    pid = "PID" + "0" * 29
    study_id = admin.create_study_user(participant_id=pid)
    col.docs[0].update(state="complete", screened=True)

    # Previously this handed them a fresh study_id in a random condition, and
    # one participant used it to work through three conditions until a screener
    # let them in.
    assert admin.create_study_user(participant_id=pid) == study_id
    assert len(col.docs) == 1


def test_duplicate_legacy_records_resolve_to_the_furthest_along_one():
    admin, col = _fresh_admin_module()

    pid = "PID" + "0" * 29
    col.docs.extend(
        [
            {"type": "study", "participant_id": pid, "study_id": "aaaaaa",
             "strategy": ALL[0], "state": "not_started", "created_at": 1, "updated_at": 1},
            {"type": "study", "participant_id": pid, "study_id": "bbbbbb",
             "strategy": ALL[1], "state": "complete", "created_at": 2, "updated_at": 2},
            {"type": "study", "participant_id": pid, "study_id": "cccccc",
             "strategy": ALL[2], "state": "pre_survey", "created_at": 3, "updated_at": 3},
        ]
    )

    assert admin.create_study_user(participant_id=pid) == "bbbbbb"
    assert len(col.docs) == 3


def test_manual_opens_are_unaffected_by_the_participant_lookup():
    admin, col = _fresh_admin_module()

    # No participant_id means no way to recognise a repeat visitor, so bare
    # /survey opens still mint a document each time — they just do not count.
    ids = [admin.create_study_user() for _ in range(3)]
    assert len(set(ids)) == 3
    assert {d["source"] for d in col.docs} == {"manual"}
