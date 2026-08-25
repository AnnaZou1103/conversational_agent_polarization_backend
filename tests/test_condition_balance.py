"""create_study_user's balanced round-robin must ignore `source: "manual"`
records — bare /survey opens (researcher testing, link-preview crawlers, stray
reloads) that mint a document but never yield data. Counting them permanently
shifts condition assignment for real participants.

Uses a hand-rolled collection stub rather than a live Mongo, following the
sys.modules-stubbing convention in test_safety_persistence.py.
"""
import collections
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

    # Five bare /survey opens. Under the old counting these occupied slots and
    # pushed real participants off the round-robin for the next five turns.
    for _ in range(5):
        admin.create_study_user()
    assert {d["source"] for d in col.docs} == {"manual"}
    assert len(col.docs) == 5

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


def test_each_project_balances_from_zero():
    admin, col = _fresh_admin_module()

    # Batch A runs to a clean round-robin.
    for i in range(5):
        admin.create_study_user(participant_id=f"A{i:030d}", project_id="proj-A")
    a = [d["strategy"] for d in col.docs if d["project_id"] == "proj-A"]
    assert sorted(a) == sorted(ALL), a

    # Batch B starts from zero rather than inheriting A's totals.
    for i in range(5):
        admin.create_study_user(participant_id=f"B{i:030d}", project_id="proj-B")
    b = [d["strategy"] for d in col.docs if d["project_id"] == "proj-B"]
    assert sorted(b) == sorted(ALL), b


def test_a_skewed_project_does_not_bleed_into_the_next_one():
    admin, col = _fresh_admin_module()

    # An earlier batch that is heavily lopsided — e.g. one condition was forced
    # with an explicit strategy for a pilot.
    for i in range(8):
        admin.create_study_user(
            strategy=Strategy.CONTROL.value,
            participant_id=f"A{i:030d}",
            project_id="proj-A",
        )

    # The next project must still hand out one of each, not spend its first
    # eight participants compensating for proj-A.
    for i in range(5):
        admin.create_study_user(participant_id=f"B{i:030d}", project_id="proj-B")
    b = [d["strategy"] for d in col.docs if d["project_id"] == "proj-B"]
    assert sorted(b) == sorted(ALL), b


def test_remainders_stay_inside_their_own_project():
    admin, col = _fresh_admin_module()

    # 7 participants over 5 conditions: two conditions get 2, the rest 1. The
    # remainder must not be carried into the next project.
    for i in range(7):
        admin.create_study_user(participant_id=f"A{i:030d}", project_id="proj-A")
    counts_a = collections.Counter(
        d["strategy"] for d in col.docs if d["project_id"] == "proj-A"
    )
    assert sorted(counts_a.values()) == [1, 1, 1, 2, 2], dict(counts_a)

    for i in range(5):
        admin.create_study_user(participant_id=f"B{i:030d}", project_id="proj-B")
    counts_b = collections.Counter(
        d["strategy"] for d in col.docs if d["project_id"] == "proj-B"
    )
    assert sorted(counts_b.values()) == [1, 1, 1, 1, 1], dict(counts_b)


def test_records_with_no_project_form_their_own_bucket():
    admin, col = _fresh_admin_module()

    for i in range(5):
        admin.create_study_user(participant_id=f"P{i:030d}", project_id="proj-A")
    # A participant arriving without a projectId is balanced against the other
    # project-less records, not against proj-A.
    for i in range(5):
        admin.create_study_user(participant_id=f"N{i:030d}")

    loose = [d["strategy"] for d in col.docs if d["project_id"] is None]
    assert sorted(loose) == sorted(ALL), loose


def test_ties_are_not_always_broken_the_same_way():
    """Every project resets to an all-zero count, so a fixed tie-break would
    give the first participant of every batch the same condition."""
    firsts = set()
    for run in range(25):
        admin, col = _fresh_admin_module()
        admin.create_study_user(participant_id=f"X{run:030d}", project_id=f"p{run}")
        firsts.add(col.docs[0]["strategy"])
    assert len(firsts) > 1, firsts
