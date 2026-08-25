"""Integration test — condition assignment against a real MongoDB instance.

tests/test_condition_balance.py covers the same rules against a hand-rolled
collection stub. That stub encodes *my* reading of MongoDB semantics, most
importantly that `{"source": {"$ne": "manual"}}` also matches documents with no
`source` field at all — the assumption the backward-compatible filter rests on.
If that reading is wrong the unit tests still pass, so the rules are re-checked
here against the real thing.

Refuses to run unless MONGODB_DB_NAME names a *_dev database and its `users`
collection is empty. Deletes everything it inserts.

Run with: python tests/integration_test_condition_balance.py
"""
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.strategies import Strategy
from app.db.admin import create_study_user, generate_users_by_agent_strategy
from app.db.documents import db, user_docs

ALL = [s.value for s in Strategy]
CONTROL = Strategy.CONTROL.value
PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    print(f"{status}  {label}" + (f"\n       got: {detail}" if not condition and detail else ""))
    return condition


def cleanup():
    user_docs.delete_many({})


def guard():
    """Never touch the production study data."""
    if not db.name.endswith("_dev"):
        print(f"\033[91mRefusing to run against database {db.name!r} — "
              f"set MONGODB_DB_NAME to a *_dev database.\033[0m")
        sys.exit(1)
    count = user_docs.count_documents({})
    if count:
        print(f"\033[91mRefusing to run — {db.name}.users already holds "
              f"{count} document(s). Clear it first.\033[0m")
        sys.exit(1)


def conditions_of(source: str) -> list[str]:
    return sorted(d["strategy"] for d in user_docs.find({"source": source}))


def run() -> int:
    guard()
    failures = 0

    print(f"\n=== database: {db.name} ===")

    print("\n=== $ne on a missing field (the filter's core assumption) ===")
    user_docs.insert_many([
        {"type": "study", "strategy": CONTROL},                            # pre-`source` record
        {"type": "study", "strategy": CONTROL, "source": "manual"},
        {"type": "study", "strategy": CONTROL, "source": "cloudresearch"},
    ])
    counted = user_docs.count_documents(
        {"type": "study", "strategy": CONTROL, "source": {"$ne": "manual"}}
    )
    failures += not check(
        "$ne:'manual' counts the legacy record and the real one, not the manual one",
        counted == 2,
        f"counted {counted}, expected 2",
    )
    cleanup()

    print("\n=== bare /survey opens do not steal slots ===")
    for _ in range(5):
        create_study_user()
    manual = list(user_docs.find({"source": "manual"}))
    failures += not check(
        "5 bare opens are all tagged manual",
        len(manual) == 5 and all(d["source"] == "manual" for d in manual),
        str([d.get("source") for d in manual]),
    )
    failures += not check(
        "and none of them is tagged cloudresearch",
        not any(d["source"] == "cloudresearch" for d in manual),
        str([d["source"] for d in manual]),
    )
    for i in range(5):
        create_study_user(participant_id=f"PID{i:029d}")
    failures += not check(
        "the next 5 real participants still get one of each condition",
        conditions_of("cloudresearch") == sorted(ALL),
        str(conditions_of("cloudresearch")),
    )
    cleanup()

    print("\n=== admin-generated links DO occupy slots ===")
    generate_users_by_agent_strategy(CONTROL, 3)
    failures += not check(
        "generate_users_by_agent_strategy tags its records admin",
        [d["source"] for d in user_docs.find({})] == ["admin"] * 3,
        str([d.get("source") for d in user_docs.find({})]),
    )
    for i in range(4):
        create_study_user(participant_id=f"PID{i:029d}")
    real = conditions_of("cloudresearch")
    failures += not check(
        "real participants skip control until the other conditions catch up",
        real == sorted(s for s in ALL if s != CONTROL),
        str(real),
    )
    cleanup()

    print("\n=== a returning participant keeps their study_id and condition ===")
    pid = "PID" + "0" * 29
    first = create_study_user(participant_id=pid)
    condition = user_docs.find_one({"study_id": first})["strategy"]
    repeats = {create_study_user(participant_id=pid) for _ in range(4)}
    failures += not check(
        "re-opening the link returns the same study_id",
        repeats == {first},
        str(repeats),
    )
    failures += not check(
        "and mints no extra document",
        user_docs.count_documents({"participant_id": pid}) == 1,
        str(user_docs.count_documents({"participant_id": pid})),
    )

    user_docs.update_one({"study_id": first}, {"$set": {"state": "complete", "screened": True}})
    failures += not check(
        "a screened-out participant cannot reroll into another condition",
        create_study_user(participant_id=pid) == first,
    )
    failures += not check(
        "their condition is unchanged",
        user_docs.find_one({"study_id": first})["strategy"] == condition,
    )
    cleanup()

    print("\n=== each project balances from zero ===")
    # `{"project_id": None}` must match documents where the key is absent, the
    # same way `$ne` does — admin-generated links have no project_id key.
    user_docs.insert_one({"type": "study", "strategy": CONTROL, "source": "cloudresearch"})
    no_project = user_docs.count_documents(
        {"type": "study", "source": {"$ne": "manual"}, "project_id": None}
    )
    failures += not check(
        "project_id: None matches a document with no project_id key",
        no_project == 1,
        f"matched {no_project}, expected 1",
    )
    cleanup()

    for i in range(5):
        create_study_user(participant_id=f"A{i:030d}", project_id="proj-A")
    a = sorted(d["strategy"] for d in user_docs.find({"project_id": "proj-A"}))
    failures += not check("batch A gets one of each condition", a == sorted(ALL), str(a))

    for i in range(5):
        create_study_user(participant_id=f"B{i:030d}", project_id="proj-B")
    b = sorted(d["strategy"] for d in user_docs.find({"project_id": "proj-B"}))
    failures += not check(
        "batch B starts from zero instead of inheriting A's totals",
        b == sorted(ALL),
        str(b),
    )
    cleanup()

    print("\n=== a lopsided project does not bleed into the next one ===")
    for i in range(8):
        create_study_user(strategy=CONTROL, participant_id=f"A{i:030d}", project_id="proj-A")
    for i in range(5):
        create_study_user(participant_id=f"B{i:030d}", project_id="proj-B")
    b = sorted(d["strategy"] for d in user_docs.find({"project_id": "proj-B"}))
    failures += not check(
        "batch B is unaffected by 8 forced control assignments in batch A",
        b == sorted(ALL),
        str(b),
    )
    cleanup()

    print("\n=== both rules together ===")
    for _ in range(3):
        create_study_user()                                   # bare opens
    pid = "PID" + "0" * 29
    create_study_user(participant_id=pid)
    for _ in range(3):
        create_study_user(participant_id=pid)                 # same person refreshing
    for i in range(1, 5):
        create_study_user(participant_id=f"PID{i:029d}")
    failures += not check(
        "5 real participants still get one of each condition",
        conditions_of("cloudresearch") == sorted(ALL),
        str(conditions_of("cloudresearch")),
    )
    total = user_docs.count_documents({})
    failures += not check(
        "only 8 documents exist: 3 manual + 5 participants",
        total == 8,
        f"{total} documents",
    )
    cleanup()

    print(f"\n{'=' * 40}")
    if failures == 0:
        print("\033[92mAll checks passed.\033[0m")
    else:
        print(f"\033[91m{failures} check(s) failed.\033[0m")
    return failures


if __name__ == "__main__":
    sys.exit(run())
