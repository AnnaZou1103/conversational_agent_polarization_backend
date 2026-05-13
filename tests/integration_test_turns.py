"""Integration test — sends real turns through the API and verifies MongoDB.

Run with: python tests/integration_test_turns.py

Requires server running on localhost:8080 (with MONGODB_URI set).
Cleans up the test user on exit.
"""
import sys
import json
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import urllib.request
import urllib.error
from app.db.documents import user_docs, conversation_docs
from app.db.admin import generate_study_id

BASE_URL = "http://localhost:8080"
STUDY_ID = "__turns_test__"
PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    suffix = f"\n       got: {detail}" if not condition and detail else ""
    print(f"{status}  {label}{suffix}")
    return condition


def post(path: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def send_turn(message: str) -> str:
    result = post("/v1/chat/completions", {
        "studyId": STUDY_ID,
        "model": "common-identity",
        "message": {"role": "user", "content": message},
        "stream": False,
    })
    return result["choices"][0]["message"]["content"]


def setup():
    """Insert test user directly into MongoDB."""
    user_docs.delete_one({"study_id": STUDY_ID})
    conversation_docs.delete_one({"study_id": STUDY_ID})
    from datetime import datetime, timezone
    user_docs.insert_one({
        "study_id": STUDY_ID,
        "type": "study",
        "strategy": "common_identity",
        "state": "intervention",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    })


def cleanup():
    user_docs.delete_one({"study_id": STUDY_ID})
    conversation_docs.delete_one({"study_id": STUDY_ID})


TURNS = [
    "I feel really angry when I think about Republican supporters.",
    "The news just makes it worse — I see things that infuriate me every day.",
    "Mostly cable news and Twitter, I guess. That's where I get most of it.",
    "Now that you say it, maybe I don't actually know that many Republicans personally.",
]


def run():
    setup()
    failures = 0

    print("\n=== Sending turns through API ===")
    responses = []
    for i, msg in enumerate(TURNS, 1):
        print(f"\nTurn {i}: \"{msg[:60]}...\"" if len(msg) > 60 else f"\nTurn {i}: \"{msg}\"")
        try:
            reply = send_turn(msg)
            responses.append(reply)
            print(f"  Agent: \"{reply[:80]}...\"" if len(reply) > 80 else f"  Agent: \"{reply}\"")
        except Exception as e:
            print(f"  ERROR: {e}")
            failures += 1

    print("\n=== Verifying MongoDB document ===")
    doc = conversation_docs.find_one({"study_id": STUDY_ID})

    failures += not check("Document exists in conversations collection", doc is not None)
    if not doc:
        cleanup()
        return failures

    payload = doc.get("payload", {})
    signal_history = doc.get("signal_history", [])
    sent = len(responses)  # only count successful turns

    # payload checks
    failures += not check(
        f"payload.turn == {sent} (latest turn number)",
        payload.get("turn") == sent,
        str(payload.get("turn")),
    )
    failures += not check(
        f"payload.messages has {sent * 2} messages (full history)",
        len(payload.get("messages", [])) == sent * 2,
        str(len(payload.get("messages", []))),
    )
    failures += not check(
        "payload.strategy == 'common_identity'",
        payload.get("strategy") == "common_identity",
        str(payload.get("strategy")),
    )
    failures += not check(
        "payload.system_prompt is non-empty",
        bool(payload.get("system_prompt")),
    )
    failures += not check(
        "payload.signals is a dict",
        isinstance(payload.get("signals"), dict),
        str(type(payload.get("signals"))),
    )

    # signal_history checks
    failures += not check(
        f"signal_history has {sent} snapshots (one per turn)",
        len(signal_history) == sent,
        str(len(signal_history)),
    )
    if signal_history:
        failures += not check(
            "signal_history[0] has turn=1",
            signal_history[0].get("turn") == 1,
            str(signal_history[0].get("turn")),
        )
        failures += not check(
            "Every snapshot has exactly {turn, stage, timestamp, signals}",
            all(set(s.keys()) == {"turn", "stage", "timestamp", "signals"} for s in signal_history),
            str([list(s.keys()) for s in signal_history]),
        )
        failures += not check(
            "No snapshot contains 'messages'",
            all("messages" not in s for s in signal_history),
        )
        failures += not check(
            "No snapshot contains 'system_prompt'",
            all("system_prompt" not in s for s in signal_history),
        )
        failures += not check(
            "signal_history stages are non-empty strings",
            all(isinstance(s.get("stage"), str) and s.get("stage") for s in signal_history),
            str([s.get("stage") for s in signal_history]),
        )

    print("\n=== Signal progression across turns ===")
    for i, snap in enumerate(signal_history, 1):
        print(f"  Turn {i} | stage={snap['stage']} | signals={snap['signals']}")

    print(f"\n{'='*40}")
    if failures == 0:
        print("\033[92mAll checks passed.\033[0m")
    else:
        print(f"\033[91m{failures} check(s) failed.\033[0m")

    cleanup()
    return failures


if __name__ == "__main__":
    sys.exit(run())
