"""Integration test — verifies MongoDB document structure against real Atlas instance.

Run with: python tests/integration_test_mongodb.py

Cleans up after itself (deletes the test document on exit).
"""
import sys
import os
from pathlib import Path
from datetime import datetime, timezone

# Load .env so MONGODB_URI is available
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.documents import conversation_docs
from app.db.conversation import save_turn_log, get_chat_history, get_conversation_observation

STUDY_ID = "__integration_test__"
PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"


def check(label: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    print(f"{status}  {label}" + (f"\n       got: {detail}" if not condition and detail else ""))
    return condition


def make_entry(turn: int, stage: str, signals: dict, messages: list) -> dict:
    return {
        "turn": turn,
        "stage": stage,
        "stage_turn_count": turn,
        "strategy": "common_identity",
        "political_party": "democrat",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system_prompt": f"system prompt for {stage}",
        "messages": messages,
        "signals": signals,
    }


def cleanup():
    conversation_docs.delete_one({"study_id": STUDY_ID})


def run():
    cleanup()  # ensure clean slate
    failures = 0

    # --- Turn 1 ---
    msgs_t1 = [
        {"role": "user",      "content": "I feel really frustrated with Republicans."},
        {"role": "assistant", "content": "That's interesting — what makes you feel that way?"},
    ]
    entry1 = make_entry(1, "stage_1", {}, msgs_t1)
    save_turn_log(STUDY_ID, entry1)

    # --- Turn 2 ---
    msgs_t2 = msgs_t1 + [
        {"role": "user",      "content": "The news just makes it worse."},
        {"role": "assistant", "content": "Where does most of your political information come from?"},
    ]
    entry2 = make_entry(2, "stage_2", {"feeling_expressed": True, "user_feeling_text": "frustrated"}, msgs_t2)
    save_turn_log(STUDY_ID, entry2)

    # --- Turn 3 ---
    msgs_t3 = msgs_t2 + [
        {"role": "user",      "content": "Mostly cable news and Twitter."},
        {"role": "assistant", "content": "How much of your picture of Republicans comes from there vs. real people?"},
    ]
    entry3 = make_entry(3, "stage_2", {
        "feeling_expressed": True,
        "user_feeling_text": "frustrated",
        "media_distortion_attempted": True,
        "user_media_text": "cable news and Twitter",
    }, msgs_t3)
    save_turn_log(STUDY_ID, entry3)

    # --- Read back from MongoDB ---
    doc = conversation_docs.find_one({"study_id": STUDY_ID})

    if not doc:
        print(f"{FAIL}  Document not found in MongoDB — check MONGODB_URI")
        return 1

    payload = doc.get("payload", {})
    signal_history = doc.get("signal_history", [])

    print("\n=== payload (latest turn) ===")
    failures += not check(
        "payload.turn == 3 (latest turn)",
        payload.get("turn") == 3,
        str(payload.get("turn")),
    )
    failures += not check(
        "payload.stage == 'stage_2'",
        payload.get("stage") == "stage_2",
        str(payload.get("stage")),
    )
    failures += not check(
        "payload.messages has all 6 messages (full history)",
        len(payload.get("messages", [])) == 6,
        str(len(payload.get("messages", []))),
    )
    failures += not check(
        "payload.signals has 4 keys (latest accumulated signals)",
        len(payload.get("signals", {})) == 4,
        str(payload.get("signals")),
    )

    print("\n=== signal_history (one snapshot per turn) ===")
    failures += not check(
        "signal_history has 3 entries (one per turn)",
        len(signal_history) == 3,
        str(len(signal_history)),
    )
    failures += not check(
        "signal_history[0] has turn=1, stage='stage_1', empty signals",
        signal_history[0].get("turn") == 1
        and signal_history[0].get("stage") == "stage_1"
        and signal_history[0].get("signals") == {},
        str(signal_history[0] if signal_history else "empty"),
    )
    failures += not check(
        "signal_history[1] has feeling_expressed=True",
        signal_history[1].get("signals", {}).get("feeling_expressed") is True,
        str(signal_history[1].get("signals") if len(signal_history) > 1 else "missing"),
    )
    failures += not check(
        "signal_history[2] has media_distortion_attempted=True (accumulated)",
        signal_history[2].get("signals", {}).get("media_distortion_attempted") is True,
        str(signal_history[2].get("signals") if len(signal_history) > 2 else "missing"),
    )
    failures += not check(
        "signal_history snapshots have no 'messages' key",
        all("messages" not in s for s in signal_history),
        str([s.keys() for s in signal_history]),
    )
    failures += not check(
        "signal_history snapshots have no 'system_prompt' key",
        all("system_prompt" not in s for s in signal_history),
        str([s.keys() for s in signal_history]),
    )

    print("\n=== get_chat_history (reads payload.messages) ===")
    history = get_chat_history(STUDY_ID)
    failures += not check(
        "get_chat_history returns 6 messages",
        len(history) == 6,
        str(len(history)),
    )
    failures += not check(
        "last message is the latest assistant response",
        history[-1].role == "assistant" and "real people" in history[-1].content,
        str(history[-1] if history else "empty"),
    )

    print("\n=== get_conversation_observation (reads payload signals) ===")
    obs = get_conversation_observation(STUDY_ID)
    failures += not check(
        "observation is not None",
        obs is not None,
    )
    if obs:
        failures += not check(
            "observation.user_feeling_text is None (not set)",
            obs.observation.user_feeling_text is None,
            str(obs.observation),
        )

    print(f"\n{'='*40}")
    if failures == 0:
        print(f"\033[92mAll checks passed.\033[0m")
    else:
        print(f"\033[91m{failures} check(s) failed.\033[0m")

    cleanup()
    return failures


if __name__ == "__main__":
    sys.exit(run())
