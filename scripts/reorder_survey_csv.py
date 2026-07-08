#!/usr/bin/env python3
"""
Reorder columns of a Compass-exported survey CSV to match the actual
survey question order (Compass's schema-based export otherwise lists
pre_survey.*/post_survey.* columns alphabetically).

New records already store keys with a zero-padded sequence prefix (e.g.
"01_AIFrequency"), so they export in order on their own — see
app/survey_order.py. This script mainly matters for older records saved
before that prefix was added.

Usage:
    python scripts/reorder_survey_csv.py input.csv output.csv
"""

import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.survey_order import PRE_SURVEY_ORDER, POST_SURVEY_ORDER

# Fixed metadata columns that should lead the file, in this order.
METADATA_ORDER = [
    "_id",
    "study_id",
    "strategy",
    "state",
    "screened",
    "created_at",
    "updated_at",
]

_PREFIX_RE = re.compile(r"^\d+_")


def _strip_prefix(key: str) -> str:
    """Undo the "01_" sequence prefix app.survey_order.apply_order_prefix
    adds, so already-prefixed and legacy unprefixed keys rank the same way."""
    return _PREFIX_RE.sub("", key)


def _rank(column: str) -> tuple:
    """Sort key placing a CSV column into metadata / pre_survey / post_survey
    canonical order, with unrecognized columns pushed to the end in their
    original relative order."""
    if column in METADATA_ORDER:
        return (0, METADATA_ORDER.index(column))

    if column.startswith("pre_survey."):
        key = _strip_prefix(column[len("pre_survey."):])
        if key in PRE_SURVEY_ORDER:
            return (1, PRE_SURVEY_ORDER.index(key))
        return (1, len(PRE_SURVEY_ORDER))

    if column.startswith("post_survey."):
        key = _strip_prefix(column[len("post_survey."):])
        if key in POST_SURVEY_ORDER:
            return (2, POST_SURVEY_ORDER.index(key))
        return (2, len(POST_SURVEY_ORDER))

    return (3, 0)


def reorder_csv(input_path: str, output_path: str) -> None:
    with open(input_path, newline="", encoding="utf-8-sig") as infile:
        reader = csv.reader(infile)
        header = next(reader)
        rows = list(reader)

    ordered_header = sorted(range(len(header)), key=lambda i: (_rank(header[i]), i))
    new_header = [header[i] for i in ordered_header]

    with open(output_path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(new_header)
        for row in rows:
            writer.writerow([row[i] if i < len(row) else "" for i in ordered_header])


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/reorder_survey_csv.py <input.csv> <output.csv>")
        sys.exit(1)
    reorder_csv(sys.argv[1], sys.argv[2])
    print(f"Wrote reordered CSV to {sys.argv[2]}")
