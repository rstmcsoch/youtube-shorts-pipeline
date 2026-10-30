"""Heuristic no-personal-names gate for public-facing draft content (CLI).

The scanner lives in verticals/name_gate.py so the same rules run before
rendering (fail fast + one regeneration in verticals.draft) and here, after
rendering, as the final safety net over metadata and burned-in captions.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verticals.name_gate import (  # noqa: E402  (path set above)
    ALLOWED, GIVEN_NAMES, NAME_HINT, NAME_RUN, find_names, flagged_phrases, scan_text, validate_draft,
)

__all__ = ["ALLOWED", "GIVEN_NAMES", "NAME_HINT", "NAME_RUN", "find_names", "flagged_phrases",
           "scan_text", "validate_draft", "main"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", required=True, type=Path)
    parser.add_argument("--captions", action="append", type=Path, default=[])
    args = parser.parse_args()
    try:
        draft = json.loads(args.draft.read_text(encoding="utf-8"))
        issues = validate_draft(draft, args.captions)
    except Exception as exc:
        print("NO-PERSONAL-NAMES GATE: ERROR: " + str(exc), file=sys.stderr)
        return 2
    if issues:
        print("NO-PERSONAL-NAMES GATE: BLOCKED")
        for issue in issues:
            print("- " + issue)
        print("DYPOL_FAILURE_CATEGORY=content_gate")
        return 1
    print("NO-PERSONAL-NAMES GATE: PASS (heuristic scan; human review remains required)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
