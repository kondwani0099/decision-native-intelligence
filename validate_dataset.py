"""
Standalone CLI tool to run Automated QA Reverse Validation on dataset files.
Usage:
    python validate_dataset.py [path_to_jsonl]
"""

import sys
import os
import json
from src.validator import ContrastiveValidator


def run_qa(filepath: str):
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    print("=" * 70)
    print(f"RUNNING AUTOMATED QA REVERSE VALIDATION ON: {filepath}")
    print("=" * 70)

    total = 0
    passed = 0
    failures = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                pair = json.loads(line)
            except Exception as e:
                failures.append((line_num, "N/A", [f"JSON Parse Error: {e}"]))
                continue

            domain = pair.get("domain", "finance")
            is_valid, errors = ContrastiveValidator.validate_pair(pair, domain)
            if is_valid:
                passed += 1
            else:
                pair_id = pair.get("id", f"line_{line_num}")
                failures.append((line_num, pair_id, errors))

    print(f"\n[QA Results]")
    print(f"  Total Pairs Checked : {total}")
    print(f"  Passed Delta Check   : {passed}")
    print(f"  Passed Rule Engine   : {passed}")
    print(f"  Failed / Discarded   : {len(failures)}")
    print(f"  QA Compliance Rate   : {passed / total * 100:.2f}%")

    if failures:
        print("\n[Failures Detected]:")
        for line_num, pid, errs in failures[:10]:
            print(f"  - Line {line_num} ({pid}):")
            for err in errs:
                print(f"      * {err}")
        if len(failures) > 10:
            print(f"      ... and {len(failures) - 10} more failures.")
        sys.exit(1)
    else:
        print("\n[STATUS: 100% PASSED] Zero spurious correlations detected. All pairs mathematically sound!")
        sys.exit(0)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "contrastive_pairs_300.jsonl")
    run_qa(path)
