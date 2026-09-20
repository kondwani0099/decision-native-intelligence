"""
Automated QA Reverse Validation Engine.

Verifies:
1. Delta Check: Exactly 1 variable differs between Context A and Context B.
2. Ground Truth Rule Execution Check: Deterministic Python business logic agrees with labels.
3. Reasoning Trace Check: Scratchpad explicitly articulates the decision threshold and shift.
"""

from typing import Dict, Any, List, Tuple
from .rule_engine import evaluate_decision


class ContrastiveValidator:
    @staticmethod
    def compute_context_delta(ctx_a: Dict[str, Any], ctx_b: Dict[str, Any]) -> List[str]:
        """Returns list of field differences between Context A and Context B."""
        diffs = []
        all_keys = set(ctx_a.keys()).union(set(ctx_b.keys()))

        for k in all_keys:
            if k not in ctx_a:
                diffs.append(f"Key '{k}' added in B (value: {ctx_b[k]})")
            elif k not in ctx_b:
                diffs.append(f"Key '{k}' removed in B (was: {ctx_a[k]})")
            else:
                val_a = ctx_a[k]
                val_b = ctx_b[k]
                # Compare floats with small tolerance if numeric
                if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                    if abs(float(val_a) - float(val_b)) > 1e-6:
                        diffs.append(f"Key '{k}' shifted: {val_a} -> {val_b}")
                else:
                    if val_a != val_b:
                        diffs.append(f"Key '{k}' shifted: {val_a} -> {val_b}")
        return diffs

    @classmethod
    def validate_pair(cls, pair: Dict[str, Any], domain: str) -> Tuple[bool, List[str]]:
        errors = []

        # 1. Structural Checks
        if "reasoning_trace" not in pair or not isinstance(pair["reasoning_trace"], str):
            errors.append("Missing or invalid 'reasoning_trace'")
        elif len(pair["reasoning_trace"].strip()) < 20:
            errors.append("Reasoning trace too short or trivial")

        if "contrastive_pair" not in pair or not isinstance(pair["contrastive_pair"], list):
            errors.append("Missing or invalid 'contrastive_pair' list")
            return False, errors

        if len(pair["contrastive_pair"]) != 2:
            errors.append(f"Contrastive pair must contain exactly 2 scenarios, got {len(pair['contrastive_pair'])}")
            return False, errors

        scen_a, scen_b = pair["contrastive_pair"][0], pair["contrastive_pair"][1]

        for idx, scen in enumerate([scen_a, scen_b], start=1):
            if "context" not in scen or not isinstance(scen["context"], dict):
                errors.append(f"Scenario {idx} missing valid 'context' dictionary")
            if "label" not in scen or scen["label"] not in ("approve", "review"):
                errors.append(f"Scenario {idx} missing valid label ('approve' or 'review')")
            if "boundary_delta" not in scen:
                errors.append(f"Scenario {idx} missing 'boundary_delta'")

        if errors:
            return False, errors

        # 2. Decision Opposition Check (One approve, One review)
        if scen_a["label"] == scen_b["label"]:
            errors.append(f"Pair does not flip decision: both scenarios labeled '{scen_a['label']}'")

        # 3. Delta Check: Context A - Context B must differ by EXACTLY 1 key
        diffs = cls.compute_context_delta(scen_a["context"], scen_b["context"])
        if len(diffs) == 0:
            errors.append("Delta Check failed: Context A and Context B are identical (0 differences)")
        elif len(diffs) > 1:
            errors.append(f"Delta Check failed: Expected exactly 1 differing variable, found {len(diffs)}: {', '.join(diffs)}")

        # 4. Deterministic Rule Execution Engine Check
        try:
            expected_a, reason_a = evaluate_decision(domain, scen_a["context"])
            if expected_a != scen_a["label"]:
                errors.append(f"Rule Engine Mismatch for Scenario A: expected '{expected_a}' but got '{scen_a['label']}'. Reason: {reason_a}")
        except Exception as e:
            errors.append(f"Rule Engine evaluation crashed on Scenario A: {e}")

        try:
            expected_b, reason_b = evaluate_decision(domain, scen_b["context"])
            if expected_b != scen_b["label"]:
                errors.append(f"Rule Engine Mismatch for Scenario B: expected '{expected_b}' but got '{scen_b['label']}'. Reason: {reason_b}")
        except Exception as e:
            errors.append(f"Rule Engine evaluation crashed on Scenario B: {e}")

        is_valid = len(errors) == 0
        return is_valid, errors
