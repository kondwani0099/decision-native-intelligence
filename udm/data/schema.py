"""
Canonical Decision Format Serializer.

Converts raw JSON decision contexts into the UDM canonical text representation.
This is the universal interface that standardizes all decision problems
into a format the model has been trained to parse.

Example output:
    <DECISION>
    TYPE: classification
    TASK: Determine whether this expense requires approval or review.
    OPTIONS:
    0: approve
    1: review
    CONTEXT:
    department: operations
    amount: 12500.0
    budget_remaining: 42000
    expense_type: equipment
    vendor: Lusaka Tech Supply
    </DECISION>
"""

from typing import Dict, Any, List, Optional
import json


# Domain-specific task descriptions
TASK_DESCRIPTIONS = {
    "finance": {
        "classification": "Determine whether this expense requires approval or review based on corporate policy thresholds.",
    },
    "sales": {
        "classification": "Determine whether this sales order should be approved or routed for review based on discount and credit policies.",
    },
    "inventory": {
        "classification": "Determine whether this inventory movement should be approved or requires supervisory review.",
    },
}

# Standard options by domain
DOMAIN_OPTIONS = {
    "finance": ["approve", "review"],
    "sales": ["approve", "review"],
    "inventory": ["approve", "review"],
}

# Fields to serialize by domain (controls ordering for consistency)
DOMAIN_CONTEXT_FIELDS = {
    "finance": ["department", "vendor", "expense_type", "amount", "budget_remaining"],
    "sales": [
        "customer_id", "customer_tier", "order_value", "discount_percent",
        "credit_limit", "outstanding_balance", "compliance_status",
    ],
    "inventory": [
        "item_code", "movement_type", "quantity", "unit_cost",
        "source_warehouse", "safety_stock_threshold", "current_stock",
        "batch_number", "destination_bin",
    ],
}


def serialize_decision(
    context: Dict[str, Any],
    domain: str,
    decision_type: str = "classification",
    options: Optional[List[str]] = None,
    task_description: Optional[str] = None,
    field_order: Optional[List[str]] = None,
) -> str:
    """
    Serialize a decision context into the canonical <DECISION> format.

    Args:
        context: raw context dictionary (e.g., {"amount": 12500, "department": "operations"})
        domain: "finance", "sales", "inventory"
        decision_type: "classification", "regression", "ranking"
        options: list of option labels (for classification)
        task_description: override for the task description
        field_order: override for context field ordering

    Returns:
        Canonical decision text string
    """
    # Resolve defaults
    if options is None:
        options = DOMAIN_OPTIONS.get(domain, ["approve", "review"])
    if task_description is None:
        domain_tasks = TASK_DESCRIPTIONS.get(domain, {})
        task_description = domain_tasks.get(
            decision_type,
            f"Make a {decision_type} decision based on the provided context."
        )
    if field_order is None:
        field_order = DOMAIN_CONTEXT_FIELDS.get(domain, sorted(context.keys()))

    # Build canonical text
    lines = ["<DECISION>"]
    lines.append(f"TYPE: {decision_type}")
    lines.append(f"TASK: {task_description}")

    # Options section
    if decision_type == "classification" and options:
        lines.append("OPTIONS:")
        for i, opt in enumerate(options):
            lines.append(f"{i}: {opt}")

    # Context section
    lines.append("CONTEXT:")
    for field in field_order:
        if field in context:
            value = context[field]
            # Format numbers cleanly
            if isinstance(value, float):
                if value == int(value):
                    value = f"{value:.1f}"
                else:
                    value = f"{value:.2f}"
            lines.append(f"{field}: {value}")

    lines.append("</DECISION>")

    return "\n".join(lines)


def serialize_from_jsonl_record(record: Dict[str, Any]) -> str:
    """
    Convenience: serialize a record from the SFT training JSONL format.

    Expected keys: domain, label, and nested context dict extracted from user_prompt.
    """
    domain = record.get("domain", "finance")
    decision_type = "classification"  # Current dataset is classification-only

    # Extract context — it may be in 'context' key or embedded in user_prompt JSON
    if "context" in record:
        context = record["context"]
    else:
        # Fallback: try parsing the context from user_prompt
        context = _extract_context_from_prompt(record.get("user_prompt", ""))

    return serialize_decision(
        context=context,
        domain=domain,
        decision_type=decision_type,
    )


def _extract_context_from_prompt(prompt: str) -> Dict[str, Any]:
    """Extract JSON context dictionary from an SFT training prompt string."""
    try:
        # Find JSON block in the prompt
        start = prompt.find("{")
        end = prompt.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(prompt[start:end])
    except (json.JSONDecodeError, ValueError):
        pass
    return {}


def deserialize_decision(canonical_text: str) -> Dict[str, Any]:
    """
    Parse a canonical decision text back into structured components.

    Returns:
        Dict with keys: type, task, options, context
    """
    result = {"type": "", "task": "", "options": [], "context": {}}
    current_section = None

    for line in canonical_text.strip().split("\n"):
        line = line.strip()
        if line in ("<DECISION>", "</DECISION>"):
            continue
        elif line.startswith("TYPE:"):
            result["type"] = line.split(":", 1)[1].strip()
        elif line.startswith("TASK:"):
            result["task"] = line.split(":", 1)[1].strip()
        elif line == "OPTIONS:":
            current_section = "options"
        elif line == "CONTEXT:":
            current_section = "context"
        elif current_section == "options" and ":" in line:
            _, opt_label = line.split(":", 1)
            result["options"].append(opt_label.strip())
        elif current_section == "context" and ":" in line:
            key, val = line.split(":", 1)
            val = val.strip()
            # Try to parse as number
            try:
                val = float(val)
                if val == int(val) and "." not in line.split(":", 1)[1].strip():
                    val = int(val)
            except ValueError:
                pass
            result["context"][key.strip()] = val

    return result
