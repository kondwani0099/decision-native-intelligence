"""
Deterministic Rule Execution Engine for Ground Truth Reverse Validation.

Covers 3 enterprise domains:
1. Finance & Expense Routing
2. Sales & Credit / Discount Approval
3. Inventory Movements & Stock Adjustments
"""

from typing import Dict, Any, Tuple, Optional


class FinanceRuleEngine:
    """
    Business Rules for Finance / Expense Routing:
    - Rule 1: Missing required fields ('department', 'vendor', 'expense_type', 'amount') -> 'review' (fallback)
    - Rule 2: If 'budget_remaining' is specified and amount > budget_remaining -> 'review'
    - Rule 3: Equipment expenses ('expense_type' == 'equipment'):
        - amount > 10,000 ZMW -> 'review'
        - amount <= 10,000 ZMW -> 'approve'
    - Rule 4: Non-equipment expenses ('software_license', 'office_supplies', 'routine_maintenance', 'travel', 'utilities'):
        - amount <= 25,000 ZMW -> 'approve'
        - amount > 25,000 ZMW -> 'review'
    """

    REQUIRED_FIELDS = {"department", "vendor", "expense_type", "amount"}
    EQUIPMENT_THRESHOLD = 10000.0
    GENERAL_THRESHOLD = 25000.0

    @classmethod
    def evaluate(cls, context: Dict[str, Any]) -> Tuple[str, str]:
        # Check missing required fields
        for field in cls.REQUIRED_FIELDS:
            if field not in context or context[field] is None or str(context[field]).strip() == "":
                return "review", f"Missing mandatory field '{field}' - fallback to review"

        try:
            amount = float(context["amount"])
        except (ValueError, TypeError):
            return "review", "Invalid amount format - fallback to review"

        expense_type = str(context["expense_type"]).lower().strip()

        # Check department budget if provided
        budget_remaining = context.get("budget_remaining")
        if budget_remaining is not None:
            try:
                budget = float(budget_remaining)
                if amount > budget:
                    return "review", f"Amount ({amount:.2f} ZMW) exceeds remaining budget ({budget:.2f} ZMW)"
            except (ValueError, TypeError):
                return "review", "Invalid budget_remaining format"

        if expense_type == "equipment":
            if amount > cls.EQUIPMENT_THRESHOLD:
                return "review", f"Equipment expense ({amount:.2f} ZMW) exceeds threshold of {cls.EQUIPMENT_THRESHOLD:.2f} ZMW"
            else:
                return "approve", f"Equipment expense ({amount:.2f} ZMW) is within threshold of {cls.EQUIPMENT_THRESHOLD:.2f} ZMW"
        else:
            if amount > cls.GENERAL_THRESHOLD:
                return "review", f"{expense_type.replace('_', ' ').capitalize()} expense ({amount:.2f} ZMW) exceeds standard limit of {cls.GENERAL_THRESHOLD:.2f} ZMW"
            else:
                return "approve", f"{expense_type.replace('_', ' ').capitalize()} expense ({amount:.2f} ZMW) is approved under standard policy"


class SalesRuleEngine:
    """
    Business Rules for Sales / Discount & Credit Approvals:
    - Rule 1: Missing required fields ('customer_id', 'customer_tier', 'order_value', 'discount_percent', 'credit_limit', 'outstanding_balance') -> 'review' (fallback)
    - Rule 2: Unverified or high-risk customer status ('compliance_status' == 'flagged') -> 'review'
    - Rule 3: Customer tier bypass:
        - 'tier_1_enterprise' or 'strategic_partner' with discount <= 25.0% and exposure <= credit_limit * 1.2 -> 'approve'
    - Rule 4: Discount threshold:
        - discount_percent > 15.0% -> 'review' (requires sales director approval)
    - Rule 5: Credit exposure threshold:
        - total_exposure = outstanding_balance + order_value
        - if total_exposure > credit_limit -> 'review'
    - Otherwise -> 'approve'
    """

    REQUIRED_FIELDS = {"customer_id", "customer_tier", "order_value", "discount_percent", "credit_limit", "outstanding_balance"}
    DISCOUNT_THRESHOLD = 15.0

    @classmethod
    def evaluate(cls, context: Dict[str, Any]) -> Tuple[str, str]:
        for field in cls.REQUIRED_FIELDS:
            if field not in context or context[field] is None or str(context[field]).strip() == "":
                return "review", f"Missing mandatory field '{field}' - fallback to review"

        if str(context.get("compliance_status", "verified")).lower() == "flagged":
            return "review", "Customer compliance status is flagged for review"

        try:
            order_value = float(context["order_value"])
            discount_percent = float(context["discount_percent"])
            credit_limit = float(context["credit_limit"])
            outstanding_balance = float(context["outstanding_balance"])
        except (ValueError, TypeError) as e:
            return "review", f"Invalid numerical field format: {e}"

        customer_tier = str(context["customer_tier"]).lower().strip()
        total_exposure = outstanding_balance + order_value

        # Tier 1 Enterprise exemption (can take up to 25% discount and 120% credit limit)
        if customer_tier in ("tier_1_enterprise", "strategic_partner"):
            if discount_percent > 25.0:
                return "review", f"Enterprise discount ({discount_percent:.1f}%) exceeds maximum enterprise limit of 25.0%"
            if total_exposure > credit_limit * 1.2:
                return "review", f"Total credit exposure ({total_exposure:.2f} ZMW) exceeds enterprise buffer of {(credit_limit * 1.2):.2f} ZMW"
            return "approve", "Approved under Tier 1 Enterprise preferred commercial terms"

        # Standard boundary checks
        if discount_percent > cls.DISCOUNT_THRESHOLD:
            return "review", f"Discount rate ({discount_percent:.1f}%) exceeds standard threshold of {cls.DISCOUNT_THRESHOLD:.1f}%"

        if total_exposure > credit_limit:
            return "review", f"Total credit exposure ({total_exposure:.2f} ZMW) exceeds customer credit limit ({credit_limit:.2f} ZMW)"

        return "approve", f"Discount ({discount_percent:.1f}%) and credit exposure ({total_exposure:.2f} ZMW) within approved limits"


class InventoryRuleEngine:
    """
    Business Rules for Inventory Movements & Stock Adjustments:
    - Rule 1: Missing required fields ('item_code', 'movement_type', 'quantity', 'unit_cost', 'source_warehouse', 'safety_stock_threshold', 'current_stock') -> 'review'
    - Rule 2: Movement type 'write_off' / 'damaged_scrap':
        - total_loss_value = quantity * unit_cost
        - total_loss_value > 5,000 ZMW or quantity > 50 -> 'review'
        - total_loss_value <= 5,000 ZMW and quantity <= 50 -> 'approve'
    - Rule 3: Movement type 'transfer' / 'internal_replenishment':
        - post_transfer_stock = current_stock - quantity
        - if post_transfer_stock < safety_stock_threshold -> 'review' (breaches safety stock buffer)
        - if post_transfer_stock >= safety_stock_threshold -> 'approve'
    - Rule 4: Movement type 'supplier_receipt' / 'production_inflow':
        - missing 'batch_number' -> 'review'
        - otherwise -> 'approve'
    """

    REQUIRED_FIELDS = {"item_code", "movement_type", "quantity", "unit_cost", "source_warehouse", "safety_stock_threshold", "current_stock"}
    WRITE_OFF_VALUE_THRESHOLD = 5000.0
    WRITE_OFF_QTY_THRESHOLD = 50

    @classmethod
    def evaluate(cls, context: Dict[str, Any]) -> Tuple[str, str]:
        for field in cls.REQUIRED_FIELDS:
            if field not in context or context[field] is None or str(context[field]).strip() == "":
                return "review", f"Missing mandatory field '{field}' - fallback to review"

        try:
            quantity = float(context["quantity"])
            unit_cost = float(context["unit_cost"])
            safety_stock = float(context["safety_stock_threshold"])
            current_stock = float(context["current_stock"])
        except (ValueError, TypeError) as e:
            return "review", f"Invalid numerical field format: {e}"

        movement_type = str(context["movement_type"]).lower().strip()
        total_value = quantity * unit_cost

        if movement_type in ("write_off", "damaged_scrap", "shrinkage_adjustment"):
            if total_value > cls.WRITE_OFF_VALUE_THRESHOLD:
                return "review", f"Write-off value ({total_value:.2f} ZMW) exceeds threshold of {cls.WRITE_OFF_VALUE_THRESHOLD:.2f} ZMW"
            if quantity > cls.WRITE_OFF_QTY_THRESHOLD:
                return "review", f"Write-off quantity ({quantity:.0f} units) exceeds threshold of {cls.WRITE_OFF_QTY_THRESHOLD} units"
            return "approve", f"Write-off ({quantity:.0f} units, {total_value:.2f} ZMW) within routine supervisor adjustment limits"

        elif movement_type in ("transfer", "inter_warehouse_transfer", "internal_replenishment"):
            post_transfer_stock = current_stock - quantity
            if post_transfer_stock < safety_stock:
                return "review", f"Stock after transfer ({post_transfer_stock:.0f} units) falls below safety buffer ({safety_stock:.0f} units)"
            return "approve", f"Transfer leaves safe remaining stock ({post_transfer_stock:.0f} units >= safety threshold {safety_stock:.0f} units)"

        elif movement_type in ("supplier_receipt", "production_inflow"):
            if "batch_number" not in context or not str(context.get("batch_number", "")).strip():
                return "review", "Inbound goods receipt missing required QA batch number"
            return "approve", "Inbound goods receipt verified with batch trace"

        else:
            if total_value > 20000.0:
                return "review", f"Custom movement type '{movement_type}' exceeding 20,000 ZMW requires review"
            return "approve", f"Routine inventory movement approved for '{movement_type}'"


def evaluate_decision(domain: str, context: Dict[str, Any]) -> Tuple[str, str]:
    """Top-level router to evaluate a decision in any domain."""
    d = domain.lower()
    if "finance" in d or "expense" in d:
        return FinanceRuleEngine.evaluate(context)
    elif "sales" in d or "discount" in d or "credit" in d:
        return SalesRuleEngine.evaluate(context)
    elif "inventory" in d or "stock" in d:
        return InventoryRuleEngine.evaluate(context)
    else:
        raise ValueError(f"Unknown domain '{domain}'. Supported: finance, sales, inventory.")
