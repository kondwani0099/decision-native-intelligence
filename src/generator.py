"""
Multi-Domain Contrastive Pair Synthetic Data Generator.

Implements:
1. Contrastive Pair Prompting (identical background, exactly 1 shifting variable).
2. Mandatory Reasoning Traces (explicit scratchpad identifying the exact threshold & delta).
3. Dimensional Perturbation ("Swiss Cheese" Method):
   - Value Boundary (tight numerical micro-shifts across threshold)
   - Categorical Boundary (category shift triggering policy bypass/enforcement)
   - Missing Data Boundary (omitted mandatory field triggering fallback rule)
"""

import random
from typing import Dict, Any, List, Tuple
from .rule_engine import FinanceRuleEngine, SalesRuleEngine, InventoryRuleEngine
from .validator import ContrastiveValidator


# --- Domain Constants & Realistic Zambia Contexts ---

DEPARTMENTS = [
    "operations", "engineering", "procurement", "field_logistics",
    "information_technology", "marketing", "finance_accounting",
    "human_resources", "mining_exploration", "fleet_management"
]

FINANCE_VENDORS = [
    "Lusaka Tech Supply Ltd", "Ndola Heavy Spares", "Copperbelt Industrial Supplies",
    "Airtel Networks Zambia", "Liquid Intelligent Technologies", "Stanbic Office Supplies",
    "Kafue Hardware Solutions", "Southern Cross Safety", "Zambezi River IT", "Kitwe Electric Spares",
    "Chingola Tooling Co", "Solwezi Mineral Logistics"
]

NON_EQUIPMENT_TYPES = [
    "software_license", "office_supplies", "routine_maintenance", "travel", "utilities"
]

CUSTOMER_NAMES = [
    "Zambezi Milling Corp", "Copperbelt Agri-Hub", "Victoria Falls Hospitality",
    "Ndola Commercial Retailers", "Kafue River Traders", "Lusaka Mega Supermarkets",
    "Choma Grain Merchants", "Luanshya Mining Contractors", "Solwezi Fuel Logistics",
    "Mpongwe Farm Supplies"
]

WAREHOUSES = [
    "Lusaka Central Distribution", "Ndola Regional Depot", "Kitwe Engineering Stores",
    "Livingstone Transit Hub", "Solwezi Mining Logistics", "Chingola Supply Yard",
    "Kafue Storage Facility"
]

ITEM_DESCRIPTIONS = [
    ("SKU-COP-101", "Hydraulic Hose 2-Inch High Pressure", 250.0),
    ("SKU-COP-204", "Heavy Duty Carbide Drill Bit", 500.0),
    ("SKU-LUS-305", "Industrial Safety Harness Class-4", 125.0),
    ("SKU-NDO-408", "Lithium Grease Cartridge 400g", 50.0),
    ("SKU-KIT-512", "Cisco Gigabit Managed Switch Blade", 2500.0),
    ("SKU-SOL-618", "Solar Inverter Power Module 5kVA", 5000.0),
    ("SKU-LUS-720", "Steel Toe Safety Boots (Box of 10)", 1000.0),
    ("SKU-KAF-830", "Submersible Water Pump Seal Kit", 200.0),
]


class FinanceContrastiveGenerator:
    """Generates contrastive pairs for Finance / Expense Routing."""

    @classmethod
    def generate_value_boundary(cls, pair_id: int) -> Dict[str, Any]:
        """Perturbs amount across the 10,000 ZMW equipment boundary."""
        dept = random.choice(DEPARTMENTS)
        vendor = random.choice(FINANCE_VENDORS)
        budget = random.randint(30000, 85000)

        # Diverse micro-deltas around 10,000 ZMW
        sub_variations = [
            (10000.0, 10001.0, "+1 ZMW"),
            (10000.0, 10005.0, "+5 ZMW"),
            (10000.0, 10015.0, "+15 ZMW"),
            (10000.0, 10050.0, "+50 ZMW"),
            (9995.0, 10010.0, "+15 ZMW crossing 10,000"),
            (9950.0, 10025.0, "+75 ZMW crossing 10,000"),
            (10000.0, 10000.50, "+0.50 ZMW (cent boundary)"),
            (9999.0, 10002.0, "+3 ZMW crossing 10,000"),
        ]
        amt_a, amt_b, delta_desc = random.choice(sub_variations)

        base_ctx = {
            "department": dept,
            "vendor": vendor,
            "expense_type": "equipment",
            "budget_remaining": budget,
            "amount": amt_a
        }
        perturbed_ctx = dict(base_ctx, amount=amt_b)

        trace = (
            f"Business rule threshold: Equipment expenses <= 10,000 ZMW are 'approve', while expenses > 10,000 ZMW "
            f"require 'review'. Department '{dept}', vendor '{vendor}', and remaining budget {budget} ZMW remain identical. "
            f"In Scenario A, amount {amt_a:.2f} ZMW is <= 10,000.00 ZMW (label: approve). "
            f"In Scenario B, amount is shifted by {delta_desc} to {amt_b:.2f} ZMW, strictly crossing the boundary (label: review)."
        )

        return {
            "id": f"fin_val_{pair_id:04d}",
            "domain": "finance",
            "perturbation_dimension": "value_boundary",
            "reasoning_trace": trace,
            "contrastive_pair": [
                {
                    "id": f"fin_val_{pair_id:04d}_A",
                    "decision_type": "classification",
                    "context": base_ctx,
                    "label": "approve",
                    "boundary_delta": "base (at/below threshold)"
                },
                {
                    "id": f"fin_val_{pair_id:04d}_B",
                    "decision_type": "classification",
                    "context": perturbed_ctx,
                    "label": "review",
                    "boundary_delta": delta_desc
                }
            ]
        }

    @classmethod
    def generate_categorical_boundary(cls, pair_id: int) -> Dict[str, Any]:
        """Keeps amount static above 10,000 ZMW; alters category between equipment and standard."""
        dept = random.choice(DEPARTMENTS)
        vendor = random.choice(FINANCE_VENDORS)
        budget = random.randint(35000, 90000)
        amount = random.choice([11500.0, 12800.0, 14200.0, 15000.0, 18500.0, 21000.0])
        non_equip = random.choice(NON_EQUIPMENT_TYPES)

        ctx_approve = {
            "department": dept,
            "vendor": vendor,
            "expense_type": non_equip,
            "budget_remaining": budget,
            "amount": amount
        }
        ctx_review = dict(ctx_approve, expense_type="equipment")

        trace = (
            f"Business rule threshold: Equipment expenses > 10,000 ZMW require 'review', whereas {non_equip.replace('_', ' ')} "
            f"under standard corporate limits (<= 25,000 ZMW) are 'approve'. Amount is held static at {amount:.2f} ZMW with {budget} ZMW budget. "
            f"In Scenario A, expense_type is '{non_equip}' (label: approve). "
            f"In Scenario B, changing solely expense_type to 'equipment' triggers the 10,000 ZMW equipment review rule (label: review)."
        )

        return {
            "id": f"fin_cat_{pair_id:04d}",
            "domain": "finance",
            "perturbation_dimension": "categorical_boundary",
            "reasoning_trace": trace,
            "contrastive_pair": [
                {
                    "id": f"fin_cat_{pair_id:04d}_A",
                    "decision_type": "classification",
                    "context": ctx_approve,
                    "label": "approve",
                    "boundary_delta": f"base (category: {non_equip})"
                },
                {
                    "id": f"fin_cat_{pair_id:04d}_B",
                    "decision_type": "classification",
                    "context": ctx_review,
                    "label": "review",
                    "boundary_delta": f"expense_type shifted to 'equipment'"
                }
            ]
        }

    @classmethod
    def generate_missing_data_boundary(cls, pair_id: int) -> Dict[str, Any]:
        """Keeps amount <= 10,000 ZMW; Scenario B omits a mandatory field (department or vendor)."""
        dept = random.choice(DEPARTMENTS)
        vendor = random.choice(FINANCE_VENDORS)
        budget = random.randint(25000, 70000)
        amount = round(random.uniform(3500.0, 9500.0), 2)
        exp_type = random.choice(["equipment", "office_supplies", "software_license"])

        ctx_approve = {
            "department": dept,
            "vendor": vendor,
            "expense_type": exp_type,
            "budget_remaining": budget,
            "amount": amount
        }

        # Omit exactly 1 field
        field_to_omit = random.choice(["department", "vendor"])
        ctx_review = {k: v for k, v in ctx_approve.items() if k != field_to_omit}

        trace = (
            f"Business rule fallback: While amount ({amount:.2f} ZMW) is below threshold, all transaction records require "
            f"both 'department' and 'vendor'. Scenario A has complete data (label: approve). "
            f"Scenario B omits mandatory field '{field_to_omit}', triggering automated fallback routing to review (label: review)."
        )

        return {
            "id": f"fin_mis_{pair_id:04d}",
            "domain": "finance",
            "perturbation_dimension": "missing_data_boundary",
            "reasoning_trace": trace,
            "contrastive_pair": [
                {
                    "id": f"fin_mis_{pair_id:04d}_A",
                    "decision_type": "classification",
                    "context": ctx_approve,
                    "label": "approve",
                    "boundary_delta": "base (complete context)"
                },
                {
                    "id": f"fin_mis_{pair_id:04d}_B",
                    "decision_type": "classification",
                    "context": ctx_review,
                    "label": "review",
                    "boundary_delta": f"omitted mandatory key '{field_to_omit}'"
                }
            ]
        }


class SalesContrastiveGenerator:
    """Generates contrastive pairs for Sales / Discounts & Credit Routing."""

    @classmethod
    def generate_value_boundary(cls, pair_id: int) -> Dict[str, Any]:
        """Varies either discount rate around 15.0% or credit limit exposure."""
        cust_name = random.choice(CUSTOMER_NAMES)
        cust_id = f"CUST-ZAM-{(pair_id * 7 + 101) % 900 + 100}"
        mode = random.choice(["discount_rate", "credit_exposure"])

        if mode == "discount_rate":
            order_val = random.randint(15000, 40000)
            credit_lim = order_val * 3
            out_bal = random.randint(5000, 15000)

            # Micro-deltas crossing 15.0%
            deltas = [
                (15.0, 15.1, "+0.1% discount"),
                (15.0, 15.2, "+0.2% discount"),
                (15.0, 15.5, "+0.5% discount"),
                (14.9, 15.1, "+0.2% discount crossing 15.0%"),
                (14.8, 15.2, "+0.4% discount crossing 15.0%"),
            ]
            disc_a, disc_b, delta_desc = random.choice(deltas)

            ctx_a = {
                "customer_id": cust_id,
                "customer_tier": "standard_retail",
                "order_value": float(order_val),
                "discount_percent": disc_a,
                "credit_limit": float(credit_lim),
                "outstanding_balance": float(out_bal)
            }
            ctx_b = dict(ctx_a, discount_percent=disc_b)

            trace = (
                f"Sales policy boundary: Discounts <= 15.0% are auto-approved for standard accounts, while discounts > 15.0% "
                f"require sales director review. Order value is {order_val} ZMW for '{cust_name}' ({cust_id}). "
                f"In Scenario A, discount is {disc_a:.1f}% <= 15.0% (label: approve). "
                f"In Scenario B, discount shifts by {delta_desc} to {disc_b:.1f}%, crossing the review threshold (label: review)."
            )

        else:  # credit exposure boundary
            credit_lim = float(random.choice([40000, 50000, 60000, 75000]))
            order_val = float(random.choice([15000, 20000, 25000]))
            # Safe balance leaves total_exposure = credit_lim exactly
            safe_bal = credit_lim - order_val
            # Excess balance pushes total over credit_lim by 50 to 500 ZMW
            delta_amt = random.choice([50.0, 100.0, 150.0, 250.0, 500.0])
            excess_bal = safe_bal + delta_amt
            delta_desc = f"+{delta_amt:.0f} ZMW balance crossing credit limit"

            ctx_a = {
                "customer_id": cust_id,
                "customer_tier": "standard_retail",
                "order_value": order_val,
                "discount_percent": 10.0,
                "credit_limit": credit_lim,
                "outstanding_balance": safe_bal
            }
            ctx_b = dict(ctx_a, outstanding_balance=excess_bal)

            trace = (
                f"Commercial credit boundary: Total exposure (outstanding_balance + order_value) must not exceed credit limit of {credit_lim:.0f} ZMW. "
                f"In Scenario A, total exposure is {safe_bal + order_val:.0f} ZMW == credit limit (label: approve). "
                f"In Scenario B, balance increases by {delta_desc} to {excess_bal:.0f} ZMW, creating total exposure of {excess_bal + order_val:.0f} ZMW (label: review)."
            )

        return {
            "id": f"sal_val_{pair_id:04d}",
            "domain": "sales",
            "perturbation_dimension": "value_boundary",
            "reasoning_trace": trace,
            "contrastive_pair": [
                {
                    "id": f"sal_val_{pair_id:04d}_A",
                    "decision_type": "classification",
                    "context": ctx_a,
                    "label": "approve",
                    "boundary_delta": "base (within authorized limits)"
                },
                {
                    "id": f"sal_val_{pair_id:04d}_B",
                    "decision_type": "classification",
                    "context": ctx_b,
                    "label": "review",
                    "boundary_delta": delta_desc
                }
            ]
        }

    @classmethod
    def generate_categorical_boundary(cls, pair_id: int) -> Dict[str, Any]:
        """Toggles customer_tier between 'tier_1_enterprise' and 'standard_retail' at 20% discount."""
        cust_id = f"CUST-ZAM-ENT-{(pair_id * 11 + 200) % 800 + 100}"
        order_val = float(random.randint(25000, 60000))
        credit_lim = order_val * 2.5
        out_bal = float(random.randint(5000, 15000))
        discount = random.choice([18.0, 19.5, 20.0, 22.0, 24.0])

        ctx_approve = {
            "customer_id": cust_id,
            "customer_tier": "tier_1_enterprise",
            "order_value": order_val,
            "discount_percent": discount,
            "credit_limit": credit_lim,
            "outstanding_balance": out_bal
        }
        ctx_review = dict(ctx_approve, customer_tier="standard_retail")

        trace = (
            f"Commercial tier boundary: Discount of {discount:.1f}% exceeds standard 15.0% cutoff, but is authorized for "
            f"'tier_1_enterprise' accounts (approved up to 25.0%). Holding order value ({order_val:.0f} ZMW) and discount constant: "
            f"In Scenario A, tier_1_enterprise qualifies for preferred terms (label: approve). "
            f"In Scenario B, changing tier to 'standard_retail' violates the standard 15.0% discount ceiling (label: review)."
        )

        return {
            "id": f"sal_cat_{pair_id:04d}",
            "domain": "sales",
            "perturbation_dimension": "categorical_boundary",
            "reasoning_trace": trace,
            "contrastive_pair": [
                {
                    "id": f"sal_cat_{pair_id:04d}_A",
                    "decision_type": "classification",
                    "context": ctx_approve,
                    "label": "approve",
                    "boundary_delta": "base (tier_1_enterprise bypass)"
                },
                {
                    "id": f"sal_cat_{pair_id:04d}_B",
                    "decision_type": "classification",
                    "context": ctx_review,
                    "label": "review",
                    "boundary_delta": "customer_tier changed to 'standard_retail'"
                }
            ]
        }

    @classmethod
    def generate_missing_data_boundary(cls, pair_id: int) -> Dict[str, Any]:
        """Omits mandatory customer_id or toggles compliance status."""
        cust_id = f"CUST-ZAM-{(pair_id * 13 + 301) % 900 + 100}"
        order_val = 18000.0
        credit_lim = 60000.0
        out_bal = 10000.0

        ctx_approve = {
            "customer_id": cust_id,
            "customer_tier": "standard_retail",
            "order_value": order_val,
            "discount_percent": 12.0,
            "credit_limit": credit_lim,
            "outstanding_balance": out_bal
        }
        # Omit customer_id
        ctx_review = {k: v for k, v in ctx_approve.items() if k != "customer_id"}

        trace = (
            f"Compliance & KYC boundary: Order discount (12.0%) and credit exposure are within limits, but order cannot be routed "
            f"without verified 'customer_id'. Scenario A contains valid customer identification (label: approve). "
            f"Scenario B omits 'customer_id', triggering mandatory automated compliance fallback to review (label: review)."
        )

        return {
            "id": f"sal_mis_{pair_id:04d}",
            "domain": "sales",
            "perturbation_dimension": "missing_data_boundary",
            "reasoning_trace": trace,
            "contrastive_pair": [
                {
                    "id": f"sal_mis_{pair_id:04d}_A",
                    "decision_type": "classification",
                    "context": ctx_approve,
                    "label": "approve",
                    "boundary_delta": "base (verified customer context)"
                },
                {
                    "id": f"sal_mis_{pair_id:04d}_B",
                    "decision_type": "classification",
                    "context": ctx_review,
                    "label": "review",
                    "boundary_delta": "omitted mandatory key 'customer_id'"
                }
            ]
        }


class InventoryContrastiveGenerator:
    """Generates contrastive pairs for Inventory Movements & Stock Adjustments."""

    @classmethod
    def generate_value_boundary(cls, pair_id: int) -> Dict[str, Any]:
        """Varies write-off value around 5,000 ZMW or transfer stock around safety buffer."""
        wh = random.choice(WAREHOUSES)
        item_code, desc, cost = random.choice(ITEM_DESCRIPTIONS)
        mode = random.choice(["write_off_value", "transfer_safety_stock"])

        if mode == "write_off_value":
            # Target exactly 5,000 ZMW: e.g., qty=20, unit_cost=250.0 -> 5,000 ZMW
            # Or qty=25, unit_cost=200.0 -> 5,000 ZMW
            qty = 20.0
            base_cost = 250.0  # 20 * 250 = 5,000 ZMW exactly
            delta_cost = random.choice([1.0, 2.0, 5.0, 10.0, 25.0])
            excess_cost = base_cost + delta_cost
            delta_desc = f"+{delta_cost * qty:.0f} ZMW write-off value crossing 5,000 ZMW limit"

            ctx_a = {
                "item_code": item_code,
                "movement_type": "write_off",
                "quantity": qty,
                "unit_cost": base_cost,
                "source_warehouse": wh,
                "safety_stock_threshold": 50.0,
                "current_stock": 300.0
            }
            ctx_b = dict(ctx_a, unit_cost=excess_cost)

            trace = (
                f"Inventory write-off boundary: Routine inventory adjustments <= 5,000.00 ZMW (and <= 50 units) are approved by supervisors. "
                f"Values > 5,000.00 ZMW require plant manager audit review. In Scenario A, 20 units @ {base_cost:.2f} ZMW = 5,000.00 ZMW exact (label: approve). "
                f"In Scenario B, unit cost increases by {delta_cost:.2f} ZMW to {excess_cost:.2f} ZMW, pushing loss value to {qty * excess_cost:.2f} ZMW (label: review)."
            )

        else:  # transfer safety stock buffer
            current_stock = 150.0
            safety_threshold = 40.0
            # Safe transfer leaves exactly safety_threshold (150 - 110 = 40 >= 40)
            safe_qty = current_stock - safety_threshold
            # Excess transfer breaches safety stock (e.g. 111 units leaves 39 units < 40)
            excess_delta = random.choice([1.0, 2.0, 5.0, 10.0])
            excess_qty = safe_qty + excess_delta
            delta_desc = f"+{excess_delta:.0f} units transferred breaching safety stock buffer"

            ctx_a = {
                "item_code": item_code,
                "movement_type": "transfer",
                "quantity": safe_qty,
                "unit_cost": cost,
                "source_warehouse": wh,
                "safety_stock_threshold": safety_threshold,
                "current_stock": current_stock
            }
            ctx_b = dict(ctx_a, quantity=excess_qty)

            trace = (
                f"Safety stock transfer boundary: Warehouse replenishment must not deplete current stock below the safety stock threshold of {safety_threshold:.0f} units. "
                f"In Scenario A, transferring {safe_qty:.0f} units leaves {current_stock - safe_qty:.0f} units remaining (== threshold) (label: approve). "
                f"In Scenario B, shifting transfer quantity by {delta_desc} to {excess_qty:.0f} units leaves {current_stock - excess_qty:.0f} units, breaching buffer (label: review)."
            )

        return {
            "id": f"inv_val_{pair_id:04d}",
            "domain": "inventory",
            "perturbation_dimension": "value_boundary",
            "reasoning_trace": trace,
            "contrastive_pair": [
                {
                    "id": f"inv_val_{pair_id:04d}_A",
                    "decision_type": "classification",
                    "context": ctx_a,
                    "label": "approve",
                    "boundary_delta": "base (within authorized limits)"
                },
                {
                    "id": f"inv_val_{pair_id:04d}_B",
                    "decision_type": "classification",
                    "context": ctx_b,
                    "label": "review",
                    "boundary_delta": delta_desc
                }
            ]
        }

    @classmethod
    def generate_categorical_boundary(cls, pair_id: int) -> Dict[str, Any]:
        """Holds quantity and unit cost static; changes movement_type from 'transfer' to 'write_off'."""
        wh = random.choice(WAREHOUSES)
        item_code, desc, cost = random.choice(ITEM_DESCRIPTIONS)
        # Value > 5,000 ZMW (e.g., 7,500 ZMW)
        qty = 30.0
        unit_cost = 250.0  # 30 * 250 = 7,500 ZMW

        ctx_approve = {
            "item_code": item_code,
            "movement_type": "transfer",
            "quantity": qty,
            "unit_cost": unit_cost,
            "source_warehouse": wh,
            "safety_stock_threshold": 20.0,
            "current_stock": 200.0  # 200 - 30 = 170 >> 20
        }
        ctx_review = dict(ctx_approve, movement_type="write_off")

        trace = (
            f"Movement type classification boundary: At 7,500.00 ZMW total value, an inter-warehouse transfer with ample stock buffer "
            f"(170 units remaining vs 20 unit safety threshold) is auto-approved. "
            f"In Scenario A, movement_type is 'transfer' (label: approve). "
            f"In Scenario B, altering solely movement_type to 'write_off' triggers mandatory audit review because 7,500 ZMW exceeds the 5,000 ZMW write-off limit (label: review)."
        )

        return {
            "id": f"inv_cat_{pair_id:04d}",
            "domain": "inventory",
            "perturbation_dimension": "categorical_boundary",
            "reasoning_trace": trace,
            "contrastive_pair": [
                {
                    "id": f"inv_cat_{pair_id:04d}_A",
                    "decision_type": "classification",
                    "context": ctx_approve,
                    "label": "approve",
                    "boundary_delta": "base (movement: transfer)"
                },
                {
                    "id": f"inv_cat_{pair_id:04d}_B",
                    "decision_type": "classification",
                    "context": ctx_review,
                    "label": "review",
                    "boundary_delta": "movement_type shifted to 'write_off'"
                }
            ]
        }

    @classmethod
    def generate_missing_data_boundary(cls, pair_id: int) -> Dict[str, Any]:
        """Omits mandatory source_warehouse field."""
        wh = random.choice(WAREHOUSES)
        item_code, desc, cost = random.choice(ITEM_DESCRIPTIONS)

        ctx_approve = {
            "item_code": item_code,
            "movement_type": "transfer",
            "quantity": 10.0,
            "unit_cost": cost,
            "source_warehouse": wh,
            "safety_stock_threshold": 15.0,
            "current_stock": 100.0
        }
        ctx_review = {k: v for k, v in ctx_approve.items() if k != "source_warehouse"}

        trace = (
            f"Inventory traceability boundary: Stock movements require verified 'source_warehouse' for chain-of-custody logging. "
            f"Scenario A includes valid source warehouse '{wh}' (label: approve). "
            f"Scenario B omits 'source_warehouse', triggering automated rejection/review fallback (label: review)."
        )

        return {
            "id": f"inv_mis_{pair_id:04d}",
            "domain": "inventory",
            "perturbation_dimension": "missing_data_boundary",
            "reasoning_trace": trace,
            "contrastive_pair": [
                {
                    "id": f"inv_mis_{pair_id:04d}_A",
                    "decision_type": "classification",
                    "context": ctx_approve,
                    "label": "approve",
                    "boundary_delta": "base (complete warehouse context)"
                },
                {
                    "id": f"inv_mis_{pair_id:04d}_B",
                    "decision_type": "classification",
                    "context": ctx_review,
                    "label": "review",
                    "boundary_delta": "omitted mandatory key 'source_warehouse'"
                }
            ]
        }


def generate_verified_domain_dataset(domain: str, count: int = 100) -> List[Dict[str, Any]]:
    """Generates and validates contrastive pairs for a specified domain."""
    generators = {
        "finance": (
            FinanceContrastiveGenerator.generate_value_boundary,
            FinanceContrastiveGenerator.generate_categorical_boundary,
            FinanceContrastiveGenerator.generate_missing_data_boundary,
        ),
        "sales": (
            SalesContrastiveGenerator.generate_value_boundary,
            SalesContrastiveGenerator.generate_categorical_boundary,
            SalesContrastiveGenerator.generate_missing_data_boundary,
        ),
        "inventory": (
            InventoryContrastiveGenerator.generate_value_boundary,
            InventoryContrastiveGenerator.generate_categorical_boundary,
            InventoryContrastiveGenerator.generate_missing_data_boundary,
        ),
    }

    gen_val, gen_cat, gen_mis = generators[domain]

    # Target distribution: 40% Value boundary, 35% Categorical boundary, 25% Missing data boundary
    val_target = int(count * 0.40)
    cat_target = int(count * 0.35)
    mis_target = count - val_target - cat_target

    targets = [
        (gen_val, val_target, "value"),
        (gen_cat, cat_target, "categorical"),
        (gen_mis, mis_target, "missing_data")
    ]

    pairs: List[Dict[str, Any]] = []
    pair_counter = 1

    for generator_fn, target_qty, ptype in targets:
        collected = 0
        attempts = 0
        while collected < target_qty and attempts < target_qty * 10:
            attempts += 1
            candidate = generator_fn(pair_counter)
            is_valid, errors = ContrastiveValidator.validate_pair(candidate, domain)
            if is_valid:
                pairs.append(candidate)
                collected += 1
                pair_counter += 1
            else:
                # Log discard in debug
                continue

    return pairs
