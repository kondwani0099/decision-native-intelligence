#!/usr/bin/env python3
"""
Test and Evaluate Saved UDM-125M Model on Curated Decision Examples.

Loads checkpoints/udm_125m_final/ and evaluates:
1. Finance: Within-budget vs Above-threshold expense claims
2. Sales: Compliant discounts vs Boundary-breached discounts & credit limits
3. Inventory: Safe stock transfers vs High-value write-offs
4. Missing Data: Handling missing mandatory fields
"""

import os
import sys
import json
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from infer import load_model, decide

def run_tests():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints", "udm_125m_final")

    print("=" * 80)
    print("  UDM-125M: SAVED MODEL INFERENCE EVALUATION")
    print("=" * 80)
    print(f"Loading checkpoint from: {checkpoint_dir}")
    print(f"Inference device:        {device}\n")

    model, config, tokenizer = load_model(checkpoint_dir, device)

    test_cases = [
        {
            "id": "FIN-01",
            "domain": "finance",
            "title": "Finance: Equipment <= 10,000 ZMW Approval Threshold",
            "context": {
                "department": "information_technology",
                "vendor": "Liquid Intelligent Technologies",
                "expense_type": "equipment",
                "budget_remaining": 35000,
                "amount": 9950.0
            },
            "expected": "APPROVE",
            "rule": "Equipment amount 9,950 ZMW <= 10,000 threshold (Compliant)"
        },
        {
            "id": "FIN-02",
            "domain": "finance",
            "title": "Finance: Equipment > 10,000 ZMW Policy Boundary",
            "context": {
                "department": "information_technology",
                "vendor": "Liquid Intelligent Technologies",
                "expense_type": "equipment",
                "budget_remaining": 35000,
                "amount": 10050.0
            },
            "expected": "REVIEW",
            "rule": "Equipment amount 10,050 ZMW > 10,000 threshold (Breached by +50 ZMW)"
        },
        {
            "id": "SALES-01",
            "domain": "sales",
            "title": "Sales: Order within Discount (14.0%) & Credit Limit",
            "context": {
                "customer_id": "CUST-ZM-101",
                "customer_segment": "tier_1_enterprise",
                "order_value": 28000,
                "discount_pct": 14.0,
                "current_outstanding_balance": 15000,
                "credit_limit": 60000
            },
            "expected": "APPROVE",
            "rule": "Discount 14.0% <= 15.0% and exposure (43,000) <= 60,000 (Compliant)"
        },
        {
            "id": "SALES-02",
            "domain": "sales",
            "title": "Sales: Order Breaching 15.0% Discount Limit",
            "context": {
                "customer_id": "CUST-ZM-102",
                "customer_segment": "tier_1_enterprise",
                "order_value": 28000,
                "discount_pct": 16.5,
                "current_outstanding_balance": 15000,
                "credit_limit": 60000
            },
            "expected": "REVIEW",
            "rule": "Discount 16.5% > 15.0% threshold (Breached by +1.5%)"
        },
        {
            "id": "SALES-03",
            "domain": "sales",
            "title": "Sales: Order Exceeding Total Credit Limit",
            "context": {
                "customer_id": "CUST-ZM-103",
                "customer_segment": "standard_retail",
                "order_value": 35000,
                "discount_pct": 10.0,
                "current_outstanding_balance": 50000,
                "credit_limit": 75000
            },
            "expected": "REVIEW",
            "rule": "Total exposure (50,000 + 35,000 = 85,000) > credit limit (75,000)"
        },
        {
            "id": "INV-01",
            "domain": "inventory",
            "title": "Inventory: Stock Transfer Preserving Safety Buffer",
            "context": {
                "item_code": "SKU-LUSAKA-01",
                "movement_type": "transfer",
                "quantity": 50,
                "unit_cost": 45.0,
                "source_warehouse": "Lusaka Distribution Center",
                "safety_stock_threshold": 40,
                "current_stock": 120
            },
            "expected": "APPROVE",
            "rule": "Stock after transfer (120 - 50 = 70) >= 40 safety buffer"
        },
        {
            "id": "INV-02",
            "domain": "inventory",
            "title": "Inventory: Write-off Exceeding 5,000 ZMW Threshold",
            "context": {
                "item_code": "SKU-KITWE-88",
                "movement_type": "write_off",
                "quantity": 30,
                "unit_cost": 220.0,
                "source_warehouse": "Kitwe Central Depot",
                "safety_stock_threshold": 20,
                "current_stock": 100
            },
            "expected": "REVIEW",
            "rule": "Write-off total (30 * 220 = 6,600 ZMW) > 5,000 ZMW threshold"
        },
        {
            "id": "INV-03",
            "domain": "inventory",
            "title": "Inventory: Omitted Source Warehouse (Missing Data)",
            "context": {
                "item_code": "SKU-NDOLA-12",
                "movement_type": "transfer",
                "quantity": 25,
                "unit_cost": 80.0,
                "safety_stock_threshold": 30,
                "current_stock": 90
            },
            "expected": "REVIEW",
            "rule": "Missing mandatory source warehouse field requires audit escalation"
        }
    ]

    print(f"{'#':<3} | {'ID':<8} | {'DOMAIN':<9} | {'EXPECTED':<8} | {'PREDICTED':<9} | {'CONF':<6} | {'P(APP)':<6} | {'P(REV)':<6} | {'ABSTAIN':<7} | {'LATENCY':<7}")
    print("-" * 95)

    detailed_outputs = []

    for i, case in enumerate(test_cases, 1):
        res = decide(model, tokenizer, case["context"], case["domain"], device)
        pred = res["decision"].upper()
        conf = res["confidence"]
        p_app = res["probabilities"]["approve"]
        p_rev = res["probabilities"]["review"]
        abstain = str(res["abstain"])
        lat = res["latency_ms"]

        print(f"{i:<3} | {case['id']:<8} | {case['domain']:<9} | {case['expected']:<8} | {pred:<9} | {conf:.3f}  | {p_app:.3f}  | {p_rev:.3f}  | {abstain:<7} | {lat:.1f}ms")
        detailed_outputs.append((case, res))

    print("=" * 95)
    print("\n" + "=" * 80)
    print("  DETAILED SCENARIO-BY-SCENARIO BREAKDOWN")
    print("=" * 80)

    for i, (case, res) in enumerate(detailed_outputs, 1):
        print(f"\n--- [Example {i}] {case['id']}: {case['title']} ---")
        print(f"Domain:              {case['domain'].capitalize()}")
        print(f"Input Context:       {json.dumps(case['context'], indent=2)}")
        print(f"Governing Rule:      {case['rule']}")
        print(f"Expected Decision:   {case['expected']}")
        print(f"Model Prediction:    {res['decision'].upper()}")
        print(f"Confidence Score:    {res['confidence']:.2%}")
        print(f"Class Probabilities: P(Approve) = {res['probabilities']['approve']:.3f} | P(Review) = {res['probabilities']['review']:.3f}")
        print(f"Abstention Trigger:  {res['abstain']} (escalation risk probability: {res['abstain_probability']:.3f})")
        print(f"Inference Latency:   {res['latency_ms']:.2f} ms")


if __name__ == "__main__":
    run_tests()
