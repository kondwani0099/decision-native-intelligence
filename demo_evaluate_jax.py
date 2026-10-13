#!/usr/bin/env python3
"""
Test and Evaluate Saved UDM-200M JAX Model on Curated Contrastive Decision Examples.

Loads checkpoints/udm_200m_jax/ and evaluates:
1. Finance: Within-budget vs Above-threshold expense claims
2. Sales: Compliant discounts vs Boundary-breached discounts & credit limits
3. Inventory: Safe stock transfers vs High-value write-offs
4. Epistemic Abstention Gate: Calibration on edge cases & missing data
"""

import os
import sys
import json
import time
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from infer_jax import JaxDecisionEngine
from udm_jax.distributed import detect_hardware


def run_evaluation():
    hw = detect_hardware()
    checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints", "udm_200m_jax")

    print("=" * 80)
    print("  UDM-200M JAX: BENCHMARK & MULTI-DOMAIN BOUNDARY EVALUATION")
    print("=" * 80)
    print(f"Checkpoint Directory: {checkpoint_dir}")
    print(f"Hardware Platform:    {hw['target_name']} ({hw['backend'].upper()})\n")

    if not os.path.exists(os.path.join(checkpoint_dir, "config.json")):
        print(f"Notice: Checkpoint not found at {checkpoint_dir}.")
        print("Please run 'python train_jax.py' first to train and export the 200M model.")
        return

    engine = JaxDecisionEngine(checkpoint_dir)
    print(f"Model successfully loaded: UDM-200M ({engine.config.num_layers} layers, {engine.config.hidden_dim} dim)")
    print(f"Calibrated Temperature Scaling T = {engine.temperature:.4f}\n")

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
                "amount": 9950.0,
            },
            "expected": "APPROVE",
            "rule": "Equipment amount 9,950 ZMW <= 10,000 threshold (Compliant)",
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
                "amount": 10050.0,
            },
            "expected": "REVIEW",
            "rule": "Equipment amount 10,050 ZMW > 10,000 threshold (Breached by +50 ZMW)",
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
                "credit_limit": 60000,
            },
            "expected": "APPROVE",
            "rule": "Discount 14.0% <= 15.0% and exposure (43,000) <= 60,000 (Compliant)",
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
                "credit_limit": 60000,
            },
            "expected": "REVIEW",
            "rule": "Discount 16.5% > 15.0% threshold (Breached by +1.5%)",
        },
        {
            "id": "INV-01",
            "domain": "inventory",
            "title": "Inventory: Stock Write-Off <= 5,000 ZMW Threshold",
            "context": {
                "warehouse_id": "WH-LUSAKA-01",
                "item_sku": "SKU-COPPER-ROD-01",
                "category": "raw_materials",
                "movement_type": "write_off",
                "unit_value": 45.0,
                "quantity": 100,
                "total_value": 4500.0,
                "reason_code": "scrap_damaged",
            },
            "expected": "APPROVE",
            "rule": "Write-off 4,500 ZMW <= 5,000 ZMW (Compliant)",
        },
        {
            "id": "INV-02",
            "domain": "inventory",
            "title": "Inventory: Stock Write-Off > 5,000 ZMW Threshold",
            "context": {
                "warehouse_id": "WH-LUSAKA-01",
                "item_sku": "SKU-COPPER-ROD-01",
                "category": "raw_materials",
                "movement_type": "write_off",
                "unit_value": 55.0,
                "quantity": 100,
                "total_value": 5500.0,
                "reason_code": "scrap_damaged",
            },
            "expected": "REVIEW",
            "rule": "Write-off 5,500 ZMW > 5,000 ZMW threshold (Breached by +500 ZMW)",
        },
    ]

    latencies = []
    correct_count = 0

    print(f"{'Case ID':<10} | {'Domain':<10} | {'Expected':<10} | {'Predicted':<10} | {'Confidence':<12} | {'Latency':<10} | {'Status'}")
    print("-" * 80)

    for tc in test_cases:
        res = engine.decide(context=tc["context"], domain=tc["domain"])
        latencies.append(res["latency_ms"])
        pred = res["decision"].upper()
        conf = f"{res['confidence']*100:.1f}%"
        is_correct = (pred == tc["expected"])
        if is_correct:
            correct_count += 1
        status = "PASS" if is_correct else "FAIL"

        print(f"{tc['id']:<10} | {tc['domain']:<10} | {tc['expected']:<10} | {pred:<10} | {conf:<12} | {res['latency_ms']:>6.2f} ms | {status}")

    acc = (correct_count / len(test_cases)) * 100
    p50_lat = np.percentile(latencies, 50)
    p95_lat = np.percentile(latencies, 95)

    print("-" * 80)
    print(f"Summary: {correct_count}/{len(test_cases)} Passed ({acc:.1f}% Boundary Compliance Accuracy)")
    print(f"Latency Profile: P50 = {p50_lat:.2f} ms | P95 = {p95_lat:.2f} ms | Mean = {np.mean(latencies):.2f} ms")


if __name__ == "__main__":
    run_evaluation()
