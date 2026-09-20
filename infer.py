#!/usr/bin/env python3
"""
UDM-125M Inference Runner

Loads a saved model checkpoint and runs fast decision inference on new contexts.

Usage:
    # 1. Run built-in multi-domain demo scenarios:
    python infer.py

    # 2. Run on custom JSON string:
    python infer.py --domain finance --context '{"department":"operations","vendor":"Atlas Mining Tools","expense_type":"equipment","budget_remaining":50000,"amount":9950}'

    # 3. Run on a JSON file containing a scenario or list of scenarios:
    python infer.py --file path/to/scenario.json
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Any

import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from udm.configs.config import UDMConfig
from udm.model.udm import UDMModel
from udm.data.schema import serialize_decision
from udm.data.tokenizer import DecisionTokenizer

LABEL_NAMES = {0: "approve", 1: "review", 2: "reject", 3: "abstain"}


def load_model(checkpoint_dir: str, device: torch.device):
    """Load model, config, and tokenizer from saved export directory."""
    config_path = os.path.join(checkpoint_dir, "config.json")
    model_path = os.path.join(checkpoint_dir, "model.pt")
    tok_path = os.path.join(checkpoint_dir, "tokenizer.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}. Run 'python train.py' first.")

    with open(config_path, "r") as f:
        cfg_dict = json.load(f)
    config = UDMConfig(**cfg_dict)

    tokenizer = DecisionTokenizer.load(tok_path)

    model = UDMModel(config).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, config, tokenizer


def decide(
    model: UDMModel,
    tokenizer: DecisionTokenizer,
    context: Dict[str, Any],
    domain: str,
    device: torch.device,
    decision_type: str = "classification",
) -> Dict[str, Any]:
    """Run decision inference for a single context dictionary."""
    canonical = serialize_decision(
        context=context,
        domain=domain,
        decision_type=decision_type,
    )

    token_ids = tokenizer.encode(canonical, max_length=512)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    t0 = time.perf_counter()
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decision_type=decision_type,
            num_options=2,
        )
    latency_ms = (time.perf_counter() - t0) * 1000

    pred_idx = output.predicted_label.item()
    pred_label = LABEL_NAMES.get(pred_idx, str(pred_idx))
    confidence = output.confidence.item()
    should_abstain = output.should_abstain.item()
    abstain_prob = output.abstain_prob.item()
    probs = output.probabilities[0].cpu().tolist()

    return {
        "decision": pred_label,
        "confidence": confidence,
        "probabilities": {
            "approve": probs[0] if len(probs) > 0 else 0.0,
            "review": probs[1] if len(probs) > 1 else 0.0,
        },
        "abstain": bool(should_abstain),
        "abstain_probability": abstain_prob,
        "latency_ms": latency_ms,
    }


def run_demo(model, tokenizer, device):
    """Run demo examples across Finance, Sales, and Inventory."""
    demos = [
        {
            "domain": "finance",
            "context": {
                "department": "operations",
                "vendor": "Atlas Mining Tools",
                "expense_type": "equipment",
                "budget_remaining": 45000,
                "amount": 9950.0,
            },
            "desc": "Equipment expense <= 10,000 ZMW threshold (Expected: approve)",
        },
        {
            "domain": "finance",
            "context": {
                "department": "operations",
                "vendor": "Atlas Mining Tools",
                "expense_type": "equipment",
                "budget_remaining": 45000,
                "amount": 10050.0,
            },
            "desc": "Equipment expense > 10,000 ZMW threshold (Expected: review)",
        },
        {
            "domain": "sales",
            "context": {
                "customer_id": "CUST-ZM-088",
                "customer_segment": "tier_1_enterprise",
                "order_value": 35000,
                "discount_pct": 14.5,
                "current_outstanding_balance": 20000,
                "credit_limit": 80000,
            },
            "desc": "Discount <= 15% and within credit limit (Expected: approve)",
        },
        {
            "domain": "inventory",
            "context": {
                "item_code": "SKU-COPPER-099",
                "movement_type": "write_off",
                "quantity": 30,
                "unit_cost": 250.0,
                "source_warehouse": "Kitwe Central Depot",
                "safety_stock_threshold": 50,
                "current_stock": 350,
            },
            "desc": "Write-off value 7,500 ZMW > 5,000 threshold (Expected: review)",
        },
    ]

    print("=" * 70)
    print("  UDM-125M INFERENCE DEMO")
    print("=" * 70)

    for i, d in enumerate(demos, 1):
        print(f"\n[{i}] {d['domain'].upper()} — {d['desc']}")
        print(f"    Context: {d['context']}")
        result = decide(model, tokenizer, d["context"], d["domain"], device)
        print(f"    >>> Decision:      {result['decision'].upper()} (confidence: {result['confidence']:.3f})")
        print(f"        Probabilities: approve={result['probabilities']['approve']:.3f}, review={result['probabilities']['review']:.3f}")
        print(f"        Abstain Gate:  {result['abstain']} (escalation prob: {result['abstain_probability']:.3f})")
        print(f"        Latency:       {result['latency_ms']:.2f} ms")


def main():
    parser = argparse.ArgumentParser(description="UDM-125M Inference Runner")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "checkpoints", "udm_125m_final"),
        help="Path to exported checkpoint directory",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="finance",
        choices=["finance", "sales", "inventory"],
        help="Decision domain",
    )
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="JSON string containing decision context",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="JSON file containing decision context",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
    )
    args = parser.parse_args()

    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Loading UDM model from: {args.checkpoint_dir} (device: {device})...")
    model, config, tokenizer = load_model(args.checkpoint_dir, device)
    print("Model loaded successfully.")

    if args.context:
        try:
            ctx = json.loads(args.context)
        except Exception:
            import ast
            ctx = ast.literal_eval(args.context)
        res = decide(model, tokenizer, ctx, args.domain, device)
        print("\nDecision Result:")
        print(json.dumps(res, indent=2))
    elif args.file:
        with open(args.file, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                ctx = item.get("context", item)
                domain = item.get("domain", args.domain)
                res = decide(model, tokenizer, ctx, domain, device)
                print(f"ID {item.get('id', 'N/A')}: {res['decision']} ({res['confidence']:.3f}) in {res['latency_ms']:.2f}ms")
        else:
            ctx = data.get("context", data)
            domain = data.get("domain", args.domain)
            res = decide(model, tokenizer, ctx, domain, device)
            print(json.dumps(res, indent=2))
    else:
        run_demo(model, tokenizer, device)


if __name__ == "__main__":
    main()
