#!/usr/bin/env python3
"""
UDM-200M JAX Sub-Millisecond Inference Engine.

Loads the exported 200M parameter model checkpoint and executes
fast, deterministic decisions on GPUs, Google TPUs, or CPUs.

Usage:
    # 1. Run built-in multi-domain validation scenarios:
    python infer_jax.py

    # 2. Run custom single context decision:
    python infer_jax.py --domain finance --context '{"department":"operations","vendor":"Atlas Mining Tools","expense_type":"equipment","budget_remaining":50000,"amount":9950}'

    # 3. Run from JSON file:
    python infer_jax.py --file path/to/scenario.json
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Any, Tuple, Optional
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from udm_jax.config import UDMJaxConfig
from udm_jax.model.udm import UDMJaxModel
from udm_jax.training.trainer import load_checkpoint
from udm.data.schema import serialize_decision
from udm.data.tokenizer import DecisionTokenizer
from udm_jax.distributed import detect_hardware

import jax
import jax.numpy as jnp

LABEL_NAMES = {0: "approve", 1: "review", 2: "reject", 3: "abstain"}


class JaxDecisionEngine:
    """
    Sub-millisecond JAX Decision Engine with pre-compiled XLA execution.
    """

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        self.config_path = os.path.join(checkpoint_dir, "config.json")
        self.tok_path = os.path.join(checkpoint_dir, "tokenizer.json")
        self.meta_path = os.path.join(checkpoint_dir, "metadata.json")

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration not found in {checkpoint_dir}. Run 'python train_jax.py' first.")

        # Load config
        with open(self.config_path, "r") as f:
            cfg_dict = json.load(f)
        self.config = UDMJaxConfig(**cfg_dict)

        # Load Tokenizer
        if os.path.exists(self.tok_path):
            self.tokenizer = DecisionTokenizer.load(self.tok_path)
        else:
            # Fallback to existing tokenizer if available
            fallback_tok = os.path.join(PROJECT_ROOT, "checkpoints", "udm_125m_final", "tokenizer.json")
            self.tokenizer = DecisionTokenizer.load(fallback_tok)

        # Load metadata / calibration temperature
        self.temperature = 1.0
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r") as f:
                meta = json.load(f)
                self.temperature = meta.get("temperature", 1.0)

        # Initialize model & load parameters
        self.model = UDMJaxModel(config=self.config)
        self.params = load_checkpoint(checkpoint_dir, self.model, self.config)

        # Compile JIT forward function
        @jax.jit
        def _jit_forward(params, input_ids, attention_mask):
            return self.model.apply(
                {"params": params},
                input_ids=input_ids,
                attention_mask=attention_mask,
                decision_type="classification",
                num_options=2,
                deterministic=True,
            )

        self._jit_forward = _jit_forward

        # Warm up JIT kernel
        dummy_ids = jnp.zeros((1, 128), dtype=jnp.int32)
        dummy_mask = jnp.ones((1, 128), dtype=jnp.int32)
        warmup_out = self._jit_forward(self.params, dummy_ids, dummy_mask)
        _ = warmup_out["logits"].block_until_ready()

    def decide(
        self,
        context: Dict[str, Any],
        domain: str,
        decision_type: str = "classification",
        max_seq_len: int = 256,
    ) -> Dict[str, Any]:
        """
        Execute deterministic decision inference on context payload.
        """
        canonical = serialize_decision(
            context=context,
            domain=domain,
            decision_type=decision_type,
        )

        token_ids = self.tokenizer.encode(
            canonical,
            add_bos=True,
            add_eos=True,
            task_type=decision_type,
            max_length=max_seq_len,
        )

        pad_id = self.tokenizer.pad_token_id
        if len(token_ids) < max_seq_len:
            mask = [1] * len(token_ids) + [0] * (max_seq_len - len(token_ids))
            token_ids = token_ids + [pad_id] * (max_seq_len - len(token_ids))
        else:
            token_ids = token_ids[:max_seq_len]
            mask = [1] * max_seq_len

        input_ids = jnp.array([token_ids], dtype=jnp.int32)
        attention_mask = jnp.array([mask], dtype=jnp.int32)

        t0 = time.perf_counter()
        out = self._jit_forward(self.params, input_ids, attention_mask)
        _ = out["logits"].block_until_ready()
        latency_ms = (time.perf_counter() - t0) * 1000

        logits = np.array(out["logits"][0])
        # Apply temperature scaling
        scaled_logits = logits / max(self.temperature, 1e-4)
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        calibrated_probs = exp_logits / np.sum(exp_logits)

        pred_idx = int(np.argmax(calibrated_probs))
        pred_label = LABEL_NAMES.get(pred_idx, str(pred_idx))
        confidence = float(np.max(calibrated_probs))
        abstain_prob = float(out["abstain_prob"][0])
        should_abstain = bool(abstain_prob > self.config.abstention_threshold)

        return {
            "decision": pred_label,
            "confidence": round(confidence, 4),
            "probabilities": {
                LABEL_NAMES.get(i, str(i)): round(float(p), 4)
                for i, p in enumerate(calibrated_probs)
            },
            "abstain_gate": {
                "abstain": should_abstain,
                "abstain_probability": round(abstain_prob, 4),
                "threshold": self.config.abstention_threshold,
            },
            "latency_ms": round(latency_ms, 3),
            "canonical_format": canonical,
        }


def run_builtin_demos(engine: JaxDecisionEngine):
    """Run verification scenarios across Finance, Sales, and Inventory."""
    scenarios = [
        {
            "domain": "finance",
            "title": "Finance: Equipment Claim <= 10,000 ZMW (Boundary Compliant)",
            "context": {
                "department": "information_technology",
                "vendor": "Liquid Intelligent Technologies",
                "expense_type": "equipment",
                "budget_remaining": 35000,
                "amount": 9950.0,
            },
            "expected": "approve",
        },
        {
            "domain": "finance",
            "title": "Finance: Equipment Claim > 10,000 ZMW (Boundary Exceeded)",
            "context": {
                "department": "information_technology",
                "vendor": "Liquid Intelligent Technologies",
                "expense_type": "equipment",
                "budget_remaining": 35000,
                "amount": 10050.0,
            },
            "expected": "review",
        },
        {
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
            "expected": "approve",
        },
        {
            "domain": "sales",
            "title": "Sales: Order Breaching 15.0% Discount Limit (16.5%)",
            "context": {
                "customer_id": "CUST-ZM-102",
                "customer_segment": "tier_1_enterprise",
                "order_value": 28000,
                "discount_pct": 16.5,
                "current_outstanding_balance": 15000,
                "credit_limit": 60000,
            },
            "expected": "review",
        },
        {
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
            "expected": "approve",
        },
    ]

    print("\n" + "=" * 80)
    print("  UDM-200M JAX BUILT-IN SCENARIO INFERENCE")
    print("=" * 80)

    for i, s in enumerate(scenarios, 1):
        res = engine.decide(context=s["context"], domain=s["domain"])
        status = "PASS" if res["decision"] == s["expected"] else "FLAG"
        print(f"\n[{i}/{len(scenarios)}] {s['title']}")
        print(f"  Result:      {res['decision'].upper()} (Confidence: {res['confidence']*100:.1f}%) [{status}]")
        print(f"  Abstain:     {res['abstain_gate']['abstain']} (p={res['abstain_gate']['abstain_probability']})")
        print(f"  Latency:     {res['latency_ms']:.2f} ms")


def main():
    parser = argparse.ArgumentParser(description="UDM-200M JAX Decision Inference")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "checkpoints", "udm_200m_jax"),
        help="Path to checkpoint directory",
    )
    parser.add_argument("--domain", type=str, choices=["finance", "sales", "inventory"], help="Domain")
    parser.add_argument("--context", type=str, help="Context JSON string")
    parser.add_argument("--file", type=str, help="Path to scenario JSON file")
    args = parser.parse_args()

    hw = detect_hardware()
    print(f"Initializing JAX Engine on {hw['target_name']} ({hw['backend'].upper()})...")

    engine = JaxDecisionEngine(args.checkpoint_dir)

    if args.domain and args.context:
        ctx = json.loads(args.context)
        res = engine.decide(ctx, args.domain)
        print(json.dumps(res, indent=2))
    elif args.file:
        with open(args.file, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                res = engine.decide(item["context"], item["domain"])
                print(f"ID: {item.get('id', 'N/A')} | Decision: {res['decision'].upper()} | Latency: {res['latency_ms']:.2f}ms")
        else:
            res = engine.decide(data["context"], data["domain"])
            print(json.dumps(res, indent=2))
    else:
        run_builtin_demos(engine)


if __name__ == "__main__":
    main()
