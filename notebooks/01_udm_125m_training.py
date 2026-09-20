#!/usr/bin/env python3
"""
=============================================================================
 UDM-125M: Uniplexity Decision Model — Architecture Validation Training
=============================================================================

 Research Hypothesis:
   Can a ~125M parameter decision-native model turn arbitrary context +
   a typed decision schema into a fast, calibrated decision more cheaply
   than a general-purpose LLM?

 This script runs the full Stage 1 (Milestone 1) pipeline:
   S1: Load 600 contrastive boundary decision scenarios (3 domains)
   S2: Build tokenizer vocabulary from canonical decision format
   S3: Augment to ~6,000 examples, split train/val/test
   S4: Instantiate UDM-125M (12L, 768H, GQA 12/4, SwiGLU, RMSNorm, RoPE)
   S5: Train with AdamW + cosine LR + BF16 mixed precision
   S6: Evaluate: accuracy, F1, ECE, Brier, abstention analysis
   S7: Calibrate with temperature scaling
   S8: Run inference demo on raw decision context

 Architecture:
                    Context + Decision Schema
                              |
                    Canonical Serializer
                              |
                    Tokenizer + Embedding
                              |
                    12-Layer Transformer (GQA + RoPE + SwiGLU + RMSNorm)
                              |
                         Mean Pool
                              |
               +--------------+--------------+
               |              |              |
         Classification  Regression    Ranking
              Head          Head         Head
               |              |              |
               +--------------+--------------+
                              |
                     Abstention Gate
                              |
                    Typed Decision Object
=============================================================================
"""

import os
import sys
import json
import time
import random
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from udm.configs.config import UDM_125M, UDMConfig
from udm.model.udm import UDMModel
from udm.data.schema import serialize_decision
from udm.data.tokenizer import DecisionTokenizer
from udm.data.dataset import (
    DecisionDataset,
    load_contrastive_pairs,
    create_data_splits,
    LABEL_MAP,
    DOMAIN_MAP,
)
from udm.losses.losses import CombinedDecisionLoss
from udm.training.trainer import UDMTrainer
from udm.evaluation.metrics import (
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_abstention_metrics,
    compute_risk_coverage_curve,
    measure_inference_latency,
    format_metrics_table,
)
from udm.model.calibration import TemperatureScaler, compute_ece, compute_brier_score

warnings.filterwarnings("ignore", category=UserWarning)


def set_seed(seed: int = 42):
    """Reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# S1: ENVIRONMENT & CONFIGURATION
# ============================================================================

def stage_1_setup():
    """Set up environment, device, and model configuration."""
    set_seed(42)

    print("=" * 70)
    print("  UDM-125M: Uniplexity Decision Model - Stage 1 Training")
    print("=" * 70)

    # Device detection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"  Device: {gpu_name} ({vram_gb:.1f} GB VRAM)")
        print(f"  CUDA: {torch.version.cuda}")
        bf16_ok = torch.cuda.is_bf16_supported()
        print(f"  BF16 support: {bf16_ok}")
    else:
        device = torch.device("cpu")
        print("  Device: CPU (no GPU detected)")
        print("  WARNING: Training on CPU will be very slow!")

    print(f"  PyTorch: {torch.__version__}")
    print(f"  Python: {sys.version.split()[0]}")

    # Model config
    config = UDM_125M
    # Override for smaller validation run if on CPU
    if device.type == "cpu":
        config.max_epochs = 5
        config.batch_size = 8
    else:
        config.max_epochs = 50
        config.batch_size = 32

    # Estimate parameters
    param_info = config.estimate_parameters()
    print(f"\n  Model: UDM-125M")
    print(f"  Estimated parameters: {param_info['total_millions']:.1f}M")
    print(f"    Embedding:    {param_info['embedding'] / 1e6:.1f}M")
    print(f"    Backbone:     {param_info['backbone'] / 1e6:.1f}M")
    print(f"    Decision heads: {param_info['decision_heads'] / 1e6:.2f}M")

    return device, config


# ============================================================================
# S2: DATA PIPELINE
# ============================================================================

def stage_2_data_pipeline(config: UDMConfig):
    """Load data, build tokenizer, create datasets."""
    print(f"\n{'='*70}")
    print("  S2: DATA PIPELINE")
    print(f"{'='*70}")

    # Load contrastive pairs (from contrastive_pairs_300.json or .jsonl)
    data_path = os.path.join(PROJECT_ROOT, "data", "contrastive_pairs_300.json")
    if not os.path.exists(data_path):
        data_path = os.path.join(PROJECT_ROOT, "data", "contrastive_pairs_300.jsonl")
    print(f"\n  Loading: {data_path}")
    scenarios = load_contrastive_pairs(data_path)
    print(f"  Loaded {len(scenarios)} scenarios from 300 contrastive pairs")

    # Domain distribution
    from collections import Counter
    domain_dist = Counter(s["domain"] for s in scenarios)
    label_dist = Counter(s["label"] for s in scenarios)
    print(f"  Domains: {dict(domain_dist)}")
    print(f"  Labels:  {dict(label_dist)}")

    # Build canonical texts for tokenizer training
    print(f"\n  Building tokenizer vocabulary...")
    canonical_texts = []
    for s in scenarios:
        text = serialize_decision(
            context=s["context"],
            domain=s["domain"],
            decision_type="classification",
        )
        canonical_texts.append(text)

    # Print sample canonical text
    print(f"\n  Sample canonical format:")
    print(f"  {'-'*40}")
    for line in canonical_texts[0].split("\n"):
        print(f"    {line}")
    print(f"  {'-'*40}")

    # Build tokenizer
    tokenizer = DecisionTokenizer(vocab_size=config.vocab_size)
    tokenizer.build_vocab(canonical_texts)

    # Update config vocab size to match actual tokenizer
    config.vocab_size = tokenizer.vocab_size

    # Save tokenizer
    tok_path = os.path.join(PROJECT_ROOT, "checkpoints", "tokenizer.json")
    os.makedirs(os.path.dirname(tok_path), exist_ok=True)
    tokenizer.save(tok_path)

    # Test encode/decode
    sample_ids = tokenizer.encode(canonical_texts[0], max_length=512)
    print(f"\n  Tokenization test:")
    print(f"    Input length: {len(canonical_texts[0])} chars")
    print(f"    Token IDs:    {len(sample_ids)} tokens")
    print(f"    First 20 IDs: {sample_ids[:20]}")

    # Split data
    print(f"\n  Splitting data (70/15/15)...")
    train_scenarios, val_scenarios, test_scenarios = create_data_splits(
        scenarios,
        train_ratio=config.train_split,
        val_ratio=config.val_split,
        test_ratio=config.test_split,
    )
    print(f"  Train: {len(train_scenarios)}, Val: {len(val_scenarios)}, Test: {len(test_scenarios)}")

    # Create datasets with augmentation
    max_seq_len = 512  # Canonical format texts are ~200-400 tokens
    print(f"\n  Creating augmented datasets (factor={config.augmentation_factor})...")

    train_dataset = DecisionDataset(
        train_scenarios, tokenizer,
        max_seq_len=max_seq_len,
        augment=True,
        augment_factor=config.augmentation_factor,
    )
    val_dataset = DecisionDataset(
        val_scenarios, tokenizer,
        max_seq_len=max_seq_len,
        augment=False,
    )
    test_dataset = DecisionDataset(
        test_scenarios, tokenizer,
        max_seq_len=max_seq_len,
        augment=False,
    )

    print(f"  Train (augmented): {len(train_dataset)} samples")
    print(f"  Val:               {len(val_dataset)} samples")
    print(f"  Test:              {len(test_dataset)} samples")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Verify a batch
    sample_batch = next(iter(train_loader))
    print(f"\n  Batch shapes:")
    print(f"    input_ids:      {sample_batch[0].shape}")
    print(f"    attention_mask:  {sample_batch[1].shape}")
    print(f"    labels:          {sample_batch[2].shape}")
    print(f"    domain_ids:      {sample_batch[3].shape}")

    return tokenizer, train_loader, val_loader, test_loader, test_scenarios


# ============================================================================
# S3: MODEL INSTANTIATION
# ============================================================================

def stage_3_model(config: UDMConfig, device: torch.device):
    """Instantiate UDM-125M and verify architecture."""
    print(f"\n{'='*70}")
    print("  S3: MODEL INSTANTIATION")
    print(f"{'='*70}")

    model = UDMModel(config)

    # Count parameters
    param_counts = model.count_parameters()
    print(f"\n  Parameter Counts:")
    for k, v in param_counts.items():
        if k in ("total", "trainable"):
            print(f"    {k:25s} {v:>12,d}  ({v/1e6:.1f}M)")
        else:
            print(f"    {k:25s} {v:>12,d}")

    # Verify forward pass with random input
    print(f"\n  Architecture validation (random input forward pass)...")
    model.to(device)
    dummy_ids = torch.randint(0, config.vocab_size, (2, 128), device=device)
    dummy_mask = torch.ones(2, 128, dtype=torch.long, device=device)

    with torch.no_grad():
        output = model(
            input_ids=dummy_ids,
            attention_mask=dummy_mask,
            decision_type="classification",
            num_options=2,
        )

    print(f"    Logits shape:       {output.logits.shape}")
    print(f"    Probabilities shape: {output.probabilities.shape}")
    print(f"    Predictions:         {output.predicted_label}")
    print(f"    Confidence:          {output.confidence}")
    print(f"    Abstain prob:        {output.abstain_prob}")
    print(f"    Should abstain:      {output.should_abstain}")
    print(f"  [OK] Forward pass successful!")

    # Memory usage
    if device.type == "cuda":
        mem_allocated = torch.cuda.memory_allocated(0) / 1e6
        mem_reserved = torch.cuda.memory_reserved(0) / 1e6
        print(f"\n  GPU Memory:")
        print(f"    Allocated: {mem_allocated:.1f} MB")
        print(f"    Reserved:  {mem_reserved:.1f} MB")

    return model


# ============================================================================
# S4: TRAINING
# ============================================================================

def stage_4_training(model, config, device, train_loader, val_loader):
    """Train UDM-125M."""
    print(f"\n{'='*70}")
    print("  S4: TRAINING")
    print(f"{'='*70}")

    loss_fn = CombinedDecisionLoss(config)
    trainer = UDMTrainer(
        model=model,
        loss_fn=loss_fn,
        config=config,
        device=device,
        checkpoint_dir=os.path.join(PROJECT_ROOT, "checkpoints"),
    )

    train_history, val_history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.max_epochs,
    )

    return trainer, train_history, val_history


# ============================================================================
# S5: EVALUATION
# ============================================================================

def stage_5_evaluation(model, config, device, test_loader):
    """Full evaluation on test set."""
    print(f"\n{'='*70}")
    print("  S5: EVALUATION")
    print(f"{'='*70}")

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_confidences = []
    all_abstain = []

    with torch.no_grad():
        for input_ids, attention_mask, labels, domain_ids in test_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decision_type="classification",
                num_options=2,
            )

            all_preds.append(output.predicted_label.cpu())
            all_labels.append(labels)
            all_probs.append(output.probabilities.float().cpu())
            all_confidences.append(output.confidence.float().cpu())
            all_abstain.append(output.should_abstain.cpu())

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    probs = torch.cat(all_probs)
    confidences = torch.cat(all_confidences)
    abstain_flags = torch.cat(all_abstain)

    # Classification metrics
    cls_metrics = compute_classification_metrics(preds, labels, num_classes=2)
    print(format_metrics_table(cls_metrics, "Classification Metrics"))

    # Calibration metrics
    ece, bin_data = compute_ece(probs, labels)
    brier = compute_brier_score(probs, labels)
    cal_metrics = {"ECE": ece, "Brier Score": brier}
    print(format_metrics_table(cal_metrics, "Calibration Metrics"))

    # Abstention metrics
    abs_metrics = compute_abstention_metrics(preds, labels, confidences, abstain_flags)
    print(format_metrics_table(abs_metrics, "Abstention Metrics"))

    # Confusion matrix
    cm = compute_confusion_matrix(preds, labels, num_classes=2)
    print(f"\n  Confusion Matrix (rows=true, cols=pred):")
    print(f"               Pred: approve  review")
    print(f"  True approve: {cm[0,0]:6d}    {cm[0,1]:6d}")
    print(f"  True review:  {cm[1,0]:6d}    {cm[1,1]:6d}")

    # Risk-coverage curve
    rc_curve = compute_risk_coverage_curve(preds, labels, confidences)
    print(f"\n  Risk-Coverage Curve (selected points):")
    for point in rc_curve[::4]:
        print(
            f"    Threshold: {point['threshold']:.2f} | "
            f"Coverage: {point['coverage']:.2f} | "
            f"Selective Acc: {point['selective_accuracy']:.3f} | "
            f"Risk: {point['risk']:.3f}"
        )

    return cls_metrics, cal_metrics, abs_metrics


# ============================================================================
# S6: CALIBRATION
# ============================================================================

def stage_6_calibration(model, config, device, val_loader):
    """Post-hoc temperature scaling calibration."""
    print(f"\n{'='*70}")
    print("  S6: POST-HOC CALIBRATION (Temperature Scaling)")
    print(f"{'='*70}")

    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, labels, _ in val_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decision_type="classification",
                num_options=2,
            )
            all_logits.append(output.logits.float().cpu())
            all_labels.append(labels)

    # Calibrate temperature
    scaler = model.temperature_scaler
    scaler = scaler.cpu()  # Calibration runs on CPU
    optimal_temp = scaler.calibrate(all_logits, all_labels)
    print(f"\n  Optimal temperature: {optimal_temp:.4f}")
    print(f"  (T > 1 = softer/less confident, T < 1 = sharper/more confident)")

    # Compute calibrated ECE
    all_logits_cat = torch.cat(all_logits)
    all_labels_cat = torch.cat(all_labels)
    calibrated_probs = scaler(all_logits_cat)
    ece_before, _ = compute_ece(torch.softmax(all_logits_cat, dim=-1), all_labels_cat)
    ece_after, _ = compute_ece(calibrated_probs, all_labels_cat)
    print(f"\n  ECE before calibration: {ece_before:.4f}")
    print(f"  ECE after calibration:  {ece_after:.4f}")
    print(f"  Improvement:            {(ece_before - ece_after) / max(ece_before, 1e-8) * 100:.1f}%")


# ============================================================================
# S7: INFERENCE DEMO
# ============================================================================

def stage_7_inference_demo(model, tokenizer, config, device):
    """Demonstrate inference on raw decision contexts."""
    print(f"\n{'='*70}")
    print("  S7: INFERENCE DEMO")
    print(f"{'='*70}")

    # Sample decision contexts
    demo_contexts = [
        {
            "domain": "finance",
            "context": {
                "department": "engineering",
                "vendor": "Kafue Hardware Solutions",
                "expense_type": "equipment",
                "budget_remaining": 55000,
                "amount": 9850.0,
            },
            "expected": "approve (equipment under 10K threshold)",
        },
        {
            "domain": "finance",
            "context": {
                "department": "engineering",
                "vendor": "Kafue Hardware Solutions",
                "expense_type": "equipment",
                "budget_remaining": 55000,
                "amount": 10250.0,
            },
            "expected": "review (equipment over 10K threshold)",
        },
        {
            "domain": "sales",
            "context": {
                "customer_id": "CUST-ZAM-DEMO-1",
                "customer_tier": "standard_retail",
                "order_value": 25000.0,
                "discount_percent": 14.5,
                "credit_limit": 80000.0,
                "outstanding_balance": 20000.0,
            },
            "expected": "approve (discount within 15% limit, credit OK)",
        },
        {
            "domain": "inventory",
            "context": {
                "item_code": "SKU-DEMO-001",
                "movement_type": "write_off",
                "quantity": 30,
                "unit_cost": 200.0,
                "source_warehouse": "Lusaka Central Distribution",
                "safety_stock_threshold": 50,
                "current_stock": 400,
            },
            "expected": "review (write-off value 6,000 > 5,000 threshold)",
        },
    ]

    LABEL_NAMES = {0: "approve", 1: "review"}
    model.eval()

    for i, demo in enumerate(demo_contexts, 1):
        # Serialize
        canonical = serialize_decision(
            context=demo["context"],
            domain=demo["domain"],
            decision_type="classification",
        )

        # Tokenize
        token_ids = tokenizer.encode(canonical, max_length=512)
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)

        # Inference
        t0 = time.perf_counter()
        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decision_type="classification",
                num_options=2,
            )
        latency_ms = (time.perf_counter() - t0) * 1000

        pred_label = LABEL_NAMES.get(output.predicted_label.item(), "unknown")
        confidence = output.confidence.item()
        abstain = output.should_abstain.item()

        print(f"\n  --- Demo {i}: {demo['domain'].upper()} ---")
        print(f"  Context: {json.dumps(demo['context'], indent=4)}")
        print(f"  Expected:   {demo['expected']}")
        print(f"  Predicted:  {pred_label} (confidence: {confidence:.3f})")
        print(f"  Probabilities: approve={output.probabilities[0][0]:.3f}, review={output.probabilities[0][1]:.3f}")
        print(f"  Abstain:    {abstain} (prob: {output.abstain_prob.item():.3f})")
        print(f"  Latency:    {latency_ms:.2f} ms")

    # Latency benchmark
    print(f"\n  --- Latency Benchmark (batch=1) ---")
    sample_ids = torch.randint(0, config.vocab_size, (1, 256), device=device)
    sample_mask = torch.ones(1, 256, dtype=torch.long, device=device)
    latency_stats = measure_inference_latency(
        model, (sample_ids, sample_mask),
        num_warmup=20, num_runs=100, device=device,
    )
    print(f"  p50: {latency_stats['p50_ms']:.2f} ms")
    print(f"  p95: {latency_stats['p95_ms']:.2f} ms")
    print(f"  p99: {latency_stats['p99_ms']:.2f} ms")
    print(f"  Throughput: {latency_stats['throughput_per_sec']:.0f} decisions/sec")


# ============================================================================
# S8: EXPORT
# ============================================================================

def stage_8_export(model, config, tokenizer):
    """Save final model and config."""
    print(f"\n{'='*70}")
    print("  S8: EXPORT")
    print(f"{'='*70}")

    export_dir = os.path.join(PROJECT_ROOT, "checkpoints", "udm_125m_final")
    os.makedirs(export_dir, exist_ok=True)

    # Save model weights
    model_path = os.path.join(export_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"  Model weights: {model_path} ({os.path.getsize(model_path) / 1e6:.1f} MB)")

    # Save config
    config_path = os.path.join(export_dir, "config.json")
    import dataclasses
    config_dict = dataclasses.asdict(config)
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"  Config:        {config_path}")

    # Save tokenizer
    tok_path = os.path.join(export_dir, "tokenizer.json")
    tokenizer.save(tok_path)
    print(f"  Tokenizer:     {tok_path}")

    print(f"\n  Model exported to: {export_dir}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    t_start = time.time()

    # S1: Setup
    device, config = stage_1_setup()

    # S2: Data pipeline
    tokenizer, train_loader, val_loader, test_loader, test_scenarios = stage_2_data_pipeline(config)

    # S3: Model
    model = stage_3_model(config, device)

    # S4: Training
    trainer, train_history, val_history = stage_4_training(
        model, config, device, train_loader, val_loader,
    )

    # S5: Evaluation
    cls_metrics, cal_metrics, abs_metrics = stage_5_evaluation(
        model, config, device, test_loader,
    )

    # S6: Calibration
    stage_6_calibration(model, config, device, val_loader)

    # S7: Inference demo
    stage_7_inference_demo(model, tokenizer, config, device)

    # S8: Export
    stage_8_export(model, config, tokenizer)

    total_time = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  COMPLETE: Total pipeline time: {total_time/60:.1f} minutes")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
