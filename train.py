#!/usr/bin/env python3
"""
UDM-125M Master Training Script

Usage:
    python train.py                     # Run training with default settings
    python train.py --quick             # Run quick 1-epoch smoke test
    python train.py --epochs 10         # Train for 10 epochs
    python train.py --data data/contrastive_pairs_300.json
"""

import os
import sys
import argparse
import time
import json
import dataclasses
from collections import Counter

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate UDM-125M model")
    parser.add_argument(
        "--data",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "contrastive_pairs_300.json"),
        help="Path to contrastive pairs JSON/JSONL file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs to train (default: 5 on CPU, 50 on GPU)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (default: 8 on CPU, 32 on GPU)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Peak learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick verification run (1 epoch, reduced augmentation)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for training",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "checkpoints"),
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "checkpoints", "udm_125m_final"),
        help="Directory to export final model",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    print("=" * 70)
    print("  UDM-125M: Uniplexity Decision Model — Training & Evaluation")
    print("=" * 70)

    # Resolve device
    if args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  BF16 Support: {torch.cuda.is_bf16_supported()}")
    else:
        print("  Running on CPU (mixed precision disabled)")

    # Data file check
    data_path = args.data
    if not os.path.exists(data_path):
        fallback = os.path.join(PROJECT_ROOT, "data", "contrastive_pairs_300.jsonl")
        if os.path.exists(fallback):
            data_path = fallback
        else:
            raise FileNotFoundError(f"Data file not found at {args.data} or fallback {fallback}")

    print(f"  Data: {data_path}")
    scenarios = load_contrastive_pairs(data_path)
    print(f"  Loaded {len(scenarios)} scenarios from contrastive pairs")

    # Domain & label distributions
    domain_dist = Counter(s["domain"] for s in scenarios)
    label_dist = Counter(s["label"] for s in scenarios)
    print(f"  Domains: {dict(domain_dist)}")
    print(f"  Labels:  {dict(label_dist)}")

    # Configuration
    config = UDM_125M
    config.learning_rate = args.lr

    if args.quick:
        config.max_epochs = 1
        config.batch_size = 4
        config.augmentation_factor = 2
        print("\n  [QUICK MODE ACTIVE] 1 epoch, minimal augmentation")
    else:
        if args.epochs is not None:
            config.max_epochs = args.epochs
        elif device.type == "cpu":
            config.max_epochs = 3
        else:
            config.max_epochs = 50

        if args.batch_size is not None:
            config.batch_size = args.batch_size
        elif device.type == "cpu":
            config.batch_size = 8
        else:
            config.batch_size = 32

    # Tokenizer build
    print("\n  Building tokenizer vocabulary...")
    canonical_texts = []
    for s in scenarios:
        canonical_texts.append(
            serialize_decision(
                context=s["context"],
                domain=s["domain"],
                decision_type="classification",
            )
        )

    tokenizer = DecisionTokenizer(vocab_size=config.vocab_size)
    tokenizer.build_vocab(canonical_texts)
    config.vocab_size = tokenizer.vocab_size

    # Save tokenizer early
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    tokenizer.save(os.path.join(args.checkpoint_dir, "tokenizer.json"))

    # Splits
    train_scenarios, val_scenarios, test_scenarios = create_data_splits(
        scenarios,
        train_ratio=config.train_split,
        val_ratio=config.val_split,
        test_ratio=config.test_split,
        seed=42,
    )

    if args.quick:
        # Keep subset for quick check
        train_scenarios = train_scenarios[:8]
        val_scenarios = val_scenarios[:4]
        test_scenarios = test_scenarios[:4]

    max_seq_len = 512
    train_dataset = DecisionDataset(
        train_scenarios,
        tokenizer,
        max_seq_len=max_seq_len,
        augment=True,
        augment_factor=config.augmentation_factor,
    )
    val_dataset = DecisionDataset(
        val_scenarios,
        tokenizer,
        max_seq_len=max_seq_len,
        augment=False,
    )
    test_dataset = DecisionDataset(
        test_scenarios,
        tokenizer,
        max_seq_len=max_seq_len,
        augment=False,
    )

    print(f"  Train: {len(train_dataset)} samples | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
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

    # Initialize model
    print("\n  Instantiating UDM-125M model...")
    model = UDMModel(config).to(device)
    param_counts = model.count_parameters()
    print(f"  Total parameters:     {param_counts['total']:,d} ({param_counts['total']/1e6:.1f}M)")
    print(f"  Trainable parameters: {param_counts['trainable']:,d}")

    # Loss & Trainer
    loss_fn = CombinedDecisionLoss(config)
    trainer = UDMTrainer(
        model=model,
        loss_fn=loss_fn,
        config=config,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Fit
    print(f"\n  Starting training for {config.max_epochs} epochs...")
    train_history, val_history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.max_epochs,
    )

    # Evaluate on test set
    print(f"\n{'='*70}")
    print("  EVALUATION ON TEST SET")
    print(f"{'='*70}")
    test_metrics = trainer.evaluate(test_loader)
    print(f"  Test Loss:     {test_metrics['loss']:.4f}")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Test ECE:      {test_metrics['ece']:.4f}")
    print(f"  Brier Score:   {test_metrics['brier_score']:.4f}")

    # Latency benchmark
    sample_ids = torch.randint(0, config.vocab_size, (1, 256), device=device)
    sample_mask = torch.ones(1, 256, dtype=torch.long, device=device)
    latencies = measure_inference_latency(
        model,
        (sample_ids, sample_mask),
        num_warmup=5 if args.quick else 10,
        num_runs=10 if args.quick else 50,
        device=device,
    )
    print(f"\n  Inference Latency: mean={latencies['mean_ms']:.2f}ms, p95={latencies['p95_ms']:.2f}ms, throughput={latencies['throughput_per_sec']:.0f} dec/s")

    # Export model & metadata
    os.makedirs(args.export_dir, exist_ok=True)
    model_path = os.path.join(args.export_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\n  Exported weights to: {model_path} ({os.path.getsize(model_path)/1e6:.1f} MB)")

    config_path = os.path.join(args.export_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(dataclasses.asdict(config), f, indent=2)
    print(f"  Exported config to:  {config_path}")

    tokenizer.save(os.path.join(args.export_dir, "tokenizer.json"))
    print(f"  Exported tokenizer to: {os.path.join(args.export_dir, 'tokenizer.json')}")

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  FINISHED in {total_time:.1f}s ({total_time/60:.2f} minutes)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
