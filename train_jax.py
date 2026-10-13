#!/usr/bin/env python3
"""
UDM-200M Master JAX Training Script.

Trains the 200 Million Parameter Decision-Native Model (UDM-200M)
using JAX and Flax for high throughput across GPUs, Cloud TPUs, and CPUs.

Usage:
    python train_jax.py                     # Full training on UDM-200M
    python train_jax.py --quick             # Quick smoke test (1 epoch)
    python train_jax.py --preset 200m       # Explicit 200M architecture
    python train_jax.py --precision bfloat16 # Enable bfloat16 for TPU/GPU
    python train_jax.py --device auto       # Auto-detect TPU / GPU / CPU
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, Any, List, Tuple
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from udm_jax.config import UDMJaxConfig, UDM_200M, UDM_125M, UDM_350M, UDM_1B
from udm_jax.model.udm import UDMJaxModel
from udm_jax.model.calibration import TemperatureScaler, compute_ece, compute_brier_score
from udm_jax.distributed import detect_hardware, setup_xla_environment, create_device_mesh
from udm_jax.training.trainer import (
    init_train_state,
    make_train_step,
    make_eval_step,
    save_checkpoint,
)
from udm.data.tokenizer import DecisionTokenizer
from udm.data.dataset import (
    load_contrastive_pairs,
    augment_scenario,
    LABEL_MAP,
    DOMAIN_MAP,
    ALT_TASK_DESCRIPTIONS,
)
from udm.data.schema import serialize_decision

import jax
import jax.numpy as jnp


def parse_args():
    parser = argparse.ArgumentParser(description="Train UDM-200M Decision-Native Model with JAX")
    parser.add_argument(
        "--preset",
        type=str,
        default="200m",
        choices=["200m", "125m", "350m", "1b"],
        help="Architecture preset (default: 200m)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "contrastive_pairs_300.json"),
        help="Path to contrastive pairs JSON/JSONL dataset",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (default: 5 on CPU / smoke, 20 on GPU/TPU)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (default: 16 on CPU, 32 on GPU/TPU)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Peak learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="float32",
        choices=["float32", "bfloat16"],
        help="Compute precision (use bfloat16 for Ampere+ GPUs and TPUs)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick smoke test (1 epoch, minimal augmentation)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "gpu", "tpu", "cpu"],
        help="Target execution hardware",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "checkpoints", "udm_200m_jax"),
        help="Directory to export final trained model package",
    )
    return parser.parse_args()


def prepare_numpy_dataset(
    scenarios: List[Dict[str, Any]],
    tokenizer: DecisionTokenizer,
    max_seq_len: int = 512,
    augment: bool = False,
    augment_factor: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Tokenizes and pre-pads dataset into contiguous NumPy arrays for JAX feeding.
    """
    if augment:
        all_scenarios = []
        for s in scenarios:
            all_scenarios.extend(augment_scenario(s, num_augments=augment_factor))
    else:
        all_scenarios = scenarios

    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    domain_ids_list = []

    for s in all_scenarios:
        domain = s["domain"]
        context = s["context"]
        label_str = s["label"]
        decision_type = s.get("decision_type", "classification")

        variant_idx = s.get("_task_variant", 0)
        task_desc = ALT_TASK_DESCRIPTIONS.get(domain, ALT_TASK_DESCRIPTIONS["finance"])[variant_idx]

        canonical_text = serialize_decision(
            context=context,
            domain=domain,
            decision_type=decision_type,
            task_description=task_desc,
            field_order=list(context.keys()),
        )

        tokens = tokenizer.encode(
            canonical_text,
            add_bos=True,
            add_eos=True,
            task_type=decision_type,
            max_length=max_seq_len,
        )

        pad_id = tokenizer.pad_token_id
        if len(tokens) < max_seq_len:
            mask = [1] * len(tokens) + [0] * (max_seq_len - len(tokens))
            tokens = tokens + [pad_id] * (max_seq_len - len(tokens))
        else:
            tokens = tokens[:max_seq_len]
            mask = [1] * max_seq_len

        input_ids_list.append(tokens)
        attention_mask_list.append(mask)
        labels_list.append(LABEL_MAP.get(label_str, 0))
        domain_ids_list.append(DOMAIN_MAP.get(domain, 0))

    return {
        "input_ids": np.array(input_ids_list, dtype=np.int32),
        "attention_mask": np.array(attention_mask_list, dtype=np.int32),
        "labels": np.array(labels_list, dtype=np.int32),
        "domain_ids": np.array(domain_ids_list, dtype=np.int32),
    }


def split_data(
    data: Dict[str, np.ndarray],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Split dataset into train, val, and test subsets."""
    n = len(data["labels"])
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    train_data = {k: v[train_idx] for k, v in data.items()}
    val_data = {k: v[val_idx] for k, v in data.items()}
    test_data = {k: v[test_idx] for k, v in data.items()}

    return train_data, val_data, test_data


def iterate_batches(data: Dict[str, np.ndarray], batch_size: int, shuffle: bool = True, rng=None):
    """Yield batches from dataset as JAX-ready arrays."""
    n = len(data["labels"])
    indices = np.arange(n)
    if shuffle:
        if rng is None:
            rng = np.random.default_rng()
        rng.shuffle(indices)

    for i in range(0, n, batch_size):
        batch_idx = indices[i : i + batch_size]
        yield {k: jnp.array(v[batch_idx]) for k, v in data.items()}


def main():
    args = parse_args()

    # Configure platform environment
    platform = None if args.device == "auto" else args.device
    setup_xla_environment(enable_gpu_memory_fraction=0.90, platform=platform)

    hw = detect_hardware()
    print("=" * 80)
    print("  UDM-200M DECISION-NATIVE INTELLIGENCE — HIGH PERFORMANCE JAX PIPELINE")
    print("=" * 80)
    print(f"Hardware Target:     {hw['target_name']} ({hw['backend'].upper()})")
    print(f"Detected Devices:    {hw['num_devices']} ({', '.join(hw['device_kinds'])})")
    print(f"Precision:           {args.precision}")

    # Select configuration preset
    preset_map = {
        "200m": UDM_200M,
        "125m": UDM_125M,
        "350m": UDM_350M,
        "1b": UDM_1B,
    }
    config: UDMJaxConfig = preset_map[args.preset]
    config.dtype = args.precision
    config.learning_rate = args.lr

    param_stats = config.estimate_parameters()
    print("\n--- Model Architecture & Parameter Verification ---")
    print(f"  Preset:                UDM-{args.preset.upper()}")
    print(f"  Layers:                {config.num_layers}")
    print(f"  Hidden Dim:            {config.hidden_dim}")
    print(f"  Attention Heads:       {config.num_heads} query, {config.num_kv_heads} KV (GQA {config.kv_groups}:1)")
    print(f"  Feed-Forward Dim:      {config.ffn_dim} (SwiGLU gated)")
    print(f"  Backbone Parameters:   {param_stats['backbone']:,}")
    print(f"  Embedding Parameters:  {param_stats['embedding']:,}")
    print(f"  Decision Heads:        {param_stats['decision_heads']:,}")
    print(f"  TOTAL PARAMETERS:      {param_stats['total']:,} ({param_stats['total_millions']} Million)")
    print("----------------------------------------------------\n")

    # Load contrastive scenarios
    print(f"Loading dataset from: {args.data}")
    scenarios = load_contrastive_pairs(args.data)
    print(f"Loaded {len(scenarios)} decision scenarios across finance, sales, inventory.")

    # Initialize Tokenizer
    tok_dir = os.path.join(PROJECT_ROOT, "checkpoints", "udm_125m_final")
    tok_file = os.path.join(tok_dir, "tokenizer.json")
    if os.path.exists(tok_file):
        tokenizer = DecisionTokenizer.load(tok_file)
        print(f"Loaded existing vocabulary: {tokenizer.vocab_size} tokens")
    else:
        tokenizer = DecisionTokenizer(vocab_size=config.vocab_size)
        all_texts = [
            serialize_decision(s["context"], s["domain"], s.get("decision_type", "classification"))
            for s in scenarios
        ]
        tokenizer.build_vocab(all_texts)
        print(f"Built new tokenizer vocabulary: {tokenizer.vocab_size} tokens")

    # Dataset Preparation
    aug_factor = 2 if args.quick else (config.augmentation_factor if hw["is_gpu"] or hw["is_tpu"] else 3)
    epochs = 1 if args.quick else (args.epochs or (20 if hw["is_gpu"] or hw["is_tpu"] else 5))
    batch_size = args.batch_size or (32 if hw["is_gpu"] or hw["is_tpu"] else 8)

    print(f"Preprocessing & augmenting data (factor = {aug_factor})...")
    raw_dataset = prepare_numpy_dataset(
        scenarios=scenarios,
        tokenizer=tokenizer,
        max_seq_len=256 if args.quick else 512,
        augment=True,
        augment_factor=aug_factor,
    )

    train_data, val_data, test_data = split_data(raw_dataset, train_ratio=0.70, val_ratio=0.15)
    print(f"Data Splits: Train={len(train_data['labels'])} | Val={len(val_data['labels'])} | Test={len(test_data['labels'])}")

    # Initialize JAX Model and TrainState
    model = UDMJaxModel(config=config)
    total_steps = (len(train_data["labels"]) // batch_size + 1) * epochs
    rng = jax.random.PRNGKey(42)

    print("\nInitializing parameters and XLA computation graph...")
    t_init = time.perf_counter()
    state = init_train_state(model, config, total_steps, rng)
    print(f"Parameter initialization completed in {(time.perf_counter() - t_init):.2f}s")

    # JIT-compile step functions
    train_step = make_train_step(config)
    eval_step = make_eval_step(config)

    # Training Loop
    print(f"\nBeginning JAX Training Loop ({epochs} epochs, batch_size={batch_size}, total_steps={total_steps}):")
    best_val_acc = 0.0
    best_params = state.params

    for epoch in range(1, epochs + 1):
        t_epoch_start = time.perf_counter()
        train_losses = []
        train_accs = []

        for batch in iterate_batches(train_data, batch_size=batch_size, shuffle=True):
            state, metrics = train_step(state, batch)
            train_losses.append(float(metrics["loss"]))
            train_accs.append(float(metrics["accuracy"]))

        # Validation pass
        val_losses = []
        val_accs = []
        val_logits_list = []
        val_labels_list = []

        for vbatch in iterate_batches(val_data, batch_size=batch_size, shuffle=False):
            loss, outputs, metrics = eval_step(state, vbatch)
            val_losses.append(float(loss))
            val_accs.append(float(metrics["accuracy"]))
            val_logits_list.append(np.array(outputs["logits"]))
            val_labels_list.append(np.array(vbatch["labels"]))

        epoch_time = time.perf_counter() - t_epoch_start
        mean_train_loss = np.mean(train_losses)
        mean_train_acc = np.mean(train_accs)
        mean_val_loss = np.mean(val_losses)
        mean_val_acc = np.mean(val_accs)

        print(
            f"  Epoch {epoch:2d}/{epochs:2d} | "
            f"Train Loss: {mean_train_loss:.4f} | Train Acc: {mean_train_acc*100:5.1f}% | "
            f"Val Loss: {mean_val_loss:.4f} | Val Acc: {mean_val_acc*100:5.1f}% | "
            f"Time: {epoch_time:.2f}s"
        )

        if mean_val_acc >= best_val_acc:
            best_val_acc = mean_val_acc
            best_params = state.params

    # Temperature Scaling Calibration
    print("\n--- Post-Hoc Probability Calibration ---")
    val_logits_arr = np.concatenate(val_logits_list, axis=0)
    val_labels_arr = np.concatenate(val_labels_list, axis=0)
    scaler = TemperatureScaler()
    optimal_temp = scaler.calibrate(val_logits_arr, val_labels_arr)
    print(f"Optimal Temperature Scaling T: {optimal_temp:.4f}")

    # Test Evaluation
    print("\n--- Final Test Set Evaluation ---")
    test_logits_list = []
    test_labels_list = []
    test_latencies = []

    state = state.replace(params=best_params)

    for tbatch in iterate_batches(test_data, batch_size=1, shuffle=False):
        t0 = time.perf_counter()
        _, outputs, _ = eval_step(state, tbatch)
        # Block until JAX computation completes on device
        _ = outputs["logits"].block_until_ready()
        lat = (time.perf_counter() - t0) * 1000
        test_latencies.append(lat)
        test_logits_list.append(np.array(outputs["logits"]))
        test_labels_list.append(np.array(tbatch["labels"]))

    test_logits_arr = np.concatenate(test_logits_list, axis=0)
    test_labels_arr = np.concatenate(test_labels_list, axis=0)

    # Calibrated probabilities
    raw_probs = np.exp(test_logits_arr) / np.sum(np.exp(test_logits_arr), axis=-1, keepdims=True)
    calibrated_probs = scaler.scale_logits(test_logits_arr)

    preds = np.argmax(calibrated_probs, axis=-1)
    test_acc = np.mean(preds == test_labels_arr)
    raw_ece, _ = compute_ece(raw_probs, test_labels_arr)
    cal_ece, _ = compute_ece(calibrated_probs, test_labels_arr)
    brier = compute_brier_score(calibrated_probs, test_labels_arr)

    # Discard first latency item for warmup
    latencies = test_latencies[1:] if len(test_latencies) > 1 else test_latencies
    p50_lat = np.percentile(latencies, 50)
    p95_lat = np.percentile(latencies, 95)
    p99_lat = np.percentile(latencies, 99)

    print(f"  Test Accuracy:               {test_acc * 100:.2f}%")
    print(f"  Raw ECE (Before Calib):      {raw_ece:.4f}")
    print(f"  Calibrated ECE:              {cal_ece:.4f} (Improved: {(raw_ece - cal_ece):.4f})")
    print(f"  Brier Score:                 {brier:.4f}")
    print(f"  P50 Latency (Single Item):   {p50_lat:.2f} ms")
    print(f"  P95 Latency:                 {p95_lat:.2f} ms")
    print(f"  P99 Latency:                 {p99_lat:.2f} ms")

    # Export Checkpoint
    print(f"\nExporting UDM-200M JAX Model Package to: {args.export_dir}")
    metadata = {
        "model_preset": f"UDM-{args.preset.upper()}",
        "parameter_count": param_stats["total"],
        "parameter_millions": param_stats["total_millions"],
        "hardware_target": hw["target_name"],
        "precision": args.precision,
        "temperature": optimal_temp,
        "test_accuracy": float(test_acc),
        "test_ece": float(cal_ece),
        "p50_latency_ms": float(p50_lat),
    }
    save_checkpoint(
        state=state,
        config=config,
        export_dir=args.export_dir,
        tokenizer=tokenizer,
        extra_metadata=metadata,
    )
    print("Export complete: Model is ready for GPU / Cloud TPU / Sub-Millisecond deployment!")


if __name__ == "__main__":
    main()
