"""
UDM Training Loop.

Implements:
  - AdamW optimizer with cosine annealing LR schedule + linear warmup
  - BF16 mixed precision via torch.amp.autocast (falls back to FP32 if BF16 unavailable)
  - Gradient clipping (max_norm=1.0)
  - Gradient accumulation for effective batch size scaling
  - Epoch-level metrics logging (loss, accuracy, abstention rate)
  - Model checkpointing
"""

import os
import time
import math
from typing import Dict, Optional, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler


class UDMTrainer:
    """
    Trainer for the Uniplexity Decision Model.

    Handles mixed precision, gradient accumulation, LR scheduling,
    and metrics logging for multi-task decision model training.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        config,
        device: torch.device,
        checkpoint_dir: str = "checkpoints",
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Optimizer: AdamW with weight decay
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        # Detect precision support
        self.use_bf16 = (
            device.type == "cuda"
            and torch.cuda.is_bf16_supported()
        )
        self.use_fp16 = device.type == "cuda" and not self.use_bf16
        self.amp_dtype = torch.bfloat16 if self.use_bf16 else torch.float16
        self.scaler = GradScaler(enabled=self.use_fp16)  # Only needed for fp16

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.train_history: List[Dict] = []
        self.val_history: List[Dict] = []

    def _get_lr(self, step: int, total_steps: int) -> float:
        """Cosine annealing with linear warmup."""
        warmup = self.config.warmup_steps
        if step < warmup:
            return self.config.learning_rate * step / max(warmup, 1)
        # Cosine decay
        progress = (step - warmup) / max(total_steps - warmup, 1)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return self.config.min_learning_rate + (
            self.config.learning_rate - self.config.min_learning_rate
        ) * cosine_decay

    def _update_lr(self, step: int, total_steps: int):
        """Update learning rate for current step."""
        lr = self._get_lr(step, total_steps)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def train_epoch(
        self,
        dataloader: DataLoader,
        total_steps: int,
    ) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        epoch_abstain = 0
        num_batches = 0

        for batch_idx, (input_ids, attention_mask, labels, domain_ids) in enumerate(dataloader):
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            labels = labels.to(self.device)

            # Update learning rate
            lr = self._update_lr(self.global_step, total_steps)

            # Forward pass with mixed precision
            use_amp = self.device.type == "cuda"
            with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=use_amp):
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decision_type="classification",
                    num_options=2,
                )

                # Compute loss
                # For now, all examples are "should not abstain" (training data is clean)
                should_abstain = torch.zeros(labels.shape[0], device=self.device)
                loss_dict = self.loss_fn(
                    logits=output.logits,
                    targets=labels,
                    decision_type="classification",
                    abstain_prob=output.abstain_prob,
                    should_abstain=should_abstain,
                )
                loss = loss_dict["total"]

                # Scale for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps

            # Backward pass
            if self.use_fp16:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient step (with accumulation)
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.use_fp16:
                    self.scaler.unscale_(self.optimizer)

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip,
                )

                if self.use_fp16:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.global_step += 1

            # Metrics
            epoch_loss += loss.item() * self.config.gradient_accumulation_steps
            predictions = output.predicted_label
            epoch_correct += (predictions == labels).sum().item()
            epoch_total += labels.shape[0]
            epoch_abstain += output.should_abstain.sum().item()
            num_batches += 1

        metrics = {
            "loss": epoch_loss / max(num_batches, 1),
            "accuracy": epoch_correct / max(epoch_total, 1),
            "abstention_rate": epoch_abstain / max(epoch_total, 1),
            "lr": lr,
            "global_step": self.global_step,
        }
        self.train_history.append(metrics)
        return metrics

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Run validation/test evaluation."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        total_abstain = 0
        all_probs = []
        all_labels = []
        num_batches = 0

        for input_ids, attention_mask, labels, domain_ids in dataloader:
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            labels = labels.to(self.device)

            use_amp = self.device.type == "cuda"
            with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=use_amp):
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decision_type="classification",
                    num_options=2,
                )

                should_abstain = torch.zeros(labels.shape[0], device=self.device)
                loss_dict = self.loss_fn(
                    logits=output.logits,
                    targets=labels,
                    decision_type="classification",
                    abstain_prob=output.abstain_prob,
                    should_abstain=should_abstain,
                )

            total_loss += loss_dict["total"].item()
            total_correct += (output.predicted_label == labels).sum().item()
            total_samples += labels.shape[0]
            total_abstain += output.should_abstain.sum().item()
            num_batches += 1

            all_probs.append(output.probabilities.float().cpu())
            all_labels.append(labels.cpu())

        all_probs = torch.cat(all_probs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Compute calibration metrics
        from udm.model.calibration import compute_ece, compute_brier_score
        ece, _ = compute_ece(all_probs, all_labels)
        brier = compute_brier_score(all_probs, all_labels)

        metrics = {
            "loss": total_loss / max(num_batches, 1),
            "accuracy": total_correct / max(total_samples, 1),
            "abstention_rate": total_abstain / max(total_samples, 1),
            "ece": ece,
            "brier_score": brier,
            "num_samples": total_samples,
        }
        self.val_history.append(metrics)
        return metrics

    validate = evaluate

    def save_checkpoint(self, path: Optional[str] = None, is_best: bool = False):
        """Save model checkpoint."""
        if path is None:
            path = os.path.join(
                self.checkpoint_dir,
                f"udm_epoch{self.current_epoch}.pt",
            )

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "train_history": self.train_history,
            "val_history": self.val_history,
        }
        torch.save(checkpoint, path)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "udm_best.pt")
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.train_history = checkpoint.get("train_history", [])
        self.val_history = checkpoint.get("val_history", [])

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: Optional[int] = None,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Full training loop.

        Returns:
            (train_history, val_history)
        """
        if num_epochs is None:
            num_epochs = self.config.max_epochs

        total_steps = num_epochs * len(train_loader)
        print(f"\n{'='*70}")
        print(f"UDM TRAINING")
        print(f"{'='*70}")
        print(f"  Device: {self.device}")
        print(f"  Precision: {'BF16' if self.use_bf16 else 'FP16' if self.use_fp16 else 'FP32'}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch size: {train_loader.batch_size}")
        print(f"  Train batches/epoch: {len(train_loader)}")
        print(f"  Total steps: {total_steps}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Warmup steps: {self.config.warmup_steps}")
        print(f"{'='*70}\n")

        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            t0 = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader, total_steps)
            train_time = time.time() - t0

            # Validate
            t1 = time.time()
            val_metrics = self.evaluate(val_loader)
            val_time = time.time() - t1

            # Check if best
            is_best = val_metrics["loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["loss"]

            # Checkpoint every 10 epochs + best
            if epoch % 10 == 0 or is_best or epoch == num_epochs - 1:
                self.save_checkpoint(is_best=is_best)

            # Log
            best_marker = " *BEST*" if is_best else ""
            print(
                f"Epoch {self.current_epoch:3d}/{num_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.3f} | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.3f} "
                f"ECE: {val_metrics['ece']:.4f} | "
                f"LR: {train_metrics['lr']:.2e} | "
                f"Time: {train_time:.1f}s{best_marker}"
            )

        print(f"\n{'='*70}")
        print(f"Training complete. Best val loss: {self.best_val_loss:.4f}")
        print(f"{'='*70}")

        return self.train_history, self.val_history
