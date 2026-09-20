"""
PyTorch Dataset for UDM training.

Pipeline:
  1. Load contrastive pairs JSONL
  2. Extract individual scenarios with context + label
  3. Serialize to canonical <DECISION> format
  4. Tokenize to integer sequences
  5. Apply data augmentation (field reorder, value jitter, noise)
  6. Pad/truncate to max_seq_len
  7. Return (input_ids, attention_mask, label, domain_id)

Data augmentation expands 600 base examples → ~6,000 training examples:
  - Context field reordering (6+ permutations per example)
  - Numeric value jittering (±0.01 on non-boundary amounts)
  - Task description paraphrasing (3 variants per domain)
"""

import json
import random
import copy
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .schema import serialize_decision, DOMAIN_OPTIONS, DOMAIN_CONTEXT_FIELDS
from .tokenizer import DecisionTokenizer


# Label encoding
LABEL_MAP = {"approve": 0, "review": 1, "reject": 2, "abstain": 3}
DOMAIN_MAP = {"finance": 0, "sales": 1, "inventory": 2}

# Alternative task descriptions for augmentation
ALT_TASK_DESCRIPTIONS = {
    "finance": [
        "Determine whether this expense requires approval or review based on corporate policy thresholds.",
        "Route this expense claim according to company financial controls.",
        "Evaluate this transaction against corporate expense authorization rules.",
    ],
    "sales": [
        "Determine whether this sales order should be approved or routed for review based on discount and credit policies.",
        "Evaluate this commercial order against discount authorization and credit exposure limits.",
        "Route this sales transaction based on commercial terms compliance.",
    ],
    "inventory": [
        "Determine whether this inventory movement should be approved or requires supervisory review.",
        "Evaluate this stock movement against warehouse safety thresholds and audit policies.",
        "Route this inventory transaction based on adjustment limits and traceability requirements.",
    ],
}


def load_contrastive_pairs(file_path: str) -> List[Dict[str, Any]]:
    """Load contrastive pairs from JSON or JSONL and extract individual scenarios."""
    scenarios = []
    path = Path(file_path)
    if path.suffix == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            pairs = json.load(f)
            if isinstance(pairs, dict) and "contrastive_pairs" in pairs:
                pairs = pairs["contrastive_pairs"]
    else:
        pairs = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    pairs.append(json.loads(line))

    for pair in pairs:
        domain = pair["domain"]
        for scenario in pair["contrastive_pair"]:
            scenarios.append({
                "id": scenario["id"],
                "domain": domain,
                "decision_type": scenario.get("decision_type", "classification"),
                "context": scenario["context"],
                "label": scenario["label"],
                "boundary_delta": scenario.get("boundary_delta", ""),
            })
    return scenarios


def augment_scenario(
    scenario: Dict[str, Any],
    num_augments: int = 10,
) -> List[Dict[str, Any]]:
    """
    Generate augmented variants of a scenario.

    Augmentation strategies:
      1. Field reordering (changes token sequence, same semantics)
      2. Numeric jittering (±tiny amount on non-critical values)
      3. Task description variation
    """
    augmented = [scenario]  # Always include the original
    domain = scenario["domain"]
    context = scenario["context"]

    for i in range(num_augments - 1):
        new_scenario = copy.deepcopy(scenario)
        new_scenario["id"] = f"{scenario['id']}_aug{i+1}"

        # Strategy 1: Field reordering
        # Shuffle the keys in the context dict
        keys = list(new_scenario["context"].keys())
        random.shuffle(keys)
        new_scenario["context"] = {k: new_scenario["context"][k] for k in keys}

        # Strategy 2: Numeric jittering on non-boundary fields
        # Only jitter budget_remaining, credit_limit, current_stock, etc.
        # NEVER jitter the boundary variable (amount at threshold, discount at threshold)
        safe_jitter_fields = {
            "finance": ["budget_remaining"],
            "sales": ["credit_limit"],
            "inventory": ["current_stock", "safety_stock_threshold"],
        }
        jitter_fields = safe_jitter_fields.get(domain, [])
        for field in jitter_fields:
            if field in new_scenario["context"]:
                val = new_scenario["context"][field]
                if isinstance(val, (int, float)):
                    # Jitter by ±1-5%
                    jitter = random.uniform(-0.05, 0.05) * abs(val)
                    new_scenario["context"][field] = round(val + jitter, 2)

        # Strategy 3: Task description variation (applied during serialization)
        new_scenario["_task_variant"] = random.randint(0, 2)

        augmented.append(new_scenario)

    return augmented


class DecisionDataset(Dataset):
    """
    PyTorch Dataset for UDM training.

    Each item returns:
      - input_ids:      [max_seq_len] padded token IDs
      - attention_mask:  [max_seq_len] boolean mask (1=valid, 0=pad)
      - label:           int class index
      - domain_id:       int domain index
    """

    def __init__(
        self,
        scenarios: List[Dict[str, Any]],
        tokenizer: DecisionTokenizer,
        max_seq_len: int = 512,
        augment: bool = False,
        augment_factor: int = 10,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # Optionally augment
        if augment:
            all_scenarios = []
            for s in scenarios:
                all_scenarios.extend(augment_scenario(s, num_augments=augment_factor))
            self.scenarios = all_scenarios
        else:
            self.scenarios = scenarios

    def __len__(self) -> int:
        return len(self.scenarios)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        scenario = self.scenarios[idx]
        domain = scenario["domain"]
        context = scenario["context"]
        label_str = scenario["label"]
        decision_type = scenario.get("decision_type", "classification")

        # Get task description variant
        variant_idx = scenario.get("_task_variant", 0)
        task_desc = ALT_TASK_DESCRIPTIONS.get(domain, ALT_TASK_DESCRIPTIONS["finance"])[variant_idx]

        # Serialize to canonical format
        # Use the shuffled field order from context dict keys
        field_order = list(context.keys())
        canonical_text = serialize_decision(
            context=context,
            domain=domain,
            decision_type=decision_type,
            task_description=task_desc,
            field_order=field_order,
        )

        # Tokenize
        token_ids = self.tokenizer.encode(
            canonical_text,
            add_bos=True,
            add_eos=True,
            task_type=decision_type,
            max_length=self.max_seq_len,
        )

        # Pad to max_seq_len
        attention_mask = [1] * len(token_ids) + [0] * (self.max_seq_len - len(token_ids))
        token_ids = token_ids + [self.tokenizer.pad_token_id] * (self.max_seq_len - len(token_ids))

        # Truncate if still over
        token_ids = token_ids[:self.max_seq_len]
        attention_mask = attention_mask[:self.max_seq_len]

        # Encode label
        label = LABEL_MAP.get(label_str, 0)
        domain_id = DOMAIN_MAP.get(domain, 0)

        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            label,
            domain_id,
        )


def create_data_splits(
    scenarios: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split scenarios into train/val/test with stratification by domain x label.

    Returns:
        (train_scenarios, val_scenarios, test_scenarios)
    """
    random.seed(seed)

    # Group by domain x label for stratified splitting
    groups: Dict[str, List[Dict]] = {}
    for s in scenarios:
        key = f"{s['domain']}_{s['label']}"
        if key not in groups:
            groups[key] = []
        groups[key].append(s)

    train, val, test = [], [], []

    for key, group in groups.items():
        random.shuffle(group)
        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:])

    # Final shuffle
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test
