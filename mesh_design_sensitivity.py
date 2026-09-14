"""Training-only matched-location differences for the 1704 design clusters."""

from collections import defaultdict
import math

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
import wandb

from utils.utils import Trainer, collate_fn


GEOMETRY_CLUSTERS = {"A": (0, 1, 2, 3), "B": (4, 5), "C": (6, 7, 8, 9), "D": (10, 11)}
DESIGN_CLUSTER = {design: cluster for cluster, designs in GEOMETRY_CLUSTERS.items() for design in designs}


class MatchedDesignDataset(Dataset):
    """Annotate the already-split training dataset; never look up held-out runs."""

    def __init__(self, dataset, samples_per_design):
        self.dataset = dataset
        self.locations = [(int(run) - 1) % samples_per_design for run in dataset.run_numbers]
        self.designs = [(int(run) - 1) // samples_per_design for run in dataset.run_numbers]
        if any(design not in DESIGN_CLUSTER for design in self.designs):
            raise ValueError("Design sensitivity requires the known 1704 geometry clusters")
        self.clusters = [DESIGN_CLUSTER[design] for design in self.designs]
        self.groups = defaultdict(list)
        for index, location in enumerate(self.locations):
            self.groups[location].append(index)
        for location, indices in self.groups.items():
            reference = indices[0]
            if len({self.designs[i] for i in indices}) != len(indices):
                raise ValueError(f"Duplicate training design at impact location {location}")
            for index in indices[1:]:
                if not np.allclose(dataset.indentor_positions[index], dataset.indentor_positions[reference], rtol=0, atol=1e-3):
                    raise ValueError(f"Impact XY mismatch at paired location {location}")
                if not np.array_equal(dataset.time_arrays[index], dataset.time_arrays[reference]):
                    raise ValueError(f"Time grids differ at paired location {location}; no implicit interpolation is allowed")
        if not any(len({self.clusters[i] for i in indices}) > 1 for indices in self.groups.values()):
            raise ValueError("No matched-location training pairs from different geometry clusters")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return {**self.dataset[index], "location": self.locations[index], "cluster": self.clusters[index]}


class MatchedDesignBatchSampler(Sampler):
    """Keep cross-cluster pairs together; use every training run once per epoch.

    At each location, repeatedly pair the two largest remaining cluster pools.
    Random tie-breaking rotates partners and unmatched designs across epochs.
    Shuffle pairs, pack them into even batches, then include all leftover runs.
    """

    def __init__(self, dataset, batch_size, seed):
        if batch_size < 2 or batch_size % 2:
            raise ValueError("Design sensitivity requires an even batch_size >= 2")
        self.dataset, self.batch_size, self.seed = dataset, batch_size, seed
        self.epoch = 0

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        pairs, singles = [], []
        for indices in self.dataset.groups.values():
            pools = defaultdict(list)
            for i in rng.permutation(indices):
                pools[self.dataset.clusters[i]].append(int(i))
            while len(pools) >= 2:
                keys = list(pools)
                rng.shuffle(keys)
                keys.sort(key=lambda key: len(pools[key]), reverse=True)
                pairs.append([pools[keys[0]].pop(), pools[keys[1]].pop()])
                pools = {key: pool for key, pool in pools.items() if pool}
            singles.extend(i for pool in pools.values() for i in pool)
        rng.shuffle(pairs)
        rng.shuffle(singles)
        order = [i for pair in pairs for i in pair] + singles
        for start in range(0, len(order), self.batch_size):
            yield order[start:start + self.batch_size]


def collate_matched_designs(items):
    batch = collate_fn(items)
    # Include every eligible pair present in the batch, not just the pairs
    # chosen to arrange it. Same-cluster near-clones do not supply this loss.
    batch["design_pairs"] = [
        (a, b) for a in range(len(items)) for b in range(a + 1, len(items))
        if items[a]["location"] == items[b]["location"] and items[a]["cluster"] != items[b]["cluster"]
    ]
    return batch


def difference_loss(prediction, target, time_batch, pairs):
    """Mean over pairs of MSE of signed response differences (normalized g)."""
    if not pairs:
        return prediction.sum() * 0.0
    residual = prediction - target
    return torch.stack([
        (residual[time_batch == a] - residual[time_batch == b]).square().mean()
        for a, b in pairs
    ]).mean()


def matched_training_loader(dataset, config):
    if config.data_format != "euroncap1704":
        raise ValueError("Design sensitivity is defined only for --data-format euroncap1704")
    matched = MatchedDesignDataset(dataset, config.samples_per_design)
    return DataLoader(
        matched, batch_sampler=MatchedDesignBatchSampler(matched, config.batch_size, config.seed),
        collate_fn=collate_matched_designs, pin_memory=config.device.type == "cuda",
    )


class DesignSensitivityTrainer(Trainer):
    """Keep the existing optimizer, validation and best-checkpoint selection."""

    def __init__(self, *args, difference_weight, **kwargs):
        super().__init__(*args, **kwargs)
        self.difference_weight = difference_weight
        self.train_mse_losses, self.train_difference_losses, self.train_pair_counts = [], [], []

    def train_epoch(self):
        self.model.train()
        objective_sum = mse_sum = pair_sum = 0.0
        point_count = pair_count = batch_count = 0
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            tensors = {key: batch[key].to(self.config.device) for key in (
                "mesh", "mesh_batch", "indentor", "time", "time_batch", "acceleration",
            )}
            target = tensors.pop("acceleration")
            prediction = self.model(**tensors, batch_size=batch["batch_size"])
            mse = self.criterion(prediction, target)
            sensitivity = difference_loss(prediction, target, tensors["time_batch"], batch["design_pairs"])
            loss = mse + self.difference_weight * sensitivity
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training objective")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            objective_sum += loss.item()
            mse_sum += mse.item() * len(target)
            pair_sum += sensitivity.item() * len(batch["design_pairs"])
            point_count += len(target)
            pair_count += len(batch["design_pairs"])
            batch_count += 1
        if not pair_count:
            raise RuntimeError("An entire training epoch contained no design pairs")
        self.train_mse_losses.append(mse_sum / point_count)
        self.train_difference_losses.append(pair_sum / pair_count)
        self.train_pair_counts.append(pair_count)
        wandb.log({
            "epoch": len(self.train_losses) + 1,
            "train/acceleration_mse": self.train_mse_losses[-1],
            "train/design_difference_mse": self.train_difference_losses[-1],
            "train/design_pairs": pair_count,
        }, commit=False)
        print(f"Design pairs: {pair_count} | Acceleration MSE: {self.train_mse_losses[-1]:.6f} | "
              f"Difference MSE: {self.train_difference_losses[-1]:.6f}")
        return objective_sum / batch_count

    def train(self):
        history = super().train()
        return {**history, "train_mse_losses": self.train_mse_losses,
                "train_difference_losses": self.train_difference_losses,
                "train_pair_counts": self.train_pair_counts,
                "design_difference_weight": self.difference_weight}

    def checkpoint_state(self):
        return {**super().checkpoint_state(),
                "train_mse_losses": self.train_mse_losses,
                "train_difference_losses": self.train_difference_losses,
                "train_pair_counts": self.train_pair_counts}

    def load_checkpoint(self, filepath):
        checkpoint = super().load_checkpoint(filepath)
        completed = len(self.train_losses)
        # Older checkpoints kept the objective, but omitted these components.
        # Keep unknown measurements missing instead of inventing MSE values.
        for key in ("train_mse_losses", "train_difference_losses", "train_pair_counts"):
            values = checkpoint.get(key, [None] * completed)
            if len(values) != completed:
                raise ValueError(f"Checkpoint {key} does not match its epoch count")
            setattr(self, key, list(values))
        self.train_loader.batch_sampler.epoch = completed
        return checkpoint
