"""Continuous nested subsets matching PiG-Avatar's Splat_LODs semantics.

Keep this module importable without torch for the manifest adapter and CLI.
The actual permutation is checkpointed, not regenerated when restoring a run.
"""

import argparse
import math


def validate_fraction(value):
    fraction = float(value)
    if not math.isfinite(fraction) or not 0.1 <= fraction <= 1.0:
        raise ValueError("LOD fraction must be finite and between 0.1 and 1.0.")
    return fraction


def fraction_argument(value):
    try:
        return validate_fraction(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def percent_argument(value):
    try:
        return validate_fraction(float(value) / 100.0) * 100.0
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("LOD percentage must be between 10 and 100.") from exc


def lod_label(fraction):
    return format(validate_fraction(fraction) * 100, ".12g").replace(".", "p") + "pct"


def lod_config(min_fraction=0.1, max_fraction=1.0, seed=0):
    minimum, maximum = validate_fraction(min_fraction), validate_fraction(max_fraction)
    if minimum > maximum:
        raise ValueError("LOD minimum must not exceed maximum.")
    return {"min_fraction": minimum, "max_fraction": maximum, "seed": int(seed)}


def add_training_arguments(parser):
    parser.add_argument("--lod-min-fraction", type=fraction_argument, default=None)
    parser.add_argument("--lod-max-fraction", type=fraction_argument, default=None)
    parser.add_argument("--lod-seed", type=int, default=None)


def resolve_config(template, args):
    """CLI overrides only LOD options; never alter the training configuration."""
    config = dict(template.get("lod", {}))
    for name in ("min_fraction", "max_fraction", "seed"):
        value = getattr(args, "lod_" + name, None)
        if value is not None:
            config[name] = value
    return lod_config(**config)


class LODSelection:
    def __init__(self, min_fraction=0.1, max_fraction=1.0, seed=0):
        self.config = lod_config(min_fraction, max_fraction, seed)
        self.permutation = None

    def initialize(self, count, device):
        import torch

        if count < 1:
            raise ValueError("LOD requires at least one Gaussian.")
        generator = torch.Generator(device="cpu").manual_seed(self.config["seed"])
        self.permutation = torch.randperm(count, generator=generator).to(device)

    def indices(self, fraction):
        fraction = validate_fraction(fraction)
        if self.permutation is None or not self.permutation.numel():
            raise ValueError("LOD permutation has not been initialized.")
        count = max(1, math.floor(fraction * self.permutation.numel()))
        return self.permutation[:count]

    def sample(self):
        import torch

        minimum, maximum = self.config["min_fraction"], self.config["max_fraction"]
        if minimum == maximum:
            return minimum
        return torch.empty(1, device=self.permutation.device, dtype=torch.float64).uniform_(
            minimum ** 2, maximum ** 2
        ).sqrt_().clamp_(minimum, maximum).item()

    def restore(self, permutation, count, device):
        import torch

        if (not torch.is_tensor(permutation) or permutation.dtype != torch.long
                or permutation.ndim != 1 or permutation.numel() != count or count < 1):
            raise ValueError("Invalid checkpoint LOD permutation.")
        permutation = permutation.to(device)
        if not torch.equal(permutation.sort().values, torch.arange(count, device=device)):
            raise ValueError("Checkpoint LOD permutation must contain every Gaussian exactly once.")
        self.permutation = permutation
