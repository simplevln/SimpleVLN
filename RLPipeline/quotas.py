"""Step-bin quotas. No simulator imports; empty ratios select random sampling."""
import math


def ratio_caps(text, total, max_bins=8):
    if not text or not text.strip():
        return None
    if total <= 0:
        raise ValueError("Sample count must be positive")
    weights = [float(x.strip()) for x in text.split(",")]
    if not 1 <= len(weights) <= max_bins:
        raise ValueError(f"Expected 1..{max_bins} comma-separated step-bin weights")
    if any(not math.isfinite(x) or x < 0 for x in weights) or not any(weights):
        raise ValueError("Ratios must be finite, nonnegative, with at least one positive weight")
    # Normalize before summing to avoid overflow for large finite weights.
    largest = max(weights)
    weights = [x / largest for x in weights]
    scaled = [total * x / sum(weights) for x in weights]
    caps = [math.floor(x) for x in scaled]
    order = sorted(range(len(caps)), key=lambda i: (-(scaled[i] - caps[i]), i))
    for i in order[:total - sum(caps)]:
        caps[i] += 1
    return caps
