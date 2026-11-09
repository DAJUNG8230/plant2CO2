"""Whitelisted, single-variable DeepLab V5 pilot experiments."""

BASE_CLASS_WEIGHTS = [1.20, 0.80, 1.25, 1.45, 1.00, 0.90]

# Do not edit these mappings with Codex output. A reviewed code change is
# required to expand the permitted search space.
EXPERIMENTS = {
    "shrub_weight": {
        "class_weights": [1.20, 0.80, 1.25, 1.65, 1.00, 0.90],
        "changed_class": "Shrub",
        "hypothesis": "Raise the contribution of Shrub pixels to weighted cross entropy.",
        "risk": "May increase shrub false positives or reduce other class IoUs.",
    },
    "cultivated_weight": {
        "class_weights": [1.20, 0.80, 1.25, 1.45, 1.00, 1.10],
        "changed_class": "Cultivated",
        "hypothesis": "Raise the contribution of Cultivated pixels to weighted cross entropy.",
        "risk": "May misclassify Shrub or bare land as cultivated.",
    },
}


def get_experiment(identifier):
    if identifier is None:
        return None
    if identifier not in EXPERIMENTS:
        raise ValueError("Unknown experiment ID")
    return dict(EXPERIMENTS[identifier])
