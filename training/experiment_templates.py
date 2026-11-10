"""Whitelisted DeepLab V5 experiments; each changes one CE class weight."""
BASE_CLASS_WEIGHTS = [1.20, 0.80, 1.25, 1.45, 1.00, 0.90]
EXPERIMENTS = {
    "shrub_weight": {
        "class_weights": [1.20, 0.80, 1.25, 1.65, 1.00, 0.90],
        "changed_class": "Shrub",
        "direction": "increase",
        "hypothesis": "More Shrub emphasis may improve recall when Shrub is underpredicted.",
        "risk": "May increase Shrub false positives.",
    },
    "shrub_weight_down": {
        "class_weights": [1.20, 0.80, 1.25, 1.25, 1.00, 0.90],
        "changed_class": "Shrub",
        "direction": "decrease",
        "hypothesis": "Less Shrub emphasis may reduce false positives when Shrub is overpredicted.",
        "risk": "May reduce Shrub recall.",
    },
    "cultivated_weight": {
        "class_weights": [1.20, 0.80, 1.25, 1.45, 1.00, 1.10],
        "changed_class": "Cultivated",
        "direction": "increase",
        "hypothesis": "More Cultivated emphasis may improve recall when underpredicted.",
        "risk": "May increase Cultivated false positives.",
    },
    "cultivated_weight_down": {
        "class_weights": [1.20, 0.80, 1.25, 1.45, 1.00, 0.75],
        "changed_class": "Cultivated",
        "direction": "decrease",
        "hypothesis": "Less Cultivated emphasis may reduce false positives when overpredicted.",
        "risk": "May reduce Cultivated recall.",
    },
    "baseline_control": {
        "class_weights": BASE_CLASS_WEIGHTS[:],
        "changed_class": "None",
        "direction": "control",
        "hypothesis": "Matched warm-start and epochs with unchanged CE weights.",
        "risk": "Consumes extra GPU time.",
    },
}


def get_experiment(identifier):
    if identifier is None:
        return None
    if identifier not in EXPERIMENTS:
        raise ValueError("Unknown experiment ID")
    return dict(EXPERIMENTS[identifier])
