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
    "shrub_weight_down_more": {
        "class_weights": [1.20, 0.80, 1.25, 1.10, 1.00, 0.90],
        "changed_class": "Shrub", "direction": "decrease",
        "hypothesis": "Test stronger reduction of overpredicted Shrub.",
        "risk": "May sharply reduce Shrub recall.",
    },
    "shrub_weight_down_soft": {
        "class_weights": [1.20, 0.80, 1.25, 1.35, 1.00, 0.90],
        "changed_class": "Shrub", "direction": "decrease",
        "hypothesis": "Test smaller reduction of overpredicted Shrub.",
        "risk": "Effect may be too small to overcome noise.",
    },
    "cultivated_weight_up_soft": {
        "class_weights": [1.20, 0.80, 1.25, 1.45, 1.00, 1.00],
        "changed_class": "Cultivated", "direction": "increase",
        "hypothesis": "Test mild improvement of underpredicted Cultivated recall.",
        "risk": "Can increase Cultivated false positives.",
    },
    "cultivated_weight_up_more": {
        "class_weights": [1.20, 0.80, 1.25, 1.45, 1.00, 1.25],
        "changed_class": "Cultivated", "direction": "increase",
        "hypothesis": "Test larger response to underpredicted Cultivated.",
        "risk": "May increase cultivated false positives.",
    },
    "herbaceous_weight_up": {
        "class_weights": [1.20, 0.80, 1.45, 1.45, 1.00, 0.90],
        "changed_class": "Herbaceous", "direction": "increase",
        "hypothesis": "Correct underpredicted Herbaceous if confusion supports it.",
        "risk": "May increase herbaceous false positives.",
    },
    "herbaceous_weight_down": {
        "class_weights": [1.20, 0.80, 1.05, 1.45, 1.00, 0.90],
        "changed_class": "Herbaceous", "direction": "decrease",
        "hypothesis": "Correct overpredicted Herbaceous if confusion supports it.",
        "risk": "May decrease Herbaceous recall.",
    },
    "trees_weight_up": {
        "class_weights": [1.20, 0.80, 1.25, 1.45, 1.15, 0.90],
        "changed_class": "Trees", "direction": "increase",
        "hypothesis": "Correct underpredicted Trees if confusion supports it.",
        "risk": "May increase Trees false positives.",
    },
    "trees_weight_down": {
        "class_weights": [1.20, 0.80, 1.25, 1.45, 0.85, 0.90],
        "changed_class": "Trees", "direction": "decrease",
        "hypothesis": "Correct overpredicted Trees if confusion supports it.",
        "risk": "May decrease Trees recall.",
    },
    "low_lr": {
        "class_weights": BASE_CLASS_WEIGHTS[:], "learning_rate": 1e-5,
        "changed_class": "None", "direction": "lr",
        "hypothesis": "Reduce LR during best-checkpoint fine tuning.",
        "risk": "May stagnate or overfit.",
    },
    "very_low_lr": {
        "class_weights": BASE_CLASS_WEIGHTS[:], "learning_rate": 5e-6,
        "changed_class": "None", "direction": "lr",
        "hypothesis": "Test conservative best-checkpoint fine tuning.",
        "risk": "May not escape plateau.",
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
