"""SegFormer v3 frontend adapter tests; no GPU or real checkpoint needed."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import inference


class _ConstantSixClassModel:
    def __call__(self, pixel_values):
        logits = torch.zeros(
            (1, 6, 128, 128), device=pixel_values.device, dtype=torch.float32
        )
        logits[:, 3] = 10
        return type("Output", (), {"logits": logits})()


class SegFormerV3InferenceTests(unittest.TestCase):
    def test_deployment_path_has_priority(self):
        self.assertEqual(
            inference.MODEL_CONFIG["segformer"]["paths"][0].name,
            "segformer_best_v3",
        )

    def test_resized_black_padded_prediction_restores_original_shape(self):
        info = {
            "mode": "teammate_v3",
            "size": 512,
            "mean": np.asarray([0.485, 0.456, 0.406], dtype=np.float32),
            "std": np.asarray([0.229, 0.224, 0.225], dtype=np.float32),
        }
        with patch.dict(inference._PROCESSORS, {"segformer": info}), patch.object(
            inference, "load_model", return_value=_ConstantSixClassModel()
        ):
            for shape in ((900, 1400, 3), (1400, 900, 3), (180, 180, 3)):
                with self.subTest(shape=shape):
                    rgb = np.zeros(shape, dtype=np.uint8)
                    mask = inference._predict_segformer(rgb)
                    self.assertEqual(mask.shape, shape[:2])
                    self.assertEqual(mask.dtype, np.uint8)
                    self.assertTrue(np.all(mask == 3))

    def test_deployment_class_order_matches_six_ground_categories(self):
        self.assertEqual(inference.VEGETATION_CLASS_IDS, [2, 3, 4, 5])
        self.assertEqual(len(inference.CLASS_NAMES), 6)


if __name__ == "__main__":
    unittest.main()
