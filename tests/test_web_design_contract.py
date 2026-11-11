"""Static regression tests for the plant2CO2 dark geo-analytics template.
Run: python -m unittest discover -s tests -p test_web_design_contract.py
"""
import re
import unittest
from pathlib import Path

HTML = (Path(__file__).resolve().parents[1] / "templates" / "index.html").read_text(encoding="utf-8")


class WebDesignContractTests(unittest.TestCase):
    def test_analysis_inputs_and_results_are_preserved(self):
        for node in (
            "file", "drop", "analyze", "reset", "model", "gsd", "coeffPreset",
            "coeff", "mode", "progress", "progressFill", "originalImg", "resultImg",
            "compareTop", "compareRange", "compareDivider", "leftLabel",
            "compareControlLabel", "dynamicClasses", "modelName", "encoderName",
            "classesInfo", "miou", "note", "vegRate", "vegArea", "carbon", "time",
            "meta", "empty",
        ):
            with self.subTest(element_id=node):
                self.assertRegex(HTML, r'\bid="' + re.escape(node) + r'"')

    def test_api_contract_and_model_selectors(self):
        self.assertIn('fetch("/api/analyze"', HTML)
        for param in ("image", "model", "gsd", "carbon_coefficient"):
            with self.subTest(param=param):
                self.assertIn('fd.append("' + param + '"', HTML)
        for model in ("deeplabv3plus", "unet", "segformer", "pspnet"):
            with self.subTest(model=model):
                self.assertIn('value="' + model + '"', HTML)
        for output in ("mask_url", "overlay_url", "class_stats",
                       "vegetation_area_m2", "carbon_sink_kg_co2e_per_year"):
            self.assertIn(output, HTML)

    def test_security_and_accessibility_controls(self):
        self.assertNotIn("dynamic.innerHTML", HTML)
        self.assertIn('role="status"', HTML)
        self.assertIn('aria-live="polite"', HTML)
        self.assertIn("prefers-reduced-motion", HTML)
        self.assertIn('type="range"', HTML)

    def test_botanical_editorial_design_palette_and_fonts(self):
        self.assertIn("--bg:#eceae0", HTML)
        self.assertIn("--lime:#376b46", HTML)
        self.assertIn("Cormorant Garamond", HTML)
        self.assertIn("DM Mono", HTML)
        self.assertIn("植被提取技術與", HTML)
        self.assertIn("影像來源分析", HTML)
        self.assertIn("建立碳匯系統", HTML)
        self.assertIn("study-leaf", HTML)
        self.assertNotIn("content=\"#101b18\"", HTML)


if __name__ == "__main__":
    unittest.main()
