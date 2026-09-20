import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FrontendGateRequirementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (ROOT / "frontend" / "main.html").read_text(encoding="utf-8")

    def test_frontend_requires_three_regular_gates(self):
        self.assertRegex(
            self.html,
            re.compile(r"const\s+MIN_REGULAR_GATES\s*=\s*3\s*;"),
        )
        self.assertIn("JSON 至少需要三條 Gate", self.html)
        self.assertIn("請先完成至少三條一般 Gate", self.html)


if __name__ == "__main__":
    unittest.main()
