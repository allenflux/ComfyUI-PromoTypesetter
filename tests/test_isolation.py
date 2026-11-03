"""Verify the public node path does not load OCR libraries into the host."""
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".venv-ocr" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


class HostIsolationTests(unittest.TestCase):
    @unittest.skipUnless(RUNTIME.is_file(), "Run scripts/setup_ocr.py first")
    def test_public_guard_keeps_host_numpy_and_does_not_import_cv2_or_onnx(self):
        code = """
import sys
from importlib.metadata import distributions, version
from PIL import Image
from promo_typesetter.ocr import inspect_background
before = {d.metadata['Name']: d.version for d in distributions()}
numpy_before = version('numpy')
image = Image.new('RGB', (512, 512), '#f6eedf')
result, checks = inspect_background(image)
assert result.tobytes() == image.tobytes()
assert checks['execution'] == 'isolated_python_process'
assert checks['unexpected_count'] == 0
assert version('numpy') == numpy_before
assert before == {d.metadata['Name']: d.version for d in distributions()}
assert 'cv2' not in sys.modules
assert 'onnxruntime' not in sys.modules
assert 'rapidocr_onnxruntime' not in sys.modules
print('Host NumPy preserved:', numpy_before)
"""
        output = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(output.returncode, 0, output.stdout + output.stderr)
        self.assertIn("Host NumPy preserved:", output.stdout)


if __name__ == "__main__":
    unittest.main()
