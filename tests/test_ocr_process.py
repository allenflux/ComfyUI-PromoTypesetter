"""Private OCR protocol checks, including actual subprocess CPU detection."""
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "promo_process_test_package"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "promo_typesetter")]
sys.modules[PACKAGE] = package
process = importlib.import_module(PACKAGE + ".ocr_process")
ocr = importlib.import_module(PACKAGE + ".ocr")
worker = importlib.import_module(PACKAGE + ".ocr_worker")


def checks(mode="reject"):
    return {"status": "passed", "mode": mode, "detector": "rapidocr-onnxruntime",
            "detector_version": "1.4.4", "provider": "CPUExecutionProvider",
            "recognition_used": False, "detected_count": 0, "unexpected_count": 0}


class WorkerProtocolTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="promo-worker-tests-")
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.interpreter = self.path / "private-python"
        self.interpreter.write_text("test marker", encoding="utf-8")
        patch = mock.patch.object(process, "runtime_python", return_value=self.interpreter)
        patch.start()
        self.addCleanup(patch.stop)
        self.image = Image.new("RGB", (80, 60), "#f6eee0")

    def request_error(self, expected, callback):
        with self.assertRaises(ocr.OCRGuardError) as caught:
            callback()
        self.assertEqual(caught.exception.code, expected)
        self.assertTrue(caught.exception.needs_adjustment)
        return caught.exception

    def reply(self, command, **kwargs):
        request_path = Path(command[command.index("--request") + 1])
        response_path = Path(command[command.index("--response") + 1])
        request = json.loads(request_path.read_text())
        with Image.open(request_path.parent / request["image"]) as opened:
            opened.save(request_path.parent / "output.png", format="PNG")
        response_path.write_text(json.dumps({"protocol": "1", "status": "ok",
                                            "output": "output.png", "checks": checks(request["mode"])}))
        self.last_command, self.last_options, self.last_request = command, kwargs, request
        return subprocess.CompletedProcess(command, 0, b"", b"")

    def test_success_runs_only_private_interpreter_and_keeps_pixels(self):
        with mock.patch.dict(os.environ, {"PYTHONPATH": "/host/site-packages", "PYTHONHOME": "/host/python",
                                         "PIP_CONSTRAINT": "/host/constraints.txt"}):
            with mock.patch.object(process.subprocess, "run", side_effect=self.reply) as run:
                image, inspected = process.inspect_isolated(self.image)
        self.assertEqual(image.tobytes(), self.image.tobytes())
        self.assertEqual(inspected["execution"], "isolated_python_process")
        self.assertEqual(run.call_count, 1)
        self.assertEqual(self.last_command[0], str(self.interpreter))
        self.assertEqual(self.last_options["timeout"], 120)
        self.assertFalse(self.last_options.get("shell", False))
        environment = self.last_options["env"]
        self.assertNotIn("PYTHONPATH", environment)
        self.assertNotIn("PYTHONHOME", environment)
        self.assertEqual(environment["PYTHONNOUSERSITE"], "1")
        self.assertEqual(environment["PIP_CONSTRAINT"], "/host/constraints.txt")

    def test_masks_are_validated_before_serialization_and_remain_binary(self):
        invalid = np.full((60, 80), np.nan)
        with mock.patch.object(process.subprocess, "run") as run:
            self.request_error("INVALID_OCR_MASK", lambda: process.inspect_isolated(self.image, cleanup_mask=invalid))
            run.assert_not_called()
        allowed = np.zeros((60, 80), dtype=bool)
        allowed[12:24, 18:32] = True

        def reply(command, **kwargs):
            path = Path(command[command.index("--request") + 1])
            request = json.loads(path.read_text())
            mask_path = path.parent / request["masks"]["allowed_text_mask"]
            with Image.open(mask_path) as opened:
                self.assertEqual(opened.mode, "L")
                np.testing.assert_array_equal(np.asarray(opened), allowed.astype(np.uint8) * 255)
            return self.reply(command, **kwargs)

        with mock.patch.object(process.subprocess, "run", side_effect=reply):
            process.inspect_isolated(self.image, allowed_text_mask=allowed)

    def test_missing_runtime_does_not_fall_back_to_host_or_allow_image(self):
        self.interpreter.unlink()
        with mock.patch.object(process.subprocess, "run") as run:
            self.request_error("OCR_RUNTIME_MISSING", lambda: process.inspect_isolated(self.image))
            run.assert_not_called()

    def test_timeout_and_start_failure_are_fail_closed(self):
        for exception, code in ((subprocess.TimeoutExpired("ocr", 120), "OCR_WORKER_TIMEOUT"),
                                (PermissionError("private path"), "OCR_WORKER_FAILED")):
            with self.subTest(code=code):
                with mock.patch.object(process.subprocess, "run", side_effect=exception):
                    error = self.request_error(code, lambda: process.inspect_isolated(self.image))
                self.assertNotIn("private path", str(error))

    def test_expected_rejection_preserves_original_error_details(self):
        detail = {"status": "rejected", "unexpected_count": 2, "boxes": []}

        def reply(command, **kwargs):
            path = Path(command[command.index("--response") + 1])
            path.write_text(json.dumps({"protocol": "1", "status": "error", "error": {
                "code": "BACKGROUND_TEXT_DETECTED", "message": "检测到未经允许的文字。",
                "needs_adjustment": ["重新生成无字底图"], "details": detail}}))
            return subprocess.CompletedProcess(command, 2, b"", b"private diagnostics must not be printed")

        with mock.patch.object(process.subprocess, "run", side_effect=reply):
            error = self.request_error("BACKGROUND_TEXT_DETECTED", lambda: process.inspect_isolated(self.image))
        self.assertEqual(error.details, detail)
        self.assertNotIn("private diagnostics", str(error))

    def test_missing_invalid_and_wrong_protocol_responses_never_pass(self):
        responses = [None, "{", {"protocol": "9", "status": "ok"},
                     {"protocol": "1", "status": "error", "error": {}},
                     {"protocol": "1", "status": "ok", "output": "/unrelated/image.png", "checks": checks()},
                     {"protocol": "1", "status": "ok", "output": "output.png", "checks": {**checks(), "status": "repaired", "post_repair_unexpected_count": 0}},
                     {"protocol": "1", "status": "ok", "output": "output.png", "checks": {**checks(), "unexpected_count": 1}}]
        for response in responses:
            with self.subTest(response=response):
                def reply(command, **kwargs):
                    path = Path(command[command.index("--response") + 1])
                    if response is not None:
                        path.write_text(response if isinstance(response, str) else json.dumps(response))
                    return subprocess.CompletedProcess(command, 0, b"", b"")
                with mock.patch.object(process.subprocess, "run", side_effect=reply):
                    with self.assertRaises(ocr.OCRGuardError):
                        process.inspect_isolated(self.image)

    def test_output_size_change_and_nonzero_success_exit_are_rejected(self):
        for change_size in (True, False):
            def reply(command, **kwargs):
                done = self.reply(command, **kwargs)
                path = Path(command[command.index("--response") + 1])
                if change_size:
                    Image.new("RGB", (40, 40)).save(path.parent / "output.png")
                else:
                    done.returncode = 1
                return done
            with mock.patch.object(process.subprocess, "run", side_effect=reply):
                self.request_error("OCR_WORKER_PROTOCOL_ERROR", lambda: process.inspect_isolated(self.image))

    def test_worker_rejects_path_escape_and_serializes_local_errors(self):
        image = self.path / "image.png"
        self.image.save(image)
        request, response = self.path / "request.json", self.path / "response.json"
        request.write_text(json.dumps({"protocol": "1", "image": "../image.png", "masks": {},
                                       "output": "output.png", "mode": "reject"}))
        self.assertEqual(worker.main(["--request", str(request), "--response", str(response)]), 1)
        self.assertEqual(json.loads(response.read_text())["error"]["code"], "OCR_WORKER_FAILED")
        request.write_text(json.dumps({"protocol": "1", "image": "image.png", "masks": {},
                                       "output": "output.png", "mode": "reject"}))
        local = types.SimpleNamespace(OCRGuardError=ocr.OCRGuardError)
        local._inspect_background_local = mock.Mock(side_effect=ocr.OCRGuardError(
            "BACKGROUND_TEXT_DETECTED", "多余文字。", ["重新生成"], {"unexpected_count": 1}))
        with mock.patch.object(worker, "_load_local_ocr", return_value=local):
            self.assertEqual(worker.process_request(request, response), 2)
        self.assertEqual(json.loads(response.read_text())["error"]["code"], "BACKGROUND_TEXT_DETECTED")
        self.assertFalse((self.path / "output.png").exists())


HAS_REAL_OCR = (importlib.util.find_spec("rapidocr_onnxruntime") is not None
                and importlib.util.find_spec("cv2") is not None)
TEST_WORKER_PYTHON = (process.runtime_python() if process.runtime_available()
                      else Path(sys.executable) if HAS_REAL_OCR else None)


@unittest.skipUnless(TEST_WORKER_PYTHON, "Install the private OCR runtime to exercise real CPU subprocesses")
class RealWorkerTests(unittest.TestCase):
    def test_real_cpu_worker_self_check_clean_frame_and_text_rejection(self):
        self_check = subprocess.run([str(TEST_WORKER_PYTHON), str(ROOT / "promo_typesetter" / "ocr_worker.py"), "--self-check"],
                                    capture_output=True, text=True, env=process.worker_environment(), timeout=120)
        self.assertEqual(self_check.returncode, 0, self_check.stdout)
        result = json.loads(self_check.stdout)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["provider"], "CPUExecutionProvider")
        self.assertGreater(result["detection_regions"], 0)
        with mock.patch.object(process, "runtime_python", return_value=TEST_WORKER_PYTHON):
            blank = Image.new("RGB", (900, 320), "#fff8ee")
            rendered, inspected = process.inspect_isolated(blank)
            self.assertEqual(rendered.tobytes(), blank.tobytes())
            self.assertEqual(inspected["execution"], "isolated_python_process")
            font = ImageFont.truetype(str(ROOT / "fonts" / "NotoSansCJKsc-Regular.otf"), 88)
            ImageDraw.Draw(blank).text((52, 76), "咖啡还是我的好", font=font, fill="#3f2519")
            with self.assertRaises(ocr.OCRGuardError) as caught:
                process.inspect_isolated(blank)
            self.assertEqual(caught.exception.code, "BACKGROUND_TEXT_DETECTED")
            self.assertGreater(caught.exception.details["unexpected_count"], 0)


if __name__ == "__main__":
    unittest.main()
