"""Real CPU detector checks plus guard-policy edge cases.

Integration tests use the wheel's bundled ONNX models, without network calls.
"""

import importlib.util
import json
import os
from pathlib import Path
import socket
import unittest
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "promo_ocr_test_module", ROOT / "promo_typesetter" / "ocr.py"
)
ocr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ocr)
FONT = ROOT / "fonts" / "NotoSansCJKsc-Regular.otf"
if not FONT.is_file():
    FONT = ROOT.parent / "qwen-promo-font-v3" / "custom_nodes" / "exact_promo_text" / "fonts" / FONT.name
HAS_RAPIDOCR = importlib.util.find_spec("rapidocr_onnxruntime") is not None
HAS_CV2 = importlib.util.find_spec("cv2") is not None
BOX = [[20.0, 20.0], [90.0, 20.0], [90.0, 40.0], [20.0, 40.0]]


class BackgroundGuardPolicyTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (200, 120), "#f6ecd9")

    def assert_error(self, code, callback):
        with self.assertRaises(ocr.OCRGuardError) as caught:
            callback()
        error = caught.exception
        self.assertEqual(error.code, code)
        self.assertEqual(json.loads(str(error))["code"], code)
        self.assertTrue(error.needs_adjustment)
        return error

    def test_default_rejects_unreadable_regions_without_recognizing_words(self):
        # Detection-only engine output has no word/confidence to filter away.
        fake_engine = mock.Mock(return_value=([BOX], [0.015]))
        with mock.patch.object(ocr, "_get_engine", return_value=fake_engine):
            error = self.assert_error(
                "BACKGROUND_TEXT_DETECTED", lambda: ocr.inspect_background(self.image)
            )
        self.assertEqual(error.details["unexpected_count"], 1)
        self.assertEqual(error.details["status"], "rejected")
        self.assertFalse(error.details["recognition_used"])
        args, kwargs = fake_engine.call_args
        self.assertEqual(args[0].shape, (120, 200, 3))
        self.assertEqual(kwargs, {"use_det": True, "use_cls": False, "use_rec": False})

    def test_engine_dependency_and_runtime_failure_never_report_success(self):
        with mock.patch.object(ocr, "_get_engine", side_effect=RuntimeError("missing ONNX model")):
            error = self.assert_error("OCR_UNAVAILABLE", lambda: ocr.inspect_background(self.image))
            self.assertIn("missing ONNX model", error.details["cause"])
        with mock.patch.object(ocr, "_get_engine", return_value=mock.Mock(side_effect=RuntimeError("inference failed"))):
            self.assert_error("OCR_UNAVAILABLE", lambda: ocr.inspect_background(self.image))

    def test_bad_or_recognition_shaped_engine_result_is_not_empty_success(self):
        for result in ([[BOX, "unreadable", 0.01]], [[[float("nan"), 2]] * 4], [[[1, 1]] * 4]):
            with self.subTest(result=result):
                engine = mock.Mock(return_value=(result, [0.01]))
                with mock.patch.object(ocr, "_get_engine", return_value=engine):
                    self.assert_error("OCR_UNAVAILABLE", lambda: ocr.inspect_background(self.image))

    def test_explicit_brand_allowance_requires_entire_polygon(self):
        allowed = np.zeros((120, 200), dtype=bool)
        allowed[19:42, 19:92] = True
        with mock.patch.object(ocr, "_detect_regions", return_value=([BOX], [0.01])):
            output, checks = ocr.inspect_background(self.image, allowed_text_mask=allowed)
            self.assertEqual(checks["allowed_count"], 1)
            self.assertEqual(checks["unexpected_count"], 0)
            self.assertEqual(output.tobytes(), self.image.tobytes())
            allowed[30, 50] = False  # Test a hole, not just polygon corners.
            self.assert_error(
                "BACKGROUND_TEXT_DETECTED",
                lambda: ocr.inspect_background(self.image, allowed_text_mask=allowed),
            )

    def test_mask_shape_nan_range_and_mode_fail_before_inference(self):
        masks = [np.ones((2, 3)), np.full((120, 200), np.nan), np.full((120, 200), 2.0),
                 np.full((120, 200), -1), Image.new("RGB", self.image.size)]
        with mock.patch.object(ocr, "_detect_regions") as detect:
            for mask in masks:
                with self.subTest(mask_type=type(mask).__name__):
                    self.assert_error(
                        "INVALID_OCR_MASK", lambda: ocr.inspect_background(self.image, cleanup_mask=mask)
                    )
            self.assert_error("INVALID_OCR_MODE", lambda: ocr.inspect_background(self.image, mode="off"))
            self.assert_error("IMAGE_UNREADABLE", lambda: ocr.inspect_background(None))
            detect.assert_not_called()

    def test_unprovided_cleanup_mask_never_erases_letters(self):
        with mock.patch.object(ocr, "_detect_regions", return_value=([BOX], [0.01])):
            error = self.assert_error(
                "UNSAFE_TEXT_REPAIR", lambda: ocr.inspect_background(self.image, mode="repair_masked")
            )
            self.assertEqual(error.details["status"], "unsafe_repair")

    def test_cleanup_requires_dilation_margin_and_respects_protected_pixels(self):
        tight = np.zeros((120, 200), dtype=np.uint8)
        tight[20:41, 20:91] = 255
        full = np.ones((120, 200), dtype=bool)
        protected = np.zeros_like(full)
        protected[18, 18] = True  # Outside box but inside its repair margin.
        with mock.patch.object(ocr, "_detect_regions", return_value=([BOX], [0.01])):
            self.assert_error(
                "UNSAFE_TEXT_REPAIR",
                lambda: ocr.inspect_background(self.image, cleanup_mask=tight, mode="repair_masked"),
            )
            self.assert_error(
                "UNSAFE_TEXT_REPAIR",
                lambda: ocr.inspect_background(self.image, cleanup_mask=full, protected_mask=protected, mode="repair_masked"),
            )

    @unittest.skipUnless(HAS_CV2, "OpenCV dependency is required for actual inpainting")
    def test_masked_repair_is_rechecked_and_does_not_touch_other_pixels(self):
        image = self.image.copy()
        ImageDraw.Draw(image).rectangle((25, 23, 85, 37), fill="#211306")
        cleanup = np.zeros((120, 200), dtype=bool)
        cleanup[15:46, 15:96] = True
        with mock.patch.object(ocr, "_detect_regions", side_effect=[([BOX], [0.01]), ([], [0.02])]) as detector:
            output, checks = ocr.inspect_background(image, cleanup_mask=cleanup, mode="repair_masked")
        self.assertEqual(detector.call_count, 2)
        self.assertEqual(checks["status"], "repaired")
        self.assertEqual(checks["post_repair_unexpected_count"], 0)
        before, after = np.asarray(image), np.asarray(output)
        self.assertTrue(np.array_equal(before[~cleanup], after[~cleanup]))
        self.assertFalse(np.array_equal(before, after))

    @unittest.skipUnless(HAS_CV2, "OpenCV dependency is required for actual inpainting")
    def test_remaining_text_and_unavailable_recheck_block_output(self):
        cleanup = np.ones((120, 200), dtype=bool)
        with mock.patch.object(ocr, "_detect_regions", side_effect=[([BOX], []), ([BOX], [])]):
            error = self.assert_error(
                "BACKGROUND_TEXT_REMAINS",
                lambda: ocr.inspect_background(self.image, cleanup_mask=cleanup, mode="repair_masked"),
            )
            self.assertEqual(error.details["status"], "rejected")
        unavailable = ocr.OCRGuardError("OCR_UNAVAILABLE", "detector unavailable", "restore detector")
        with mock.patch.object(ocr, "_detect_regions", side_effect=[([BOX], []), unavailable]):
            self.assert_error(
                "OCR_UNAVAILABLE",
                lambda: ocr.inspect_background(self.image, cleanup_mask=cleanup, mode="repair_masked"),
            )


@unittest.skipUnless(HAS_RAPIDOCR and HAS_CV2 and FONT.is_file(), "Install requirements and bundled CJK font for real CPU integration")
class RealCPUOCRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A dependency/model-loading failure is a failure, not a skipped OCR pass.
        ocr._ENGINE = None
        with mock.patch.object(socket.socket, "connect", side_effect=AssertionError("OCR must work offline")):
            cls.engine = ocr._get_engine()

    def text_image(self, text="咖啡还是我的好"):
        image = Image.new("RGB", (900, 320), "#fff8ee")
        font = ImageFont.truetype(str(FONT), 88)
        ImageDraw.Draw(image).text((52, 76), text, font=font, fill="#3f2519")
        return image

    def test_bundled_models_run_offline_on_cpu_and_detect_cjk(self):
        self.assertEqual(self.engine.text_det.infer.session.get_providers(), ["CPUExecutionProvider"])
        with self.assertRaises(ocr.OCRGuardError) as caught:
            ocr.inspect_background(self.text_image())
        self.assertEqual(caught.exception.code, "BACKGROUND_TEXT_DETECTED")
        self.assertGreater(caught.exception.details["unexpected_count"], 0)
        self.assertFalse(caught.exception.details["recognition_used"])

    def test_chinese_traditional_english_and_mixed_regions_are_all_guarded(self):
        for text in ("原樣繁體文字", "Coffee & Croissant", "咖啡 Coffee 原樣"):
            with self.subTest(text=text):
                with self.assertRaises(ocr.OCRGuardError) as caught:
                    ocr.inspect_background(self.text_image(text))
                self.assertEqual(caught.exception.code, "BACKGROUND_TEXT_DETECTED")

    def test_clean_soft_product_scene_passes(self):
        yy, xx = np.mgrid[:512, :768]
        rgb = np.stack((244 - yy * .04, 226 - yy * .06, 193 - yy * .07), axis=-1)
        image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
        shadow = Image.new("RGBA", image.size)
        draw = ImageDraw.Draw(shadow)
        draw.ellipse((170, 340, 620, 450), fill=(54, 30, 10, 70))
        image = Image.alpha_composite(image.convert("RGBA"), shadow.filter(ImageFilter.GaussianBlur(24))).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.ellipse((255, 160, 530, 250), fill="#765139")
        draw.rounded_rectangle((255, 202, 530, 376), radius=50, fill="#936442")
        draw.ellipse((255, 152, 530, 250), fill="#c3946e", outline="#ddc5a6", width=8)
        draw.ellipse((273, 170, 512, 234), fill="#503624")
        output, checks = ocr.inspect_background(image)
        self.assertEqual(checks["unexpected_count"], 0)
        self.assertEqual(checks["status"], "passed")
        self.assertEqual(output.tobytes(), image.tobytes())

    def test_real_detector_masked_repair_removes_only_explicit_safe_text(self):
        image = self.text_image()
        cleanup = np.zeros((image.height, image.width), dtype=np.uint8)
        cleanup[30:270, 15:860] = 255
        repaired, checks = ocr.inspect_background(image, cleanup_mask=cleanup, mode="repair_masked")
        self.assertEqual(checks["status"], "repaired")
        self.assertGreater(checks["detected_count"], 0)
        self.assertEqual(checks["post_repair_unexpected_count"], 0)
        outside = cleanup == 0
        self.assertTrue(np.array_equal(np.asarray(image)[outside], np.asarray(repaired)[outside]))

    @unittest.skipUnless(os.environ.get("PROMO_BAD_POSTER"), "Optional private poster regression; set PROMO_BAD_POSTER")
    def test_private_bad_poster_is_rejected(self):
        # Do not publish a user's screenshot or require it for package CI.
        with Image.open(os.environ["PROMO_BAD_POSTER"]) as image:
            with self.assertRaises(ocr.OCRGuardError) as caught:
                ocr.inspect_background(image)
        self.assertEqual(caught.exception.code, "BACKGROUND_TEXT_DETECTED")
        self.assertGreater(caught.exception.details["unexpected_count"], 0)


if __name__ == "__main__":
    unittest.main()
