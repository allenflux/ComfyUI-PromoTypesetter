"""ComfyUI adapter integration with real CPU tensors and isolated output files."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "promo_nodes_test_package"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "promo_typesetter")]
sys.modules[PACKAGE] = package
SPEC = importlib.util.spec_from_file_location(
    PACKAGE + ".nodes", ROOT / "promo_typesetter" / "nodes.py"
)
nodes = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = nodes
SPEC.loader.exec_module(nodes)
HAS_OCR_RUNTIME = ((ROOT / ".venv-ocr" / "bin" / "python").is_file()
                   or (ROOT / ".venv-ocr" / "Scripts" / "python.exe").is_file())


def passed_ocr(image, **kwargs):
    return image.copy(), {"status": "passed", "detected_count": 0,
                          "unexpected_count": 0, "recognition_used": False}


class NodeIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="promo-node-tests-")
        self.addCleanup(self.temp.cleanup)
        self.output_dir = Path(self.temp.name)
        folder_paths = types.ModuleType("folder_paths")
        folder_paths.get_output_directory = lambda: str(self.output_dir)

        def get_save_image_path(prefix, output, width, height):
            prefix_path = Path(prefix)
            subfolder = str(prefix_path.parent) if prefix_path.parent != Path(".") else ""
            directory = Path(output) / subfolder
            directory.mkdir(parents=True, exist_ok=True)
            return str(directory), prefix_path.name, 1, subfolder, prefix

        folder_paths.get_save_image_path = get_save_image_path
        # Restore only our ComfyUI stub. Restoring the whole sys.modules mapping
        # unloads cv2/ONNX modules imported by a test and breaks later C-extension imports.
        missing = object()
        previous = sys.modules.get("folder_paths", missing)
        sys.modules["folder_paths"] = folder_paths

        def restore_folder_paths():
            if previous is missing:
                sys.modules.pop("folder_paths", None)
            else:
                sys.modules["folder_paths"] = previous

        self.addCleanup(restore_folder_paths)
        self.material = torch.full((1, 128, 160, 3), .6)

    def prepare(self, text="咖啡还是我的好", **kwargs):
        options = {"material": self.material, "copy_text": text, "width": 1024,
                   "height": 1024, "title_size": 72, "body_size": 28,
                   "design_prompt": "柔和自然光，产品位于下半部。",
                   "typography_mode": "manual"}
        options.update(kwargs)
        return nodes.PromoTextPrepare().run(**options)

    def render(self, images, context, **kwargs):
        output = nodes.PromoRenderAndSave().run(images, context, **kwargs)
        self.assertEqual(len(output["result"]), 1)
        payload = json.loads(output["result"][0])
        self.assertEqual(json.loads(output["ui"]["text"][0]), payload)
        self.assertEqual(len(output["ui"]["images"]), payload["success_count"])
        self.assertEqual(payload["success_count"] + payload["failed_count"], payload["requested_count"])
        return payload, output["ui"]["images"]

    def image_path(self, item):
        return self.output_dir / item["file"]["subfolder"] / item["file"]["filename"]

    def assert_prepare_error(self, code, **kwargs):
        with self.assertRaises(nodes.PromoError) as caught:
            self.prepare(**kwargs)
        self.assertEqual(caught.exception.code, code)
        self.assertTrue(caught.exception.needs_adjustment)

    def test_default_campaign_balances_title_and_returns_four_exact_font_images_without_ocr(self):
        material = torch.tensor([.64, .35, .13]).expand(1, 128, 160, 3).clone()
        original_material = material.clone()
        title, body = "咖啡还是我的好", "午后一起，咖啡配可颂。"
        combined = title + "\n" + body
        # Deliberately omit typography_mode: new/default UI and API callers
        # must receive the coordinated campaign layout, not the old blank wall.
        canvas, context, prompt, count = nodes.PromoTextPrepare().run(
            material=material, copy_text=title, body_text=body, width=1152,
            height=1536, count=4, title_size=112, body_size=40)
        self.assertEqual(count, 4)
        self.assertEqual(context["typography_mode"], "campaign")
        self.assertEqual((context["title_text"], context["body_text"]), (title, body))
        self.assertEqual(context["copy_text"], combined)
        self.assertEqual(canvas.shape, (1, 1536, 1152, 3))
        spatial_change = (canvas - canvas[:, :1, :1, :]).abs().max().item()
        self.assertGreater(spatial_change, .02)
        self.assertTrue(torch.equal(material, original_material))
        self.assertNotIn(title, prompt)
        self.assertNotIn(body, prompt)
        self.assertTrue(context["font"].startswith("NotoSerif"))

        batch = torch.full((4, 1536, 1152, 3), .94)
        with mock.patch.object(nodes, "inspect_background", side_effect=AssertionError("OCR must stay off")) as guard:
            result, previews = self.render(batch, context)
        guard.assert_not_called()
        self.assertEqual((result["success_count"], result["failed_count"]), (4, 0))
        self.assertEqual((result["title_text"], result["body_text"]), (title, body))
        self.assertEqual(len(previews), 4)
        self.assertEqual(len(list(self.output_dir.rglob("*.png"))), 4)
        for item in result["images"]:
            self.assertEqual(item["status"], "success")
            self.assertEqual(item["copy_text"], combined)
            self.assertEqual((item["title_text"], item["body_text"]), (title, body))
            with Image.open(self.image_path(item)) as output:
                manifest = json.loads(output.info["promo"])
                self.assertEqual(output.size, (1152, 1536))
                self.assertEqual(manifest["typography_mode"], "campaign")
                self.assertEqual((manifest["title_text"], manifest["body_text"]), (title, body))
                self.assertEqual(manifest["title_lines"], ["咖啡还是", "我的好"])
                self.assertEqual("".join(manifest["title_lines"]), title)
                self.assertEqual("".join(manifest["body_lines"]), body)
                title_styles = [line for line in manifest["line_styles"] if line["role"] == "title"]
                self.assertEqual(len(title_styles), 2)
                self.assertGreater(min(line["font_size"] for line in title_styles), 112)
                self.assertEqual(manifest["ocr"], {"status": "not_run", "mode": "off"})
                self.assertEqual(manifest["font_rendering"], "real_font_glyphs")

    def test_separate_title_and_body_preserve_crlf_spaces_and_metadata(self):
        title = "  咖啡\r\n还是我的好  "
        body = "  午后 Coffee  \r\n完整原文，保留空格。  "
        combined = title + "\n" + body
        _, context, prompt, _ = self.prepare(
            title, body_text=body, allow_mixed_scripts=True, title_size=44, body_size=26)
        self.assertEqual(context["title_text"].encode(), title.encode())
        self.assertEqual(context["body_text"].encode(), body.encode())
        self.assertNotIn(title, prompt)
        self.assertNotIn(body, prompt)
        result, _ = self.render(torch.full((1, 1024, 1024, 3), .94), context)
        self.assertEqual(result["title_text"].encode(), title.encode())
        self.assertEqual(result["body_text"].encode(), body.encode())
        item = result["images"][0]
        self.assertEqual(item["title_text"].encode(), title.encode())
        self.assertEqual(item["body_text"].encode(), body.encode())
        self.assertEqual(item["copy_text"].encode(), combined.encode())
        self.assertEqual(item["copy_sha256"], hashlib.sha256(combined.encode()).hexdigest())
        with Image.open(self.image_path(item)) as output:
            manifest = json.loads(output.info["promo"])
            self.assertEqual(manifest["title_text"].encode(), title.encode())
            self.assertEqual(manifest["body_text"].encode(), body.encode())
            self.assertEqual(manifest["title_lines"], ["  咖啡", "还是我的好  "])
            self.assertEqual(manifest["body_lines"], ["  午后 Coffee  ", "完整原文，保留空格。  "])

    def test_empty_separate_body_and_legacy_none_do_not_invent_body_copy(self):
        title = "咖啡还是我的好"
        for body in ("", None):
            with self.subTest(body=body):
                _, context, _, _ = self.prepare(title, body_text=body)
                self.assertEqual((context["title_text"], context["body_text"]), (title, ""))
                result, _ = self.render(torch.full((1, 1024, 1024, 3), .94), context)
                item = result["images"][0]
                self.assertEqual((item["title_text"], item["body_text"]), (title, ""))
                self.assertEqual(item["copy_text"], title)
                self.assertEqual(item["typography"]["body_lines"], [])

        legacy = "咖啡还是我的好\r\n原样正文  "
        _, context, _, _ = self.prepare(legacy, body_text=None, title_size=52)
        result, _ = self.render(torch.full((1, 1024, 1024, 3), .94), context)
        self.assertEqual(result["copy_text"].encode(), legacy.encode())
        self.assertEqual((result["title_text"], result["body_text"]), (title, "原样正文  "))
        self.assertEqual(result["images"][0]["typography"]["body_lines"], ["原样正文  "])

    def test_separate_body_overflow_and_language_conflicts_fail_before_sampling(self):
        body = "完整正文保持原样。" * 300
        self.assert_prepare_error("TEXT_TOO_LONG", body_text=body)
        self.assert_prepare_error("TEXT_TOO_LONG", text="完整标题保持原样。" * 300, body_text="正文")
        self.assert_prepare_error("LANGUAGE_CONFLICT", body_text="繁體廣告", language="zh-Hans")
        self.assert_prepare_error("LANGUAGE_CONFLICT", text="Coffee", body_text="中文正文", language="en")

    def test_whitespace_only_separate_body_is_preserved_without_trimming(self):
        title, body = "咖啡还是我的好", "  \r\n "
        _, context, _, _ = self.prepare(title, body_text=body)
        self.assertEqual(context["body_text"].encode(), body.encode())
        result, _ = self.render(torch.full((1, 1024, 1024, 3), .94), context)
        self.assertEqual(result["copy_text"].encode(), (title + "\n" + body).encode())
        self.assertEqual(result["body_text"].encode(), body.encode())
        manifest = result["images"][0]["typography"]
        self.assertEqual(manifest["body_text"].encode(), body.encode())
        self.assertEqual(manifest["body_lines"], ["  ", " "])

    def test_explicit_manual_mode_keeps_requested_size_and_uniform_canvas(self):
        canvas, context, _, _ = self.prepare(title_size=72, body_size=28, typography_mode="manual")
        self.assertEqual(context["typography_mode"], "manual")
        self.assertEqual(context["fonts"]["title"].size, 72)
        self.assertEqual(context["fonts"]["body"].size, 28)
        self.assertTrue(torch.all(canvas == canvas[:, :1, :1, :]).item())
        result, _ = self.render(torch.full((1, 1024, 1024, 3), .94), context)
        self.assertEqual(result["images"][0]["typography"]["title_lines"], ["咖啡还是我的好"])

    def test_campaign_honors_explicit_font_family_and_rejects_unknown_mode(self):
        material = torch.tensor([.64, .35, .13]).expand(1, 128, 160, 3).clone()
        _, context, _, _ = self.prepare(material=material, typography_mode="campaign", font_family="sans")
        self.assertTrue(context["font"].startswith("NotoSans"))
        self.assert_prepare_error("INVALID_TYPOGRAPHY_MODE", typography_mode="unknown")

    def test_one_render_invocation_handles_each_requested_batch_count(self):
        for count in (1, 2, 3, 4):
            with self.subTest(count=count):
                canvas, context, prompt, count_output = self.prepare(count=count)
                self.assertEqual(canvas.shape, (1, 1024, 1024, 3))
                self.assertEqual(count_output, count)
                self.assertNotIn(context["copy_text"], prompt)
                batch = torch.full((count, 1024, 1024, 3), .94)
                with mock.patch.object(nodes, "inspect_background", side_effect=passed_ocr) as guard:
                    result, previews = self.render(batch, context, ocr_mode="reject", filename_prefix=f"count_{count}")
                self.assertEqual(guard.call_count, count)
                self.assertEqual(result["status"], "success")
                self.assertEqual(result["success_count"], count)
                self.assertEqual(len(result["images"]), count)
                self.assertEqual([item["index"] for item in result["images"]], list(range(count)))
                self.assertEqual(len({item["filename"] for item in previews}), count)
                for item in result["images"]:
                    self.assertTrue(self.image_path(item).is_file())
                    self.assertEqual(item["copy_text"], context["copy_text"])
                    self.assertEqual((item["width"], item["height"], item["format"]), (1024, 1024, "png"))

    def test_default_ocr_off_returns_four_font_images_without_a_detector_or_runtime(self):
        _, context, _, _ = self.prepare(count=4)
        batch = torch.full((4, 1024, 1024, 3), .94)
        unavailable = nodes.OCRGuardError("OCR_RUNTIME_MISSING", "not installed", ["setup OCR"])
        with mock.patch.object(nodes, "inspect_background", side_effect=unavailable) as guard:
            result, previews = self.render(batch, context)
        guard.assert_not_called()
        self.assertEqual(result["status"], "success")
        self.assertEqual((result["success_count"], result["passed_count"], result["review_count"],
                          result["failed_count"]), (4, 4, 0, 0))
        self.assertEqual(len(previews), 4)
        self.assertEqual(len(list(self.output_dir.rglob("*.png"))), 4)
        for item in result["images"]:
            self.assertEqual(item["status"], "success")
            self.assertEqual(item["warnings"], [])
            self.assertIsNone(item["error"])
            with Image.open(self.image_path(item)) as image:
                manifest = json.loads(image.info["promo"])
                self.assertEqual(manifest["copy_text"], "咖啡还是我的好")
                self.assertEqual(manifest["font_rendering"], "real_font_glyphs")
                self.assertEqual(manifest["ocr"], {"status": "not_run", "mode": "off"})
                self.assertEqual(manifest["checks"]["background_incidental_text"], "not_checked")
                self.assertLess(np.asarray(image).min(), 240)

    def test_partial_success_preserves_one_result_per_frame_and_error_details(self):
        _, context, _, _ = self.prepare(count=4)
        batch = torch.full((4, 1024, 1024, 3), .94)
        calls = []

        def guard(image, **kwargs):
            calls.append(image)
            if len(calls) == 2:
                raise nodes.OCRGuardError(
                    "BACKGROUND_TEXT_DETECTED", "found unexpected text", ["regenerate background"],
                    {"unexpected_count": 1, "boxes": [[[10, 10], [100, 10], [100, 50], [10, 50]]]},
                )
            return passed_ocr(image, **kwargs)

        with mock.patch.object(nodes, "inspect_background", side_effect=guard):
            result, previews = self.render(batch, context, ocr_mode="reject", quality_policy="strict")
        self.assertEqual(result["status"], "partial_success")
        self.assertEqual((result["requested_count"], result["success_count"], result["failed_count"]), (4, 3, 1))
        self.assertEqual([item["status"] for item in result["images"]], ["success", "failed", "success", "success"])
        failed = result["images"][1]
        self.assertIsNone(failed["url"])
        self.assertEqual(failed["error"]["code"], "BACKGROUND_TEXT_DETECTED")
        self.assertEqual(failed["error"]["details"]["unexpected_count"], 1)
        self.assertEqual(len(list(self.output_dir.rglob("*.png"))), 3)
        self.assertEqual(len(previews), 3)

    def test_default_returns_all_four_images_when_ocr_requires_review(self):
        _, context, _, _ = self.prepare(count=4)
        batch = torch.full((4, 1024, 1024, 3), .94)
        codes = ("BACKGROUND_TEXT_DETECTED", "OCR_UNAVAILABLE", "OCR_RUNTIME_MISSING", "OCR_WORKER_TIMEOUT")
        failures = [nodes.OCRGuardError(code, "manual review needed", ["review the image"],
                    {"unexpected_count": 1, "frame": index}) for index, code in enumerate(codes)]
        with mock.patch.object(nodes, "inspect_background", side_effect=failures) as guard:
            result, previews = self.render(batch, context, ocr_mode="reject")
        self.assertEqual(guard.call_count, 4)
        self.assertEqual(result["status"], "review_required")
        self.assertEqual((result["requested_count"], result["success_count"], result["passed_count"],
                          result["review_count"], result["failed_count"]), (4, 4, 0, 4, 0))
        self.assertEqual(len(previews), 4)
        self.assertEqual(len(list(self.output_dir.rglob("*.png"))), 4)
        for index, item in enumerate(result["images"]):
            self.assertEqual(item["status"], "review_required")
            self.assertEqual(item["generation_status"], "success")
            self.assertEqual(item["validation_status"], "review_required")
            self.assertIsNone(item["error"])
            self.assertEqual(item["warnings"][0]["code"], codes[index])
            self.assertEqual(item["warnings"][0]["details"]["frame"], index)
            self.assertTrue(item["url"])
            with Image.open(self.image_path(item)) as image:
                manifest = json.loads(image.info["promo"])
                self.assertEqual(manifest["copy_text"], context["copy_text"])
                self.assertEqual(manifest["font_rendering"], "real_font_glyphs")
                self.assertEqual(manifest["checks"]["background_incidental_text"], "review_required")
                self.assertEqual(manifest["ocr"]["status"], "review_required")
                self.assertEqual(manifest["ocr"]["error"]["code"], codes[index])
                # A warning retains the actual generated scene, while the exact
                # caller copy is still drawn as a font layer in the reserved region.
                self.assertEqual(image.getpixel((1023, 1023)), (240, 240, 240))
                self.assertLess(np.asarray(image).min(), 240)

    def test_default_mixed_pass_and_review_counts_both_as_generated_images(self):
        _, context, _, _ = self.prepare(count=2)
        batch = torch.full((2, 1024, 1024, 3), .94)
        seen = []

        def guard(image, **kwargs):
            seen.append(image)
            if len(seen) == 2:
                raise nodes.OCRGuardError("BACKGROUND_TEXT_DETECTED", "possible texture", ["review"],
                                         {"unexpected_count": 1})
            return passed_ocr(image, **kwargs)

        with mock.patch.object(nodes, "inspect_background", side_effect=guard):
            result, previews = self.render(batch, context, ocr_mode="reject")
        self.assertEqual(result["status"], "review_required")
        self.assertEqual((result["success_count"], result["passed_count"], result["review_count"],
                          result["failed_count"]), (2, 1, 1, 0))
        self.assertEqual([item["status"] for item in result["images"]], ["success", "review_required"])
        self.assertEqual(len(previews), 2)
        self.assertTrue(all(self.image_path(item).is_file() for item in result["images"]))

    def test_review_image_save_failure_remains_a_real_failure(self):
        _, context, _, _ = self.prepare()
        batch = torch.full((1, 1024, 1024, 3), .94)
        warning = nodes.OCRGuardError("OCR_UNAVAILABLE", "not checked", ["review"])
        with mock.patch.object(nodes, "inspect_background", side_effect=warning):
            with mock.patch.object(Image.Image, "save", side_effect=OSError("disk full")):
                with self.assertLogs(nodes.__name__, level="ERROR"):
                    result, previews = self.render(batch, context, ocr_mode="reject")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["success_count"], 0)
        self.assertIsNone(result["images"][0]["url"])
        self.assertEqual(result["images"][0]["error"]["code"], "OUTPUT_FAILED")
        self.assertEqual(previews, [])
        self.assertEqual(list(self.output_dir.rglob("*.png")), [])

    def test_low_contrast_background_still_returns_exact_font_image_for_review(self):
        _, context, _, _ = self.prepare(text_color="#777777")
        batch = torch.full((1, 1024, 1024, 3), 119 / 255)
        with mock.patch.object(nodes, "inspect_background", side_effect=passed_ocr):
            result, previews = self.render(batch, context, ocr_mode="reject")
            strict, strict_previews = self.render(batch, context, ocr_mode="reject", quality_policy="strict")
        self.assertEqual(result["status"], "review_required")
        self.assertEqual((result["success_count"], result["review_count"], result["failed_count"]), (1, 1, 0))
        self.assertEqual(len(previews), 1)
        item = result["images"][0]
        self.assertIn("LOW_TEXT_CONTRAST", [warning["code"] for warning in item["warnings"]])
        with Image.open(self.image_path(item)) as image:
            manifest = json.loads(image.info["promo"])
            self.assertEqual(manifest["copy_text"], "咖啡还是我的好")
            self.assertEqual(manifest["checks"]["contrast"], "review_required")
            self.assertEqual(manifest["checks"]["background_incidental_text"], "passed_detection")
            self.assertEqual(image.getpixel((1023, 1023)), (119, 119, 119))
            # The outline keeps actual font pixels visible instead of silently
            # returning a blank, same-colored text layer.
            self.assertGreater(np.asarray(image).max(), 119)
        self.assertEqual(strict["status"], "failed")
        self.assertEqual(strict["images"][0]["error"]["code"], "LOW_TEXT_CONTRAST")
        self.assertEqual(strict_previews, [])
        self.assertEqual(len(list(self.output_dir.rglob("*.png"))), 1)

    def test_batch_or_dimensions_mismatch_saves_nothing_and_skips_ocr(self):
        _, context, _, _ = self.prepare(count=3)
        for shape in ((2, 1024, 1024, 3), (3, 1024, 992, 3), (3, 1024, 1024, 1)):
            with self.subTest(shape=shape):
                with mock.patch.object(nodes, "inspect_background") as guard:
                    result, previews = self.render(torch.zeros(shape), context)
                    guard.assert_not_called()
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["success_count"], 0)
                self.assertEqual(len(result["images"]), 3)
                self.assertEqual(previews, [])
                self.assertTrue(all(item["error"]["code"] == "BATCH_SIZE_MISMATCH" for item in result["images"]))
                self.assertEqual(list(self.output_dir.rglob("*.png")), [])

    def test_png_metadata_and_api_copy_are_exact_utf8_without_rewrites(self):
        original = "  咖啡 Coffee 原樣  \r\n完整文案，保持原样。  "
        canvas, context, prompt, _ = self.prepare(original, allow_mixed_scripts=True, title_size=52)
        self.assertNotIn(original, prompt)
        self.assertEqual(context["copy_text"].encode("utf-8"), original.encode("utf-8"))
        # Input canvas contains no pre-rendered copy for Qwen to redraw.
        self.assertTrue(torch.all(canvas == canvas[0, 0, 0]).item())
        batch = torch.full((1, 1024, 1024, 3), .94)
        with mock.patch.object(nodes, "inspect_background", side_effect=passed_ocr):
            result, _ = self.render(batch, context, filename_prefix="metadata/check")
        item = result["images"][0]
        self.assertEqual(item["status"], "success")
        digest = hashlib.sha256(original.encode("utf-8")).hexdigest()
        self.assertEqual(result["copy_text"], original)
        self.assertEqual(item["copy_text"], original)
        self.assertEqual(item["copy_sha256"], digest)
        with Image.open(self.image_path(item)) as image:
            self.assertEqual(image.size, (1024, 1024))
            self.assertEqual(image.format, "PNG")
            metadata = json.loads(image.info["promo"])
            self.assertEqual(metadata["copy_text"], original)
            self.assertEqual(metadata["copy_sha256"], digest)
            self.assertEqual(metadata["language"], "zh-Hans")
            self.assertEqual(metadata["font_rendering"], "real_font_glyphs")
            self.assertNotEqual(np.asarray(image).min(), np.asarray(image).max())

    def test_failed_png_save_does_not_leave_partial_files_or_report_success(self):
        _, context, _, _ = self.prepare()
        batch = torch.full((1, 1024, 1024, 3), .94)
        with mock.patch.object(nodes, "inspect_background", side_effect=passed_ocr):
            with mock.patch.object(Image.Image, "save", side_effect=OSError("simulated disk full")):
                with self.assertLogs(nodes.__name__, level="ERROR"):
                    result, previews = self.render(batch, context)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["images"][0]["error"]["code"], "OUTPUT_FAILED")
        self.assertEqual(result["images"][0]["error"]["error_type"], "OSError")
        self.assertEqual(previews, [])
        self.assertEqual(list(self.output_dir.rglob("*.png")), [])

    def test_missing_unreadable_or_batch_material_never_falls_back_to_text_only(self):
        self.assert_prepare_error("MATERIAL_REQUIRED", material=None)
        for material in (torch.zeros((1, 30, 50, 3)), torch.zeros((2, 128, 128, 3)),
                         torch.full((1, 128, 128, 3), float("nan")), "unreadable.png"):
            with self.subTest(material_type=type(material).__name__):
                self.assert_prepare_error("INVALID_MATERIAL", material=material)

    def test_invalid_target_size_ratio_count_and_language_are_explicit(self):
        self.assert_prepare_error("INVALID_SIZE", width=1000, height=1000)
        self.assert_prepare_error("INVALID_SIZE", width=512, height=512)
        self.assert_prepare_error("INVALID_RATIO", width=1024, height=1056)
        self.assert_prepare_error("INVALID_COUNT", count=5)
        self.assert_prepare_error("LANGUAGE_CONFLICT", text="繁體廣告", language="zh-Hans")
        self.assert_prepare_error("LANGUAGE_CONFLICT", text="简体广告", language="zh-Hant")
        self.assert_prepare_error("LANGUAGE_CONFLICT", text="Coffee 中文", language="en")
        self.assert_prepare_error("COPY_REQUIRED", text=" \r\n ")

    def test_language_selection_uses_matching_font_and_keeps_original_characters(self):
        for text, language, suffix in (("咖啡还是我的好", "zh-Hans", "sc-Regular.otf"),
                                       ("咖啡還是我的好", "zh-Hant", "tc-Regular.otf"),
                                       ("Coffee & Croissant", "en", "sc-Regular.otf")):
            with self.subTest(language=language):
                _, context, _, _ = self.prepare(text, language=language, title_size=48)
                self.assertEqual(context["copy_text"], text)
                self.assertTrue(context["font"].endswith(suffix))

    def test_ocr_unavailable_invalid_generated_pixels_and_unsafe_prefix_fail(self):
        _, context, _, _ = self.prepare()
        good = torch.full((1, 1024, 1024, 3), .94)
        unavailable = nodes.OCRGuardError("OCR_UNAVAILABLE", "models unavailable", ["restore models"])
        with mock.patch.object(nodes, "inspect_background", side_effect=unavailable):
            result, _ = self.render(good, context, ocr_mode="reject", quality_policy="strict")
        self.assertEqual(result["images"][0]["error"]["code"], "OCR_UNAVAILABLE")
        with mock.patch.object(nodes, "inspect_background") as guard:
            for prefix in ("../escape", "/tmp/escape"):
                result, _ = self.render(good, context, ocr_mode="reject", filename_prefix=prefix)
                self.assertEqual(result["images"][0]["error"]["code"], "INVALID_FILENAME")
            bad = good.clone()
            bad[0, 0, 0, 0] = float("nan")
            result, _ = self.render(bad, context, ocr_mode="reject")
            self.assertEqual(result["images"][0]["error"]["code"], "INVALID_GENERATED_IMAGE")
            guard.assert_not_called()
        self.assertEqual(list(self.output_dir.rglob("*.png")), [])

    def test_each_frame_receives_its_own_explicit_ocr_mask(self):
        _, context, _, _ = self.prepare(count=2)
        batch = torch.full((2, 1024, 1024, 3), .94)
        allowed = torch.zeros((2, 1024, 1024))
        allowed[0, 10:20, 30:40] = 1
        allowed[1, 50:70, 80:90] = 1
        with mock.patch.object(nodes, "inspect_background", side_effect=passed_ocr) as guard:
            result, _ = self.render(batch, context, ocr_mode="reject", allowed_text_mask=allowed)
        self.assertEqual(result["success_count"], 2)
        first = guard.call_args_list[0].kwargs["allowed_text_mask"]
        second = guard.call_args_list[1].kwargs["allowed_text_mask"]
        self.assertTrue(first[10, 30])
        self.assertFalse(first[50, 80])
        self.assertTrue(second[50, 80])
        self.assertFalse(second[10, 30])

    @unittest.skipUnless(HAS_OCR_RUNTIME, "Run scripts/setup_ocr.py for real isolated OCR end-to-end")
    def test_real_cpu_ocr_and_font_render_process_four_images_in_one_call(self):
        _, context, _, _ = self.prepare(count=4)
        image = Image.new("RGB", (1024, 1024), "#f7edda")
        draw = ImageDraw.Draw(image)
        draw.ellipse((320, 710, 880, 930), fill="#c6ac8c")
        draw.rounded_rectangle((415, 490, 760, 840), radius=70, fill="#a06b44")
        draw.ellipse((415, 450, 760, 600), fill="#d4b08b", outline="#ead9be", width=10)
        draw.ellipse((440, 475, 735, 575), fill="#65402b")
        # Antialias hard vector edges into a smooth, unlettered product scene.
        image = image.filter(ImageFilter.GaussianBlur(1))
        frame = torch.from_numpy(np.asarray(image).copy()).float().div(255)
        batch = torch.stack([frame] * 4)
        # No detector patch: bundled ONNX CPU detection, Pillow glyph rendering,
        # PNG files and their metadata all run through this single render call.
        result, previews = self.render(batch, context, ocr_mode="reject", filename_prefix="real_cpu")
        self.assertEqual(result["status"], "success", json.dumps(result, ensure_ascii=False))
        self.assertEqual(result["success_count"], 4)
        self.assertEqual(len(previews), 4)
        for item in result["images"]:
            self.assertEqual(item["typography"]["ocr"]["provider"], "CPUExecutionProvider")
            self.assertEqual(item["typography"]["ocr"]["unexpected_count"], 0)
            with Image.open(self.image_path(item)) as output:
                self.assertEqual(json.loads(output.info["promo"])["copy_text"], "咖啡还是我的好")

    @unittest.skipUnless(HAS_OCR_RUNTIME, "Run scripts/setup_ocr.py for real isolated OCR end-to-end")
    def test_detector_flagged_graphic_shape_is_returned_for_review_and_strict_can_reject(self):
        _, context, _, _ = self.prepare()
        # This hard-edged synthetic cup has a small ambiguous detector region.
        # The strict policy may reject it, while the default now preserves the
        # generated image and exact font layer with a truthful review warning.
        image = Image.new("RGB", (1024, 1024), "#f7edda")
        draw = ImageDraw.Draw(image)
        draw.ellipse((320, 710, 880, 930), fill="#c6ac8c")
        draw.rounded_rectangle((415, 490, 760, 840), radius=70, fill="#a06b44")
        draw.ellipse((415, 450, 760, 600), fill="#d4b08b", outline="#ead9be", width=10)
        draw.ellipse((440, 475, 735, 575), fill="#65402b")
        batch = torch.from_numpy(np.asarray(image).copy()).float().div(255).unsqueeze(0)
        result, previews = self.render(batch, context, ocr_mode="reject", quality_policy="strict")
        self.assertEqual(result["status"], "failed")
        error = result["images"][0]["error"]
        self.assertEqual(error["code"], "BACKGROUND_TEXT_DETECTED")
        self.assertGreater(error["details"]["unexpected_count"], 0)
        self.assertEqual(error["details"]["status"], "rejected")
        self.assertEqual(previews, [])
        self.assertEqual(list(self.output_dir.rglob("*.png")), [])
        result, previews = self.render(batch, context, ocr_mode="reject")
        self.assertEqual(result["status"], "review_required")
        self.assertEqual(result["success_count"], 1)
        self.assertEqual(result["passed_count"], 0)
        self.assertEqual(result["review_count"], 1)
        self.assertEqual(len(previews), 1)
        item = result["images"][0]
        self.assertEqual(item["warnings"][0]["code"], "BACKGROUND_TEXT_DETECTED")
        self.assertEqual(item["copy_text"], "咖啡还是我的好")
        self.assertTrue(self.image_path(item).is_file())
        with mock.patch.object(nodes, "inspect_background", side_effect=AssertionError("OCR must stay off")) as guard:
            uninspected, uninspected_previews = self.render(batch, context)
        guard.assert_not_called()
        self.assertEqual(uninspected["status"], "success")
        self.assertEqual(uninspected["success_count"], 1)
        self.assertEqual(uninspected["review_count"], 0)
        self.assertEqual(len(uninspected_previews), 1)
        self.assertEqual(uninspected["images"][0]["typography"]["checks"]["background_incidental_text"], "not_checked")


if __name__ == "__main__":
    unittest.main()
