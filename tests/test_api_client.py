"""Network-stub coverage: one submission per batch, exact copy and clear errors."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from urllib.error import HTTPError


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "call_api.py"
SPEC = importlib.util.spec_from_file_location("promo_call_api", MODULE_PATH)
client = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(client)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class StubHTTP:
    def __init__(self, history=None, queue=None, post_error=None):
        self.calls = []
        self.history = history if history is not None else {}
        self.queue = queue if queue is not None else {"prompt_id": "p-one", "node_errors": {}}
        self.post_error = post_error

    def __call__(self, req, timeout):
        self.calls.append((req.get_method(), req.full_url, req.data, timeout))
        if req.get_method() == "POST" and self.post_error:
            raw = json.dumps(self.post_error).encode("utf-8")
            raise HTTPError(req.full_url, 400, "Bad Request", {}, io.BytesIO(raw))
        payload = self.queue if req.get_method() == "POST" else self.history
        return io.BytesIO(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    @property
    def posts(self):
        return [call for call in self.calls if call[0] == "POST"]


class APIClientTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workflow = Path(self.temp.name) / "workflow.json"
        self.prompt = {
            "copy": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": "旧文案"}},
            "source": {"class_type": "LoadImage", "inputs": {"image": "placeholder.png"}},
            "prepare": {"class_type": "PromoTextPrepare", "inputs": {
                "material": ["source", 0], "copy_text": ["copy", 0], "count": 1}},
            "repeat": {"class_type": "RepeatLatentBatch", "inputs": {"amount": ["prepare", 3]}},
            "sample": {"class_type": "KSampler", "inputs": {"seed": 999}},
            "render": {"class_type": "PromoRenderAndSave", "inputs": {"context": ["prepare", 1]}},
        }
        self.workflow.write_text(json.dumps(self.prompt), encoding="utf-8")
        self.clock = FakeClock()

    def run_request(self, http, **options):
        values = dict(base_url="https://comfy.example.test/", material="input/product.png",
                      copy_text="原样文案", count=1, workflow_path=self.workflow,
                      opener=http, clock=self.clock, sleeper=self.clock.sleep,
                      timeout=2, poll_interval=.25)
        values.update(options)
        return client.request_generation(**values)

    @staticmethod
    def completed(result):
        return {"p-one": {"status": {"completed": True, "status_str": "success"},
                          "outputs": {"render": {"text": [json.dumps(result, ensure_ascii=False)]}}}}

    @staticmethod
    def success(count):
        return {"status": "success", "requested_count": count, "success_count": count,
                "failed_count": 0, "copy_text": "原样文案", "language": "zh-Hans",
                "images": [{"index": index, "status": "success",
                            "url": f"/view?filename=poster_{index}.png&type=output"}
                           for index in range(count)]}

    @classmethod
    def review(cls, count):
        result = cls.success(count)
        result.update(status="review_required", passed_count=0, review_count=count)
        for image in result["images"]:
            image.update(status="review_required", warnings=[{
                "code": "BACKGROUND_TEXT_DETECTED", "message": "底图有疑似文字，图片已返回待复核。",
                "needs_adjustment": ["检查底图"], "details": {"unexpected_count": 1},
            }], checks={"background_incidental_text": "review_required"})
        return result

    def test_counts_one_through_four_are_one_post_each(self):
        for count in range(1, 5):
            with self.subTest(count=count):
                http = StubHTTP(self.completed(self.success(count)))
                result = self.run_request(http, count=count, seed=42)
                self.assertEqual("success", result["status"])
                self.assertEqual(count, result["success_count"])
                self.assertEqual(1, len(http.posts))
                submitted = json.loads(http.posts[0][2])["prompt"]
                self.assertEqual(count, submitted["prepare"]["inputs"]["count"])
                self.assertEqual(["prepare", 3], submitted["repeat"]["inputs"]["amount"])
                self.assertEqual(42, submitted["sample"]["inputs"]["seed"])
                self.assertEqual("input/product.png", submitted["source"]["inputs"]["image"])
                self.assertEqual("return_with_warnings", submitted["render"]["inputs"]["quality_policy"])
                self.assertEqual("off", submitted["render"]["inputs"]["ocr_mode"])
                self.assertEqual("campaign", submitted["prepare"]["inputs"]["typography_mode"])
                self.assertEqual(count, len(result["images"]))
                self.assertTrue(all(image["url"].startswith("https://comfy.example.test/view?")
                                    for image in result["images"]))

    def test_manual_typography_is_explicit_and_invalid_mode_never_submits(self):
        http = StubHTTP(self.completed(self.success(1)))
        self.assertEqual("success", self.run_request(http, typography_mode="manual")["status"])
        submitted = json.loads(http.posts[0][2])["prompt"]
        self.assertEqual("manual", submitted["prepare"]["inputs"]["typography_mode"])
        invalid = StubHTTP()
        result = self.run_request(invalid, typography_mode="invented")
        self.assertEqual("INVALID_TYPOGRAPHY_MODE", result["error"]["code"])
        self.assertEqual([], invalid.posts)

    def test_separate_body_preserves_raw_fields_and_old_copy_stays_compatible(self):
        self.prompt["body"] = {"class_type": "PrimitiveStringMultiline", "inputs": {"value": "placeholder"}}
        self.prompt["prepare"]["inputs"]["body_text"] = ["body", 0]
        self.workflow.write_text(json.dumps(self.prompt), encoding="utf-8")
        title, body = "  原樣 Title  \r\n第二行", " Body\r\n繁體 EN  "
        http = StubHTTP(self.completed(self.success(1)))
        self.run_request(http, copy_text=title, body_text=body)
        submitted = json.loads(http.posts[0][2])["prompt"]
        self.assertEqual(title, submitted["copy"]["inputs"]["value"])
        self.assertEqual(body, submitted["body"]["inputs"]["value"])
        self.assertEqual(["body", 0], submitted["prepare"]["inputs"]["body_text"])
        old = StubHTTP(self.completed(self.success(1)))
        self.run_request(old, copy_text=title + "\n" + body)
        legacy = json.loads(old.posts[0][2])["prompt"]
        self.assertNotIn("body_text", legacy["prepare"]["inputs"])
        self.assertEqual(title + "\n" + body, legacy["copy"]["inputs"]["value"])

    def test_review_images_warnings_and_counts_are_returned_without_resubmission(self):
        for count in range(1, 5):
            with self.subTest(count=count):
                payload = self.review(count)
                http = StubHTTP(self.completed(payload))
                result = self.run_request(http, count=count)
                self.assertEqual("review_required", result["status"])
                self.assertEqual(count, result["success_count"])
                self.assertEqual(count, result["review_count"])
                self.assertEqual(0, result["passed_count"])
                self.assertEqual(0, result["failed_count"])
                self.assertEqual(1, len(http.posts))
                for original, returned in zip(payload["images"], result["images"]):
                    self.assertEqual(original["warnings"], returned["warnings"])
                    self.assertEqual(original["checks"], returned["checks"])
                    self.assertEqual("review_required", returned["status"])
                    self.assertTrue(returned["url"].startswith("https://comfy.example.test/view?"))

    def test_explicit_strict_policy_is_sent_to_render_node(self):
        http = StubHTTP(self.completed(self.success(1)))
        result = self.run_request(http, quality_policy="strict")
        self.assertEqual("success", result["status"])
        submitted = json.loads(http.posts[0][2])["prompt"]
        self.assertEqual("strict", submitted["render"]["inputs"]["quality_policy"])
        self.assertEqual("off", submitted["render"]["inputs"]["ocr_mode"])
        self.assertEqual(1, len(http.posts))

    def test_default_off_overrides_legacy_template_and_explicit_ocr_is_preserved(self):
        self.prompt["render"]["inputs"]["ocr_mode"] = "reject"
        self.workflow.write_text(json.dumps(self.prompt), encoding="utf-8")
        for supplied, expected in (({}, "off"), ({"ocr_mode": "reject"}, "reject"),
                                   ({"ocr_mode": "repair_masked"}, "repair_masked")):
            with self.subTest(mode=expected):
                http = StubHTTP(self.completed(self.success(1)))
                result = self.run_request(http, **supplied)
                self.assertEqual("success", result["status"])
                submitted = json.loads(http.posts[0][2])["prompt"]
                self.assertEqual(expected, submitted["render"]["inputs"]["ocr_mode"])
                self.assertEqual(1, len(http.posts))

    def test_invalid_ocr_mode_is_rejected_without_submitting(self):
        http = StubHTTP()
        result = self.run_request(http, ocr_mode="auto_accept")
        self.assertEqual("INVALID_OCR_MODE", result["error"]["code"])
        self.assertEqual([], http.calls)

    def test_invalid_quality_policy_is_rejected_without_submitting(self):
        http = StubHTTP()
        result = self.run_request(http, quality_policy="silently_pass")
        self.assertEqual("INVALID_QUALITY_POLICY", result["error"]["code"])
        self.assertEqual([], http.calls)

    def test_utf8_copy_file_and_post_keep_crlf_and_mixed_scripts(self):
        original = "優惠 Offer\r\n咖啡 Coffee 50%\r\n"
        path = Path(self.temp.name) / "copy.txt"
        path.write_bytes(original.encode("utf-8"))
        text = client.read_copy_file(path)
        self.assertEqual(original, text)
        http = StubHTTP(self.completed(self.success(1)))
        self.run_request(http, copy_text=text, language="zh-Hant", allow_mixed_scripts=True)
        submitted = json.loads(http.posts[0][2])["prompt"]
        self.assertEqual(original, submitted["copy"]["inputs"]["value"])
        self.assertTrue(submitted["prepare"]["inputs"]["allow_mixed_scripts"])
        self.assertEqual("zh-Hant", submitted["prepare"]["inputs"]["language"])
        self.assertEqual(["copy", 0], submitted["prepare"]["inputs"]["copy_text"])

    def test_direct_copy_input_is_overridden_verbatim(self):
        self.prompt["prepare"]["inputs"]["copy_text"] = "旧文案"
        self.workflow.write_text(json.dumps(self.prompt), encoding="utf-8")
        http = StubHTTP(self.completed(self.success(1)))
        self.run_request(http, copy_text="  指定原文\n")
        self.assertEqual("  指定原文\n", json.loads(http.posts[0][2])["prompt"]["prepare"]["inputs"]["copy_text"])

    def test_partial_status_and_failure_item_are_preserved(self):
        payload = self.success(2)
        payload.update(status="partial_success", success_count=1, failed_count=1)
        payload["images"][1] = {"index": 1, "status": "failed", "url": None,
                                "error": {"code": "INCIDENTAL_TEXT", "message": "底图含文字"}}
        http = StubHTTP(self.completed(payload))
        result = self.run_request(http, count=2)
        self.assertEqual("partial_success", result["status"])
        self.assertEqual(payload["images"][1], result["images"][1])
        self.assertEqual(1, len(http.posts))

    def test_queue_validation_failure_is_structured_and_not_resubmitted(self):
        http = StubHTTP(post_error={"error": {"message": "Prompt outputs failed validation"},
                                   "node_errors": {"source": {"class_type": "LoadImage", "errors": [
                                       {"message": "Invalid image file", "details": "missing.png"}]}}})
        result = self.run_request(http, count=3)
        self.assertEqual("failed", result["status"])
        self.assertEqual("COMFY_QUEUE_REJECTED", result["error"]["code"])
        self.assertIn("missing.png", result["error"]["message"])
        self.assertEqual(3, result["failed_count"])
        self.assertEqual(1, len(http.calls))
        self.assertEqual(1, len(http.posts))

    def test_upstream_execution_error_is_clear_and_not_resubmitted(self):
        reason = {"code": "TEXT_TOO_LONG", "message": "全文放不下", "needs_adjustment": ["copy_text", "width"]}
        http = StubHTTP({"p-one": {"status": {"completed": False, "status_str": "error", "messages": [
            ["execution_error", {"node_id": "prepare", "node_type": "PromoTextPrepare",
                                 "exception_message": json.dumps(reason, ensure_ascii=False)}]]}}})
        result = self.run_request(http, count=4)
        self.assertEqual("COMFY_EXECUTION_ERROR", result["error"]["code"])
        self.assertIn("全文放不下", result["error"]["message"])
        self.assertEqual(["copy_text", "width"], result["error"]["needs_adjustment"])
        self.assertEqual("TEXT_TOO_LONG", result["error"]["details"]["upstream_error"]["code"])
        self.assertEqual(1, len(http.posts))
        self.assertEqual(2, len(http.calls))

    def test_timeout_does_not_resubmit_or_claim_server_cancellation(self):
        http = StubHTTP({})
        result = self.run_request(http, count=2, timeout=.5)
        self.assertEqual("TIMEOUT", result["error"]["code"])
        self.assertEqual("p-one", result["prompt_id"])
        self.assertEqual(1, len(http.posts))
        self.assertIn("仍在服务器运行", result["error"]["message"])
        self.assertTrue(all(call[3] > 0 for call in http.calls))

    def test_embedded_credentials_are_rejected_without_network_or_leak(self):
        http = StubHTTP()
        result = self.run_request(http, base_url="https://secret-user:secret-pass@example.test")
        self.assertEqual("INVALID_BASE_URL", result["error"]["code"])
        self.assertEqual([], http.calls)
        self.assertNotIn("secret", json.dumps(result))

    def test_reverse_proxy_prefix_and_absolute_urls(self):
        payload = self.success(1)
        http = StubHTTP(self.completed(payload))
        result = self.run_request(http, base_url="https://example.test/comfy/")
        self.assertEqual("https://example.test/comfy/prompt", http.posts[0][1])
        self.assertTrue(result["images"][0]["url"].startswith("https://example.test/comfy/view?"))

    def test_completed_without_result_reports_failure(self):
        http = StubHTTP({"p-one": {"status": {"completed": True}, "outputs": {}}})
        result = self.run_request(http)
        self.assertEqual("RESULT_MISSING", result["error"]["code"])
        self.assertEqual(1, len(http.posts))

    def test_cli_preserves_copy_and_saves_only_returned_json(self):
        copy_path = Path(self.temp.name) / "copy.txt"
        original = "原文\r\n優惠 Offer\r\n"
        copy_path.write_bytes(original.encode("utf-8"))
        output = Path(self.temp.name) / "result.json"
        stdout = io.StringIO()
        with mock.patch.object(client, "request_generation", return_value=self.success(2)) as generate:
            with contextlib.redirect_stdout(stdout):
                status = client.main(["--base-url", "https://example.test", "--material", "uploaded.png",
                                      "--copy-file", str(copy_path), "--count", "2", "--output", str(output)])
        self.assertEqual(0, status)
        self.assertEqual(original, generate.call_args.kwargs["copy_text"])
        self.assertEqual(2, generate.call_args.kwargs["count"])
        self.assertEqual("return_with_warnings", generate.call_args.kwargs["quality_policy"])
        self.assertEqual("off", generate.call_args.kwargs["ocr_mode"])
        self.assertEqual(json.loads(stdout.getvalue()), json.loads(output.read_text(encoding="utf-8")))

    def test_cli_returned_review_images_are_successful_delivery_without_hiding_warnings(self):
        copy_path = Path(self.temp.name) / "copy.txt"
        copy_path.write_text("原样文案", encoding="utf-8")
        payload = self.review(2)
        stdout = io.StringIO()
        with mock.patch.object(client, "request_generation", return_value=payload) as generate:
            with contextlib.redirect_stdout(stdout):
                status = client.main(["--base-url", "https://example.test", "--material", "uploaded.png",
                                      "--copy-file", str(copy_path), "--count", "2"])
        self.assertEqual(0, status)
        self.assertEqual(payload, json.loads(stdout.getvalue()))
        self.assertEqual("return_with_warnings", generate.call_args.kwargs["quality_policy"])

    def test_cli_strict_option_is_forwarded_and_real_partial_failure_still_exits_one(self):
        copy_path = Path(self.temp.name) / "copy.txt"
        copy_path.write_text("原样文案", encoding="utf-8")
        payload = self.review(2)
        payload.update(status="partial_success", success_count=1, failed_count=1, review_count=1)
        payload["images"][1] = {"index": 1, "status": "failed", "url": None,
                                "error": {"code": "OUTPUT_FAILED", "message": "保存失败"}}
        with mock.patch.object(client, "request_generation", return_value=payload) as generate:
            with contextlib.redirect_stdout(io.StringIO()):
                status = client.main(["--base-url", "https://example.test", "--material", "uploaded.png",
                                      "--copy-file", str(copy_path), "--count", "2", "--quality-policy", "strict",
                                      "--ocr-mode", "reject"])
        self.assertEqual(1, status)
        self.assertEqual("strict", generate.call_args.kwargs["quality_policy"])
        self.assertEqual("reject", generate.call_args.kwargs["ocr_mode"])


if __name__ == "__main__":
    unittest.main()
