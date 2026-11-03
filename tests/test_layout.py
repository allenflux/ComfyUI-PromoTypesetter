import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("promo_layout_test_module", ROOT / "promo_typesetter" / "layout.py")
layout = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(layout)
FONT = ROOT / "fonts" / "NotoSansCJKsc-Regular.otf"
if not FONT.is_file():
    FONT = ROOT.parent / "qwen-promo-font-v3" / "custom_nodes" / "exact_promo_text" / "fonts" / FONT.name


class TransparentLayoutTests(unittest.TestCase):
    def test_contrast_luminance_is_finite_and_matches_reference_colors(self):
        colors = np.array([[0, 0, 0], [255, 255, 255], [255, 0, 0],
                           [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
        with np.errstate(all="raise"):
            actual = layout._linear_luminance(colors)
            raster = layout._linear_luminance(np.tile(colors, (32, 32, 1)))
        np.testing.assert_allclose(actual, [0, 1, .2126, .7152, .0722], atol=1e-12)
        self.assertTrue(np.isfinite(raster).all())
        self.assertTrue(((raster >= 0) & (raster <= 1)).all())

    def make(self, text="咖啡还是我的好", **kwargs):
        return layout.prepare(text, kwargs.pop("design_prompt", "柔和自然光，产品位于下半部。"),
                              kwargs.pop("font_path", FONT), kwargs.pop("width", 1088),
                              kwargs.pop("height", 1440), **kwargs)

    def error(self, code, callback):
        with self.assertRaises(layout.PromoError) as raised:
            callback()
        self.assertEqual(raised.exception.code, code)
        self.assertEqual(json.loads(str(raised.exception))["code"], code)

    def test_actual_copy_is_only_title_and_never_sent_to_model(self):
        context = self.make()
        self.assertEqual(context["copy_text"], "咖啡还是我的好")
        self.assertNotIn(context["copy_text"], context["prompt"])
        layer, manifest = layout.render_text_layer(context, Image.new("RGB", (1088, 1440), "#fff8ed"))
        self.assertEqual(manifest["copy_text"], context["copy_text"])
        self.assertEqual(manifest["title_lines"], ["咖啡还是我的好"])
        self.assertEqual(manifest["body_lines"], [])
        self.assertEqual(layer.mode, "RGBA")

    def test_no_opaque_footer_or_background_pixels(self):
        context = self.make(layout="top")
        layer, manifest = layout.render_text_layer(context, Image.new("RGB", (1088, 1440), "#fff8ed"))
        alpha = np.asarray(layer)[:, :, 3]
        self.assertEqual(int(alpha[-416:, :].max()), 0)
        self.assertLess(np.count_nonzero(alpha), alpha.size * .12)
        self.assertEqual(int(alpha[:, :50].max()), 0)
        self.assertEqual(manifest["checks"]["subject_protection"], "not_provided")

    def test_utf8_spaces_newlines_and_fullwidth_forms_are_unchanged(self):
        text = "  原樣標題  \r\n Body  EN：１２３ \r\n\r\n末尾  "
        context = self.make(text, language="mixed", title_size=44, body_size=24)
        layer, manifest = layout.render_text_layer(context, Image.new("RGB", (1088, 1440), "#fff8ed"))
        self.assertEqual(manifest["copy_text"].encode(), text.encode())
        self.assertEqual("".join(manifest["title_lines"]), "  原樣標題  ")
        self.assertEqual(manifest["body_lines"], [" Body  EN：１２３ ", "", "末尾  "])
        self.assertIsNotNone(layer.getbbox())

    def test_chinese_traditional_english_and_mixed_real_fonts(self):
        cases = [("简体中文，保持原样。", "zh-Hans"), ("繁體中文，保持原樣。", "zh-Hant"),
                 ("Coffee & Croissant 2026!", "en"), ("简體 Mixed EN，原樣保留。", "mixed")]
        for text, language in cases:
            with self.subTest(language=language):
                context = self.make(text, language=language, title_size=64)
                _, manifest = layout.render_text_layer(context, Image.new("RGB", (1088, 1440), "#fff8ed"))
                self.assertEqual(manifest["copy_text"], text)
                self.assertEqual("".join(manifest["title_lines"]), text)

    def test_language_conflicts_are_rejected_without_conversion(self):
        self.error("LANGUAGE_CONFLICT", lambda: self.make("繁體廣告", language="zh-Hans"))
        self.error("LANGUAGE_CONFLICT", lambda: self.make("简体广告", language="zh-Hant"))
        self.error("LANGUAGE_CONFLICT", lambda: self.make("Coffee 中文", language="en"))
        context = self.make("繁體 Coffee", language="zh-Hans", allow_mixed_scripts=True)
        self.assertEqual(context["copy_text"], "繁體 Coffee")

    def test_unicode_graphemes_are_never_split(self):
        text = "Cafe\u0301 crème"
        tokens = layout._tokens(text)
        self.assertEqual("".join(tokens), text)
        self.assertIn("Cafe\u0301", tokens)
        context = self.make(text, language="en", first_line_title=False)
        _, metadata = layout.render_text_layer(context, Image.new("RGB", (1088, 1440), "#fff8ed"))
        self.assertEqual(metadata["copy_text"], text)

    def test_missing_copy_font_and_glyph_fail_before_sampling(self):
        for value in ("", "  \r\n ", None):
            self.error("COPY_REQUIRED", lambda value=value: self.make(value))
        self.error("FONT_UNREADABLE", lambda: self.make(font_path=ROOT / "missing.otf"))
        self.error("FONT_MISSING_GLYPHS", lambda: self.make("字\U0010ffff"))

    def test_controls_are_explicit_failures(self):
        for value in ("文\t案", "文\u200d案", "文\x00案"):
            self.error("UNSUPPORTED_TEXT_CONTROL", lambda value=value: self.make(value))

    def test_overflow_never_shrinks_or_rewrites_copy(self):
        text = "完整文案保持原样。" * 300
        self.error("TEXT_TOO_LONG", lambda: self.make(text))
        self.error("TEXT_TOO_LONG", lambda: self.make("W" * 300, language="en"))

    def test_requested_count_and_exact_final_dimensions(self):
        for count in (1, 2, 3, 4):
            context = self.make(width=1080, height=1437, count=count)
            layer, metadata = layout.render_text_layer(context, Image.new("RGB", (1080, 1437), "#fff8ed"))
            self.assertEqual(layer.size, (1080, 1437))
            self.assertEqual(context["generation_canvas"].size, (1088, 1440))
            self.assertEqual(metadata["count"], count)
        for count in (0, 5, True, 2.0):
            self.error("INVALID_COUNT", lambda count=count: self.make(count=count))

    def test_protected_subject_uses_other_region_and_no_glyph_overlap(self):
        mask = np.zeros((1440, 1088), dtype=np.uint8)
        mask[:550] = 255
        context = self.make(protected_mask=mask)
        layer, manifest = layout.render_text_layer(context, Image.new("RGB", (1088, 1440), "#fff8ed"))
        self.assertEqual(manifest["layout"], "bottom")
        self.assertEqual(np.asarray(layer)[:, :, 3][mask > 0].sum(), 0)
        self.assertEqual(manifest["checks"]["subject_protection"], "passed")
        self.error("PROTECTED_SUBJECT_OVERLAP", lambda: self.make(layout="top", protected_mask=mask))

    def test_full_subject_mask_refuses_overlay(self):
        mask = np.ones((1440, 1088), dtype=np.float32)
        self.error("PROTECTED_SUBJECT_OVERLAP", lambda: self.make(protected_mask=mask))

    def test_auto_layout_prefers_flat_region_over_busy_region(self):
        background = np.full((1440, 1088, 3), [255, 248, 237], dtype=np.uint8)
        rng = np.random.default_rng(3)
        background[:600] = rng.integers(0, 256, size=background[:600].shape, dtype=np.uint8)
        context = self.make()
        _, manifest = layout.render_text_layer(context, Image.fromarray(background))
        self.assertEqual(manifest["layout"], "bottom")

    def test_explicit_color_low_contrast_and_size_errors(self):
        context = self.make(text_color="#fff8ed")
        self.error("LOW_TEXT_CONTRAST", lambda: layout.render_text_layer(
            context, Image.new("RGB", (1088, 1440), "#fff8ed")))
        self.error("INVALID_BACKGROUND_SIZE", lambda: layout.render_text_layer(
            self.make(), Image.new("RGB", (100, 100))))

    def test_readable_region_takes_priority_over_low_contrast_fallback(self):
        context = self.make(width=1024, height=1024, title_size=72,
                            body_size=28, text_color="#777777")
        background = Image.new("RGB", (1024, 1024), "white")
        background.paste("#777777", (0, 0, 1024, 351))
        # The preferred top region is flat but matches the text color. A clear
        # bottom region must be used before adding an outline to the top.
        layer, manifest = layout.render_text_layer(context, background, allow_low_contrast=True)
        self.assertEqual(manifest["layout"], "bottom")
        self.assertEqual(manifest["checks"]["contrast"], "passed")
        self.assertGreaterEqual(manifest["contrast_10th_percentile"], 3.0)
        self.assertEqual(int(np.asarray(layer)[:351, :, 3].max()), 0)

    def test_style_conflict_and_copy_in_visual_instructions(self):
        self.error("STYLE_CONFLICT", lambda: self.make(style="现代简约", design_prompt="采用科技未来风格"))
        self.error("COPY_IN_DESIGN_PROMPT", lambda: self.make(design_prompt='写上“另外的标题”'))
        self.make(style="现代简约", design_prompt="不要古风典雅，采用干净构图")
        self.make(style="现代简约", design_prompt="温暖自然材质与柔和光线")
        self.assertEqual(layout.resolve_style("auto", "采用古风典雅，产品在下方"), "古风典雅")
        self.assertEqual(layout.resolve_style("auto", "without a futuristic style; soft light"), "auto")
        context = self.make(style="auto", design_prompt="采用科技未来，主体在下方")
        self.assertEqual(context["style"], "科技未来")
        self.assertEqual(context["requested_style"], "auto")

    def test_all_requested_styles_have_visual_and_typographic_effects(self):
        canvases, palettes, advances = set(), set(), set()
        for style in ("古风典雅", "现代简约", "时尚潮流", "悬疑暗调", "科技未来"):
            context = self.make(style=style)
            layer, manifest = layout.render_text_layer(context, context["generation_canvas"])
            self.assertEqual(manifest["style"], style)
            self.assertIsNotNone(layer.getbbox())
            canvases.add(context["generation_canvas"].getpixel((0, 0)))
            palettes.add(context["foreground_palette"])
            advances.add(context["candidate_plans"][0]["lines"][0]["advance"])
        self.assertEqual(len(canvases), 5)
        self.assertEqual(len(palettes), 5)
        self.assertGreater(len(advances), 2)

    def test_malformed_inputs_are_actionable_structured_errors(self):
        self.error("INVALID_STYLE", lambda: self.make(style=[]))
        self.error("INVALID_TEXT_COLOR", lambda: self.make(text_color=None))
        self.error("FONT_UNREADABLE", lambda: self.make(font_path=None))
        self.error("INVALID_PROTECTED_MASK", lambda: self.make(protected_mask="not a mask"))
        self.error("INVALID_PROTECTED_MASK", lambda: self.make(protected_mask=np.full((1440, 1088), np.nan)))


if __name__ == "__main__":
    unittest.main()
