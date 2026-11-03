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
    def test_campaign_separate_title_and_body_have_large_hierarchy_and_exact_glyphs(self):
        title, body = "咖啡还是我的好", "okok 哈喽哈喽"
        context = self.make(title, width=1152, height=1536, title_size=112, body_size=40,
                            typography_mode="campaign", title_text=title, body_text=body)
        layer, manifest = layout.render_text_layer(context, Image.new("RGB", (1152, 1536), "#fff8ed"))
        self.assertEqual(manifest["title_lines"], ["咖啡还是", "我的好"])
        self.assertEqual(manifest["body_lines"], [body])
        self.assertEqual(manifest["copy_text"], title + "\n" + body)
        self.assertEqual(manifest["title_text"], title)
        self.assertEqual(manifest["body_text"], body)
        title_lines = [line for line in manifest["line_styles"] if line["role"] == "title"]
        body_lines = [line for line in manifest["line_styles"] if line["role"] == "body"]
        self.assertTrue(all(line["font_size"] >= 180 for line in title_lines))
        self.assertEqual([line["font_size"] for line in body_lines], [40])
        self.assertGreater(np.count_nonzero(np.asarray(layer)[:, :, 3]), 1000)
        plan = next(p for p in context["candidate_plans"] if p["name"] == manifest["layout"])
        last_title = [line for line in plan["lines"] if line["role"] == "title"][-1]
        first_body = next(line for line in plan["lines"] if line["role"] == "body")
        self.assertGreater(first_body["y"], last_title["y"] + last_title["height"])
        self.assertNotIn(title, context["prompt"])
        self.assertNotIn(body, context["prompt"])

    def test_campaign_legacy_combined_copy_still_expands_short_title_with_body(self):
        text = "咖啡还是我的好\r\nokok 哈喽哈喽"
        context = self.make(text, width=1152, height=1536, title_size=112, body_size=40,
                            typography_mode="campaign")
        _, manifest = layout.render_text_layer(context, Image.new("RGB", (1152, 1536), "#fff8ed"))
        self.assertEqual(manifest["copy_text"].encode(), text.encode())
        self.assertEqual(manifest["title_lines"], ["咖啡还是", "我的好"])
        self.assertEqual(manifest["body_lines"], ["okok 哈喽哈喽"])
        self.assertGreaterEqual(manifest["line_styles"][0]["font_size"], 180)

    def test_separate_multiline_title_and_body_keep_their_own_newlines(self):
        title = "咖啡还是\r\n我的好"
        body = "  原樣正文  \r\n Mixed EN：１２３ \r\n\r\n末尾  "
        context = self.make(title, width=1152, height=1536, title_text=title, body_text=body,
                            typography_mode="campaign", title_size=112, body_size=32, language="mixed")
        _, manifest = layout.render_text_layer(context, Image.new("RGB", (1152, 1536), "#fff8ed"))
        self.assertEqual(manifest["title_text"].encode(), title.encode())
        self.assertEqual(manifest["body_text"].encode(), body.encode())
        self.assertEqual(manifest["copy_text"], title + "\n" + body)
        self.assertEqual(manifest["title_lines"], ["咖啡还是", "我的好"])
        self.assertEqual(manifest["body_lines"], ["  原樣正文  ", " Mixed EN：１２３ ", "", "末尾  "])
        self.assertEqual([line["role"] for line in manifest["line_styles"][:2]], ["title", "title"])
        self.assertTrue(all(line["role"] == "body" for line in manifest["line_styles"][2:]))
        self.assertTrue(all(line["font_size"] >= 112 for line in manifest["line_styles"][:2]))

    def test_explicit_text_fields_work_in_manual_mode_and_empty_body_adds_no_copy(self):
        title = "第一行\r\n第二行"
        context = self.make(title, title_text=title, body_text="", title_size=72,
                            first_line_title=False, typography_mode="manual")
        _, manifest = layout.render_text_layer(context, Image.new("RGB", (1088, 1440), "#fff8ed"))
        self.assertEqual(manifest["title_lines"], ["第一行", "第二行"])
        self.assertEqual(manifest["body_lines"], [])
        self.assertEqual(manifest["copy_text"], title)
        self.assertEqual(manifest["body_text"], "")

    def test_campaign_does_not_reduce_requested_font_sizes_to_fit_body(self):
        context = self.make("咖啡还是我的好", width=1152, height=1536,
                            typography_mode="campaign", title_text="咖啡还是我的好",
                            body_text="正文保持完整。" * 8, title_size=111, body_size=40)
        for plan in context["candidate_plans"]:
            self.assertTrue(all(line.get("font_size", 111) >= 111
                                for line in plan["lines"] if line["role"] == "title"))
            self.assertTrue(all(line.get("font_size", 40) == 40
                                for line in plan["lines"] if line["role"] == "body"))
        self.error("TEXT_TOO_LONG", lambda: self.make("咖啡还是我的好", typography_mode="campaign",
                   title_text="咖啡还是我的好", body_text="全部正文不得删减。" * 1000,
                   title_size=112, body_size=40))
        self.error("COPY_REQUIRED", lambda: self.make("", title_text="", body_text="正文不是标题"))
        self.error("INVALID_BODY_TEXT", lambda: self.make(body_text=123))

    def test_campaign_short_title_is_large_balanced_and_exact(self):
        context = self.make(width=1152, height=1536, title_size=112,
                            typography_mode="campaign")
        layer, manifest = layout.render_text_layer(context, Image.new("RGB", (1152, 1536), "#fff8ed"))
        self.assertEqual(manifest["typography_mode"], "campaign")
        self.assertEqual(manifest["title_lines"], ["咖啡还是", "我的好"])
        self.assertEqual("".join(manifest["title_lines"]), context["copy_text"])
        self.assertGreaterEqual(manifest["line_styles"][0]["font_size"], 180)
        self.assertGreater(layer.getbbox()[3] - layer.getbbox()[1], 350)
        self.assertLess(layer.getbbox()[3], 1536 * .43)
        self.assertEqual(len(manifest["line_styles"]), 2)
        self.assertNotEqual(manifest["line_styles"][0]["color"], manifest["line_styles"][1]["color"])
        self.assertNotIn(context["copy_text"], context["prompt"])
        self.assertIn("无字色带", context["prompt"])
        self.assertNotIn("画面仅包含实际主体", context["prompt"])

    def test_campaign_guide_has_real_style_color_and_geometry(self):
        guides = []
        for style in ("古风典雅", "现代简约", "时尚潮流", "悬疑暗调", "科技未来"):
            context = self.make(width=1152, height=1536, typography_mode="campaign", style=style)
            guide = np.asarray(context["generation_canvas"])
            self.assertEqual(guide.shape, (1536, 1152, 3))
            self.assertGreater(np.std(guide.reshape(-1, 3), axis=0).max(), 2)
            guides.append(guide.tobytes())
        self.assertEqual(len(set(guides)), 5)

    def test_campaign_keeps_the_planned_readable_composition(self):
        context = self.make(width=1152, height=1536, typography_mode="campaign")
        background = np.full((1536, 1152, 3), 249, dtype=np.uint8)
        # Texture alone must not relocate a readable art-directed title.
        background[:700:2] = 236
        _, manifest = layout.render_text_layer(context, Image.fromarray(background))
        self.assertEqual(manifest["layout"], context["reserved_layout"])
        self.assertEqual(manifest["layout"], "top")
        self.assertEqual(manifest["checks"]["contrast"], "passed")

    def test_campaign_still_respects_protection_and_readability(self):
        mask = np.zeros((1536, 1152), dtype=np.uint8)
        mask[:730] = 255
        context = self.make(width=1152, height=1536, typography_mode="campaign", protected_mask=mask)
        layer, manifest = layout.render_text_layer(context, Image.new("RGB", (1152, 1536), "#fff8ed"))
        self.assertEqual(manifest["layout"], "bottom")
        self.assertEqual(np.asarray(layer)[:, :, 3][mask > 0].sum(), 0)
        context = self.make(width=1152, height=1536, typography_mode="campaign", text_color="#777777")
        background = Image.new("RGB", (1152, 1536), "white")
        background.paste("#777777", (0, 0, 1152, 730))
        _, manifest = layout.render_text_layer(context, background, allow_low_contrast=True)
        self.assertEqual(manifest["layout"], "bottom")
        self.assertEqual({line["color"] for line in manifest["line_styles"]}, {"#777777"})

    def test_campaign_preserves_explicit_lines_and_words(self):
        text = "完整标题\r\n Body  EN：１２３ \r\n\r\n末尾  "
        context = self.make(text, typography_mode="campaign", language="mixed", title_size=44, body_size=24)
        _, manifest = layout.render_text_layer(context, Image.new("RGB", (1088, 1440), "#fff8ed"))
        self.assertEqual(manifest["copy_text"].encode(), text.encode())
        self.assertEqual(manifest["title_lines"], ["完整标题"])
        self.assertEqual(manifest["body_lines"], [" Body  EN：１２３ ", "", "末尾  "])
        context = self.make("Fresh Coffee & Croissant", typography_mode="campaign", language="en")
        _, manifest = layout.render_text_layer(context, Image.new("RGB", (1088, 1440), "#fff8ed"))
        self.assertEqual("".join(manifest["title_lines"]), "Fresh Coffee & Croissant")
        for word in ("Fresh", "Coffee", "Croissant"):
            self.assertTrue(any(word in line for line in manifest["title_lines"]))

    def test_campaign_accent_is_internal_and_explicit_text_color_wins(self):
        context = self.make(width=1152, height=1536, typography_mode="campaign",
                            campaign_accent=(166, 65, 28), text_color="#19232e")
        _, manifest = layout.render_text_layer(context, Image.new("RGB", (1152, 1536), "#fff8ed"))
        self.assertEqual(manifest["accent_color"], "#a6411c")
        self.assertEqual({line["color"] for line in manifest["line_styles"]}, {"#19232e"})
        self.error("INVALID_TYPOGRAPHY_MODE", lambda: self.make(typography_mode="unknown"))
        self.error("INVALID_ACCENT", lambda: self.make(campaign_accent=(1, 2, 500)))

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
