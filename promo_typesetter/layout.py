"""Exact-copy typography on transparent layers, independent of any image model.

Only caller-supplied characters are drawn. Layout adds soft line breaks but
never trims, normalizes, translates, converts scripts, or shortens the copy.
The sampling canvas and model prompt deliberately contain no rendered copy.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path

import numpy as np
import regex
from fontTools.ttLib import TTFont
from PIL import Image, ImageColor, ImageDraw, ImageFont


class PromoError(ValueError):
    """User-adjustable validation failure, safe to serialize in API results."""

    def __init__(self, code, message, needs_adjustment=None, details=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.needs_adjustment = list(needs_adjustment or [])
        self.details = details or {}

    def as_dict(self):
        result = {"code": self.code, "message": self.message,
                  "needs_adjustment": self.needs_adjustment}
        if self.details:
            result["details"] = self.details
        return result

    def __str__(self):
        return json.dumps(self.as_dict(), ensure_ascii=False)


def _fail(code, message, adjustments=None, **details):
    raise PromoError(code, message, adjustments, details)


_STYLES = {
    "auto": {"canvas": "#fff8ed", "direction": "依据视觉要求、参考图片与素材内容选择匹配的视觉风格、配色、光线与材质",
             "foreground": ((43, 27, 18), (255, 250, 241)), "spacing": .15},
    "古风典雅": {"canvas": "#f0e4ce", "direction": "宣纸暖色、墨色层次、柔和侧光、含蓄雅致的自然材质",
                "foreground": ((54, 38, 28), (255, 244, 219)), "spacing": .22},
    "现代简约": {"canvas": "#f5f5f1", "direction": "黑白灰与克制的主体色、简洁几何构图、干净柔光与真实材质",
                "foreground": ((24, 25, 27), (252, 252, 248)), "spacing": .10},
    "时尚潮流": {"canvas": "#f1d9eb", "direction": "高辨识度撞色、现代时尚摄影、清晰轮廓与富有活力的斜向构图",
                "foreground": ((47, 13, 61), (255, 249, 235)), "spacing": .18},
    "悬疑暗调": {"canvas": "#191b25", "direction": "低照度深蓝灰色调、电影感明暗、局部侧光与清晰可辨的主体轮廓",
                "foreground": ((230, 223, 204), (22, 25, 34)), "spacing": .20},
    "科技未来": {"canvas": "#111e31", "direction": "冷蓝与青色光、精确几何结构、玻璃金属质感与未来感空间",
                "foreground": ((211, 249, 255), (9, 30, 49)), "spacing": .14},
}
_STYLE_ALIASES = {
    "classical": "古风典雅", "modern": "现代简约", "trendy": "时尚潮流",
    "mystery": "悬疑暗调", "futuristic": "科技未来",
    "natural": "auto", "minimal": "现代简约", "luxury": "古风典雅", "retro": "古风典雅",
}
_STYLE_WORDS = {
    "古风典雅": ("古风典雅", "古風典雅", "古风风格", "古風風格", "classical chinese style"),
    "现代简约": ("现代简约", "現代簡約", "极简风格", "極簡風格", "minimalist style"),
    "时尚潮流": ("时尚潮流", "時尚潮流", "潮流风格", "潮流風格", "fashion-forward style"),
    "悬疑暗调": ("悬疑暗调", "懸疑暗調", "悬疑风格", "懸疑風格", "mystery style"),
    "科技未来": ("科技未来", "科技未來", "未来科技风格", "未來科技風格", "futuristic style"),
}
_OPENING = set("（([【《〈「『“‘")
_CLOSING = set("，。！？；：、）)]】》〉」』”’!?;:,%％")


def _integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        _fail("INVALID_" + name.upper(), f"{name} 必须是 {minimum}–{maximum} 范围内的整数。",
              [name])
    return value


def _validate_language(text, language, allow_mixed_scripts):
    if language not in ("zh-Hans", "zh-Hant", "en", "mixed"):
        _fail("INVALID_LANGUAGE", "请选择 zh-Hans、zh-Hant、en 或 mixed。", ["language"])
    if type(allow_mixed_scripts) is not bool:
        _fail("INVALID_LANGUAGE", "allow_mixed_scripts 必须是布尔值。", ["allow_mixed_scripts"])
    unsupported = [character for character in text
                   if unicodedata.category(character).startswith("L")
                   and not regex.fullmatch(r"\p{Latin}|\p{Han}", character)]
    if unsupported:
        _fail("LANGUAGE_CONFLICT", "文案含当前简繁中文与英文之外的文字系统。",
              ["copy_text", "language"], characters="".join(sorted(set(unsupported))))
    if language == "mixed" or allow_mixed_scripts:
        return
    if language == "en":
        if regex.search(r"\p{Han}", text):
            _fail("LANGUAGE_CONFLICT", "英文目标语言与文案中的汉字冲突。",
                  ["language", "allow_mixed_scripts"])
        return
    try:
        from opencc import OpenCC
    except ImportError as exc:
        raise PromoError("LANGUAGE_VALIDATION_UNAVAILABLE", "缺少简繁体校验依赖 OpenCC。",
                         ["安装 requirements.txt 中的依赖"]) from exc
    # Used only as a conflict detector. Never draw or retain converted text.
    compared = OpenCC("t2s" if language == "zh-Hans" else "s2t").convert(text)
    if compared != text:
        _fail("LANGUAGE_CONFLICT", "文案与所选简繁体语言可能冲突；系统不会转换原文。",
              ["language", "allow_mixed_scripts"])


def _validate_copy(copy_text, language, allow_mixed_scripts):
    if not isinstance(copy_text, str) or not any(not c.isspace() for c in copy_text):
        _fail("COPY_REQUIRED", "请补充完整上图文案；不会自动生成或补写。", ["copy_text"])
    if len(copy_text) > 12000:
        _fail("TEXT_TOO_LONG", "完整文案超出当前可处理范围。", ["copy_text", "width", "height"])
    controls = {c for c in copy_text if unicodedata.category(c) in {"Cc", "Cf", "Cs", "Zl", "Zp"}
                and c not in "\r\n"}
    if controls:
        _fail("UNSUPPORTED_TEXT_CONTROL", "原文含不能可靠绘制的控制字符；不会自动删除。",
              ["copy_text"], codepoints=[f"U+{ord(c):04X}" for c in sorted(controls)])
    _validate_language(copy_text, language, allow_mixed_scripts)


def _explicit_styles(design_prompt):
    detected = set()
    for name, words in _STYLE_WORDS.items():
        for word in words:
            for match in re.finditer(re.escape(word), design_prompt.lower()):
                prefix = design_prompt.lower()[max(0, match.start() - 24):match.start()]
                if not re.search(r"(?:不要|不用|避免|禁止|不采用|不採用|无需|無需|非)\s*"
                                 r"(?:使用|采用|採用|呈现|呈現)?\s*$"
                                 r"|(?:not|no|avoid|without)\s+(?:(?:a|an|the)\s+)?$", prefix):
                    detected.add(name)
    return detected


def _validate_design(design_prompt, style):
    if not isinstance(design_prompt, str) or len(design_prompt) > 8000:
        _fail("INVALID_DESIGN_PROMPT", "设计要求必须是不超过 8000 字符的视觉描述。", ["design_prompt"])
    if not isinstance(style, str) or style not in _STYLES:
        _fail("INVALID_STYLE", "不支持的视觉风格。", ["style"])
    if re.search(r"(?:写上|寫上|添加文字|加入文字|标题(?:为|是)|標題(?:為|是))\s*[\"“'「『]"
                 r"|(?:add\s+(?:the\s+)?text|write)\s*[:：]?\s*[\"“']", design_prompt, re.I):
        _fail("COPY_IN_DESIGN_PROMPT", "请把完整上图文字放入独立文案输入，设计要求只描述画面。",
              ["copy_text", "design_prompt"])
    if style != "auto":
        detected = _explicit_styles(design_prompt)
        if detected - {style}:
            _fail("STYLE_CONFLICT", "所选风格与视觉要求中的明确风格词冲突。",
                  ["style", "design_prompt"])


def resolve_style(style, design_prompt):
    """Resolve aliases and optional prompt style before a wrapper selects fonts.

    Automatic choice uses only unambiguous named styles. Image references still
    influence the model's composition through the visual prompt.
    """
    if isinstance(style, str):
        style = _STYLE_ALIASES.get(style, style)
    _validate_design(design_prompt, style)
    explicit = _explicit_styles(design_prompt)
    return next(iter(explicit)) if style == "auto" and len(explicit) == 1 else style


def _load_fonts(font_path, title_size, body_size, copy_text):
    try:
        path = Path(font_path)
    except (TypeError, ValueError) as exc:
        raise PromoError("FONT_UNREADABLE", "请提供有效的字体文件路径。", ["font"]) from exc
    if not path.is_file():
        _fail("FONT_UNREADABLE", "所选字体文件不存在。", ["font"], font=path.name)
    try:
        with TTFont(str(path), fontNumber=0, lazy=True) as ft:
            cmap = ft.getBestCmap() or {}
            missing = sorted({c for c in copy_text if c not in "\r\n"
                              and (ord(c) not in cmap or cmap[ord(c)] == ".notdef")})
        if missing:
            _fail("FONT_MISSING_GLYPHS", "字体缺少原文字形，不能用方框或其他字符替代。",
                  ["font"], characters="".join(missing))
        return {"title": ImageFont.truetype(str(path), title_size),
                "body": ImageFont.truetype(str(path), body_size)}
    except PromoError:
        raise
    except Exception as exc:
        raise PromoError("FONT_UNREADABLE", "字体文件无法加载。", ["font"], {"font": path.name}) from exc


def _width(text, font):
    box = font.getbbox(text, anchor="lt")
    return max(float(font.getlength(text)), float(box[2] - box[0]))


def _tokens(paragraph):
    """Keep non-Han words and all Unicode grapheme clusters intact."""
    result, word = [], ""
    for cluster in regex.findall(r"\X", paragraph):
        is_word = not regex.search(r"\p{Han}", cluster) and bool(
            regex.fullmatch(r"[\p{L}\p{M}\p{N}]+", cluster))
        if is_word:
            word += cluster
        elif word and cluster in ("'", "’", "-", "‑"):
            word += cluster
        else:
            if word:
                result.append(word)
                word = ""
            result.append(cluster)
    if word:
        result.append(word)
    assert "".join(result) == paragraph
    return result


def _wrap(text, font, available_width):
    lines = []
    for paragraph in re.split(r"\r\n|\r|\n", text):
        current = []
        for token in _tokens(paragraph):
            if _width(token, font) > available_width:
                raise PromoError("TEXT_TOO_LONG", "完整单词或字形超过当前文字区域的宽度。",
                                 ["width", "title_size", "body_size", "font", "copy_text"])
            if current and _width("".join(current) + token, font) > available_width:
                carry = []
                while current and current[-1] in _OPENING:
                    carry.insert(0, current.pop())
                if token in _CLOSING and current:
                    carry.insert(0, current.pop())
                if carry and (not current or _width("".join(carry) + token, font) > available_width):
                    raise PromoError("TEXT_TOO_LONG", "标点和完整字符在此字号下无法安全换行。",
                                     ["width", "title_size", "body_size", "copy_text"])
                lines.append("".join(current))
                current = carry
            current.append(token)
        lines.append("".join(current))
    return lines


def _advance(font, lines, spacing=.15):
    ascent, descent = font.getmetrics()
    ink_height = max((font.getbbox(line, anchor="lt")[3] - font.getbbox(line, anchor="lt")[1]
                      for line in lines), default=0)
    return max(ascent + descent, ink_height) + max(3, math.ceil(font.size * spacing))


def _mask_array(mask, width, height):
    if mask is None:
        return None
    try:
        if isinstance(mask, Image.Image):
            array = np.asarray(mask.convert("L"))
        else:
            array = np.asarray(mask)
        finite = bool(np.isfinite(array).all())
    except (TypeError, ValueError) as exc:
        raise PromoError("INVALID_PROTECTED_MASK", "主体保护蒙版必须是有效的像素数值数组。",
                         ["protected_mask"]) from exc
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    elif array.ndim == 3 and array.shape[-1] == 1:
        array = array[:, :, 0]
    if array.shape != (height, width) or not finite:
        _fail("INVALID_PROTECTED_MASK", "主体保护蒙版必须与最终画布宽高一致且数值有效。",
              ["protected_mask"], expected=[height, width])
    return array > 0


def _regions(width, height, margin_ratio):
    mx = max(12, round(width * margin_ratio))
    my = max(12, round(height * margin_ratio))
    return {
        "top": (mx, my, width - mx, min(height - my, my + round(.30 * height))),
        "left": (mx, max(my, round(.08 * height)), min(width - mx, mx + round(.45 * width)),
                 min(height - my, round(.86 * height))),
        "bottom": (mx, max(my, round(.65 * height)), width - mx, height - my),
    }


def _plan(name, region, copy_text, first_line_title, fonts, spacing=.15):
    left, top, right, bottom = region
    if first_line_title:
        parts = re.split(r"(\r\n|\r|\n)", copy_text, maxsplit=1)
        title = parts[0]
        body = parts[2] if len(parts) == 3 else None
    else:
        title, body = None, copy_text
    # Small transparent padding includes glyph side bearings and accent edges.
    padding = 3
    width = right - left - padding * 2
    title_lines = _wrap(title, fonts["title"], width) if title is not None else []
    body_lines = _wrap(body, fonts["body"], width) if body is not None else []
    title_advance = _advance(fonts["title"], title_lines, spacing)
    body_advance = _advance(fonts["body"], body_lines, spacing)
    gap = max(10, round(fonts["body"].size * .5)) if title_lines and body_lines else 0
    required_height = len(title_lines) * title_advance + len(body_lines) * body_advance + gap
    if required_height + padding * 2 > bottom - top:
        raise PromoError("TEXT_TOO_LONG", "完整原文在该区域及当前字号下放不下。",
                         ["layout", "width", "height", "title_size", "body_size", "copy_text"])
    y, lines = top + padding, []
    for role, wrapped, advance in (("title", title_lines, title_advance),
                                    ("body", body_lines, body_advance)):
        for text in wrapped:
            bounds = fonts[role].getbbox(text, anchor="lt")
            ink_height = bounds[3] - bounds[1]
            line_width = _width(text, fonts[role])
            lines.append({"text": text, "role": role, "x": left + padding, "y": y,
                          "width": line_width, "height": ink_height, "advance": advance})
            y += advance
        if role == "title":
            y += gap
    ink_lines = [line for line in lines if line["width"] and line["height"]]
    bbox = (left + padding, top + padding,
            math.ceil(max((line["x"] + line["width"] for line in ink_lines), default=left + padding)),
            math.ceil(max((line["y"] + line["height"] for line in ink_lines), default=top + padding)))
    return {"name": name, "region": region, "text_bbox": bbox,
            "title_lines": title_lines, "body_lines": body_lines, "lines": lines,
            "required_height": required_height}


def _overlaps_mask(plan, protected):
    if protected is None:
        return False
    left, top, right, bottom = plan["text_bbox"]
    return bool(protected[top:bottom, left:right].any())


def _rgb_color(value):
    try:
        if not isinstance(value, str):
            raise ValueError("color must be a string")
        rgba = ImageColor.getcolor(value, "RGBA")
        if rgba[3] != 255:
            raise ValueError("transparent foreground")
        return rgba[:3]
    except (TypeError, ValueError) as exc:
        raise PromoError("INVALID_TEXT_COLOR", "文字颜色必须是 auto 或有效的不透明颜色。",
                         ["text_color"]) from exc


def prepare(copy_text, design_prompt, font_path, width, height, count=1,
            first_line_title=True, title_size=96, body_size=36, layout="auto",
            style="auto", text_color="auto", margin_ratio=.06, protected_mask=None,
            language="zh-Hans", allow_mixed_scripts=False):
    """Measure exact copy before sampling and reserve a transparent text area.

    ``protected_mask`` marks pixels that may not be covered by the text block.
    The optional mask must already use the requested final canvas dimensions.
    """
    width = _integer(width, "width", 512, 4096)
    height = _integer(height, "height", 512, 4096)
    count = _integer(count, "count", 1, 4)
    title_size = _integer(title_size, "title_size", 8, 256)
    body_size = _integer(body_size, "body_size", 8, 128)
    if type(first_line_title) is not bool:
        _fail("INVALID_TITLE_MODE", "first_line_title 必须为布尔值。", ["first_line_title"])
    if type(margin_ratio) not in (int, float) or not math.isfinite(margin_ratio) or not .02 <= margin_ratio <= .15:
        _fail("INVALID_MARGIN", "margin_ratio 必须在 0.02–0.15 之间。", ["margin_ratio"])
    if layout not in ("auto", "top", "left", "bottom"):
        _fail("INVALID_LAYOUT", "请选择 auto、top、left 或 bottom。", ["layout"])
    if text_color != "auto":
        _rgb_color(text_color)
    _validate_copy(copy_text, language, allow_mixed_scripts)
    requested_style = style
    style = resolve_style(style, design_prompt)
    fonts = _load_fonts(font_path, title_size, body_size, copy_text)
    protected = _mask_array(protected_mask, width, height)
    regions = _regions(width, height, float(margin_ratio))
    names = list(regions) if layout == "auto" else [layout]
    plans, fitted = [], []
    for name in names:
        try:
            plan = _plan(name, regions[name], copy_text, first_line_title, fonts, _STYLES[style]["spacing"])
        except PromoError as exc:
            if exc.code != "TEXT_TOO_LONG":
                raise
            continue
        fitted.append(plan)
        if not _overlaps_mask(plan, protected):
            plans.append(plan)
    if not fitted:
        _fail("TEXT_TOO_LONG", "完整文案在当前画布、字号及文字区内放不下；不会删减或缩写。",
              ["layout", "width", "height", "title_size", "body_size", "font", "copy_text"])
    if not plans:
        _fail("PROTECTED_SUBJECT_OVERLAP", "可用文字区与主体保护蒙版重叠，已停止排字。",
              ["layout", "protected_mask", "width", "height"])
    reserve = plans[0]
    position = {"top": "画面上方约三分之一保留连续、干净、平整的背景，主体位于下半部",
                "left": "画面左侧约一半保留连续、干净、平整的背景，主体位于右侧",
                "bottom": "画面下方约三分之一保留连续、干净、平整的背景，主体位于上半部"}[reserve["name"]]
    prompt = ("以 <image2> 的人物、产品或界面为主体，在 <image1> 的尺寸内制作完整、连贯的无字主视觉底图，"
              "保留主体外形、身份、数量及品牌特征。" + _STYLES[style]["direction"] + "。" + position
              + "；背景自然延伸到主体周围，整个画面连贯。视觉要求：" + design_prompt
              + "。画面仅包含实际主体、环境、材质和光影。")
    generation_canvas = Image.new("RGB", (math.ceil(width / 32) * 32, math.ceil(height / 32) * 32),
                                  _STYLES[style]["canvas"])
    return {"copy_text": copy_text, "prompt": prompt, "generation_canvas": generation_canvas,
            "width": width, "height": height, "count": count, "language": language,
            "allow_mixed_scripts": allow_mixed_scripts, "candidate_plans": plans,
            "reserved_region": reserve["region"], "reserved_layout": reserve["name"],
            "font_path": str(Path(font_path)), "font": Path(font_path).name, "fonts": fonts,
            "layout": layout, "style": style, "requested_style": requested_style,
            "text_color": text_color,
            "foreground_palette": _STYLES[style]["foreground"],
            "line_spacing_ratio": _STYLES[style]["spacing"],
            "protected_mask": protected, "title_size": title_size, "body_size": body_size,
            "first_line_title": first_line_title}


def _linear_luminance(rgb):
    values = np.asarray(rgb, dtype=np.float64) / 255.0
    values = np.where(values <= .04045, values / 12.92, ((values + .055) / 1.055) ** 2.4)
    # RGB weights are elementwise: avoid platform-specific BLAS warnings in
    # NumPy 2.2 while calculating bounded color luminance.
    return values[..., 0] * .2126 + values[..., 1] * .7152 + values[..., 2] * .0722


def _region_score(background, plan, color_setting, palette=None):
    left, top, right, bottom = plan["text_bbox"]
    crop = np.asarray(background.crop((left, top, max(left + 1, right), max(top + 1, bottom))), dtype=np.float64)
    luminance = _linear_luminance(crop)
    edge_x = float(np.abs(np.diff(luminance, axis=1)).mean()) if crop.shape[1] > 1 else 0.0
    edge_y = float(np.abs(np.diff(luminance, axis=0)).mean()) if crop.shape[0] > 1 else 0.0
    spread = float(np.std(luminance))
    colors = (palette or _STYLES["auto"]["foreground"]) if color_setting == "auto" else (_rgb_color(color_setting),)
    options = []
    for rgb in colors:
        fg_luminance = float(_linear_luminance(rgb))
        contrast = (np.maximum(luminance, fg_luminance) + .05) / (np.minimum(luminance, fg_luminance) + .05)
        lower_contrast = float(np.percentile(contrast, 10))
        options.append((lower_contrast, rgb))
    lower_contrast, color = max(options, key=lambda value: value[0])
    score = edge_x + edge_y + .15 * spread
    return score, lower_contrast, color


def render_text_layer(context, background, protected_mask=None, allow_low_contrast=False):
    """Choose a safe, readable candidate and draw only original glyph pixels.

    No pixels outside the transparent glyph layer are supplied with an opaque
    fill. Caller compositing must happen after image-model generation.
    """
    if not isinstance(background, Image.Image) or background.size != (context["width"], context["height"]):
        _fail("INVALID_BACKGROUND_SIZE", "底图必须已经适配最终目标尺寸，排字节点不会拉伸或裁切它。",
              ["background", "width", "height"], expected=[context["width"], context["height"]])
    background = background.convert("RGB")
    protected = (_mask_array(protected_mask, context["width"], context["height"])
                 if protected_mask is not None else context.get("protected_mask"))
    candidates, fallback_candidates = [], []
    for index, plan in enumerate(context["candidate_plans"]):
        if _overlaps_mask(plan, protected):
            continue
        texture, contrast, color = _region_score(background, plan, context["text_color"],
                                                context["foreground_palette"])
        # Prefer readable regions before considering an outlined fallback.
        # Within either group, retain the reserved-region tie breaker.
        candidate = (texture + index * .003, plan, contrast, color)
        if contrast >= 3.0:
            candidates.append(candidate)
        elif allow_low_contrast:
            fallback_candidates.append(candidate)
    if not candidates:
        candidates = fallback_candidates
    if not candidates:
        if all(_overlaps_mask(plan, protected) for plan in context["candidate_plans"]):
            _fail("PROTECTED_SUBJECT_OVERLAP", "所有可用文字区都与主体保护蒙版重叠。",
                  ["layout", "protected_mask"])
        _fail("LOW_TEXT_CONTRAST", "可用文字区无法在当前文字颜色下保证清晰对比度。",
              ["text_color", "layout", "background"])
    _, plan, contrast, color = min(candidates, key=lambda candidate: candidate[0])
    layer = Image.new("RGBA", background.size, (0, 0, 0, 0))
    low_contrast = contrast < 3.0

    def draw_lines(outline):
        target = Image.new("RGBA", background.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(target)
        stroke_color = (255, 255, 255) if float(_linear_luminance(color)) < .5 else (20, 20, 20)
        for line in plan["lines"]:
            font = context["fonts"][line["role"]]
            bounds = font.getbbox(line["text"], anchor="lt")
            stroke = min(3, max(1, round(font.size * .02))) if outline else 0
            draw.text((line["x"] - bounds[0], line["y"] - bounds[1]), line["text"],
                      font=font, fill=(*color, 255), anchor="lt", stroke_width=stroke,
                      stroke_fill=(*stroke_color, 220))
        return target

    layer = draw_lines(low_contrast)
    alpha = np.asarray(layer)[:, :, 3]
    if low_contrast and protected is not None and (alpha[protected] != 0).any():
        # Keep exact glyphs and subject protection if only the optional outline overlaps.
        layer = draw_lines(False)
        alpha = np.asarray(layer)[:, :, 3]
    if protected is not None and (alpha[protected] != 0).any():
        _fail("PROTECTED_SUBJECT_OVERLAP", "字形像素与保护主体重叠，已停止合成。",
              ["layout", "protected_mask"])
    box = layer.getbbox()
    if box is None:
        _fail("EMPTY_TEXT_LAYER", "字体没有生成可见字形。", ["copy_text", "font"])
    if not (0 <= box[0] < box[2] <= context["width"] and 0 <= box[1] < box[3] <= context["height"]):
        _fail("TEXT_OVERFLOW", "实际字形超出最终画布。", ["layout", "title_size", "body_size"])
    manifest = {"copy_text": context["copy_text"], "language": context["language"],
                "width": context["width"], "height": context["height"], "count": context["count"],
                "font": context["font"], "font_rendering": "real_font_glyphs",
                "style": context["style"], "requested_style": context["requested_style"],
                "line_spacing_ratio": context["line_spacing_ratio"],
                "layout": plan["name"], "text_region": list(plan["region"]),
                "text_bbox": list(box), "text_color": "#%02x%02x%02x" % color,
                "contrast_10th_percentile": round(contrast, 3),
                "title_lines": list(plan["title_lines"]), "body_lines": list(plan["body_lines"]),
                "checks": {"copy": "passed", "glyph_coverage": "passed", "layout": "passed",
                           "contrast": "review_required" if low_contrast else "passed",
                           "subject_protection": "passed" if protected is not None else "not_provided",
                           "background_incidental_text": "not_checked_by_typesetter"},
                "visual_acceptance": "requires_review"}
    return layer, manifest


class LayoutEngine:
    """Convenient stateless facade for the pure typography implementation."""

    prepare = staticmethod(prepare)
    render_text_layer = staticmethod(render_text_layer)


# Friendly alias for integrations that used the earlier prototype function name.
prepare_layout = prepare
