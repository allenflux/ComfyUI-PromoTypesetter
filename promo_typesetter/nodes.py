"""ComfyUI adapters. Model generation stays in the native Qwen workflow."""
from __future__ import annotations

import hashlib
import json
import math
import re
import logging
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin
import torch

from .layout import PromoError, prepare, render_text_layer, resolve_style
from .ocr import OCRGuardError, inspect_background


ROOT = Path(__file__).resolve().parents[1]
STYLES = ["自动", "古风典雅", "现代简约", "时尚潮流", "悬疑暗调", "科技未来"]
RATIOS = {(1, 1), (3, 4), (4, 3), (9, 16), (16, 9)}


def _error(code, message, fields=None, **details):
    raise PromoError(code, message, fields or [], details)


def _material(image, label):
    if image is None:
        _error("MATERIAL_REQUIRED", "请上传可读取的宣传素材；不能退化为纯文生图。", [label])
    if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[0] != 1:
        _error("INVALID_MATERIAL", "每个素材输入须为一张有效 IMAGE，不支持批量素材槽。", [label])
    if image.shape[-1] not in (3, 4) or min(image.shape[1:3]) < 128:
        _error("INVALID_MATERIAL", "素材须为 RGB/RGBA 图片且短边至少 128 像素，建议不低于 512。", [label])
    if image.shape[1] * image.shape[2] > 40_000_000:
        _error("MATERIAL_TOO_LARGE", "单张素材超过 4000 万像素，请先调整上传图片。", [label])
    if not bool(torch.isfinite(image).all()) or bool((image < 0).any()) or bool((image > 1).any()):
        _error("INVALID_MATERIAL", "素材像素数据无效，需重新上传。", [label])
    return {"slot": label, "width": int(image.shape[2]), "height": int(image.shape[1]),
            "warning": "建议上传短边至少512像素的原图" if min(image.shape[1:3]) < 512 else None}


def _target_size(width, height):
    if type(width) is not int or type(height) is not int:
        _error("INVALID_SIZE", "宽高必须为整数。", ["width", "height"])
    if min(width, height) < 512 or not 1024 <= max(width, height) <= 2048:
        _error("INVALID_SIZE", "长边应为1024–2048，短边至少512像素。", ["width", "height"])
    if width % 32 or height % 32:
        _error("INVALID_SIZE", "宽高须为32的倍数，以直接生成目标尺寸而不拉伸或裁切。", ["width", "height"],
               examples=[[1024, 1024], [1152, 1536], [1536, 1152], [864, 1536], [1536, 864]])
    divisor = math.gcd(width, height)
    if (width // divisor, height // divisor) not in RATIOS:
        _error("INVALID_RATIO", "仅支持1:1、3:4、4:3、9:16、16:9。", ["width", "height"])


def _mask(mask, width, height, label, index=0, count=1):
    if mask is None:
        return None
    if isinstance(mask, torch.Tensor):
        array = mask.detach().cpu().numpy()
    else:
        array = np.asarray(mask)
    if array.ndim == 3 and array.shape[0] in (1, count):
        array = array[index if array.shape[0] > 1 else 0]
    if array.shape != (height, width) or not np.isfinite(array).all():
        _error("INVALID_MASK", "蒙版必须与目标尺寸一致；白色表示启用区域。", [label],
               expected=[height, width])
    if (array < 0).any() or (array > 1).any():
        _error("INVALID_MASK", "蒙版数值必须在0–1之间。", [label])
    return array > 0


def _material_accent(material):
    """Choose a usable scene color without a detector or a new model call."""
    thumb = material[0, :, :, :3].detach().cpu().numpy()
    thumb = Image.fromarray(np.round(thumb * 255).astype(np.uint8), "RGB")
    thumb.thumbnail((128, 128))
    quantized = thumb.quantize(colors=12)
    palette = quantized.getpalette()
    candidates = []
    for amount, index in quantized.getcolors() or []:
        rgb = tuple(palette[index * 3:index * 3 + 3])
        high, low = max(rgb) / 255, min(rgb) / 255
        saturation = (high - low) / max(high, .001)
        if .2 < high < .96 and saturation > .18:
            candidates.append((amount * (.2 + saturation), rgb))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


class PromoTextPrepare:
    """Validate original copy before expensive sampling; prepare a full canvas."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "material": ("IMAGE", {"tooltip": "必需的实际宣传主体素材，不能仅用风格参考图"}),
            "copy_text": ("STRING", {"multiline": True, "default": "", "tooltip": "新工作流的标题原文；正文在独立body_text输入。旧图仍支持第一行标题、其余正文的完整原文"}),
            "language": (["zh-Hans", "zh-Hant", "en"],),
            "design_prompt": ("STRING", {"multiline": True, "default": "完整商业海报设计，主体突出，标题区与主体紧凑呼应；用协调的色块、曲线与光影建立层次，保留产品外观。", "tooltip": "只描述视觉，不填上图文案。image1是无字构图与配色底板，image2是主要素材"}),
            "style": (STYLES,),
            "layout": (["auto", "top", "left", "bottom"],),
            "width": ("INT", {"default": 1152, "min": 512, "max": 2048, "step": 32}),
            "height": ("INT", {"default": 1536, "min": 512, "max": 2048, "step": 32}),
            "count": ("INT", {"default": 1, "min": 1, "max": 4}),
            "first_line_title": ("BOOLEAN", {"default": True}),
            "title_size": ("INT", {"default": 112, "min": 8, "max": 256}),
            "body_size": ("INT", {"default": 40, "min": 8, "max": 128}),
            "font_family": (["auto", "sans", "serif"],),
            "text_color": ("STRING", {"default": "auto"}),
            "allow_mixed_scripts": ("BOOLEAN", {"default": False, "tooltip": "允许原文简繁/中英混排，不转换任何字符"}),
        }, "optional": {
            "material_2": ("IMAGE",), "material_3": ("IMAGE",),
            "material_4": ("IMAGE",), "material_5": ("IMAGE",),
            "protected_mask": ("MASK", {"tooltip": "最终画布上不可盖字/修补的主体、品牌区域；需要调用方标记"}),
            "typography_mode": (["campaign", "manual"], {"tooltip": "campaign：按原文规划大标题、配色和无字海报构图；manual：使用指定字号及原有排版"}),
            "body_text": ("STRING", {"multiline": True, "default": "", "tooltip": "完整正文原文，可留空。不会补写、翻译、改写或转换简繁，也不会发送给Qwen"}),
        }}

    RETURN_TYPES = ("IMAGE", "PROMO_CONTEXT", "STRING", "INT")
    RETURN_NAMES = ("generation_canvas", "context", "background_prompt", "count")
    FUNCTION = "run"
    CATEGORY = "Promo Typesetter"
    DESCRIPTION = "先规划真实字体标题和主体构图，再输出不含文字的海报配色底板及Qwen视觉提示词。"

    def run(self, material, copy_text, language="zh-Hans", design_prompt="", style="自动",
            layout="auto", width=1152, height=1536, count=1, first_line_title=True,
            title_size=112, body_size=40, font_family="auto", text_color="auto",
            allow_mixed_scripts=False, material_2=None, material_3=None, material_4=None,
            material_5=None, protected_mask=None, typography_mode="campaign", body_text=None):
        _target_size(width, height)
        materials = [_material(material, "material")]
        for i, image in enumerate((material_2, material_3, material_4, material_5), 2):
            if image is not None:
                materials.append(_material(image, f"material_{i}"))
        if style not in STYLES or font_family not in ("auto", "sans", "serif"):
            _error("INVALID_STYLE", "请选择可用的风格与字体家族。", ["style", "font_family"])
        if typography_mode not in ("campaign", "manual"):
            _error("INVALID_TYPOGRAPHY_MODE", "请选择 campaign 或 manual。", ["typography_mode"])
        if body_text is not None and not isinstance(body_text, str):
            _error("INVALID_BODY_TEXT", "正文必须是原文字符串，可留空。", ["body_text"])
        selected_style = resolve_style("auto" if style == "自动" else style, design_prompt)
        accent = _material_accent(material) if typography_mode == "campaign" and selected_style == "auto" else None
        warm_scene = accent is not None and accent[0] > accent[2] * 1.15
        family = ("serif" if selected_style == "古风典雅" or (
            typography_mode == "campaign" and selected_style == "auto" and warm_scene)
            else "sans") if font_family == "auto" else font_family
        script = "tc" if language == "zh-Hant" else "sc"
        filename = f"Noto{'Serif' if family == 'serif' else 'Sans'}CJK{script}-Regular.otf"
        protected = _mask(protected_mask, width, height, "protected_mask")
        context = prepare(copy_text, design_prompt, ROOT / "fonts" / filename, width, height,
                          count=count, first_line_title=first_line_title, title_size=title_size,
                          body_size=body_size, layout=layout, style=selected_style,
                          text_color=text_color, protected_mask=protected, language=language,
                          allow_mixed_scripts=allow_mixed_scripts, typography_mode=typography_mode,
                          campaign_accent=accent,
                          title_text=copy_text if body_text is not None else None, body_text=body_text)
        context["materials"] = materials
        # Semantic roles and actual material images are fed directly to the native Qwen encoder.
        # Context NEVER substitutes the uploaded image for a missing model input.
        context["version"] = "0.1.0"
        context["copy_sha256"] = hashlib.sha256(context["copy_text"].encode("utf-8")).hexdigest()
        context["prompt"] += (" 不生成任何标题、正文、字母、数字、标语、按钮文案、标签或伪文字。"
                              "本阶段只生成无字底图，完整指定文案由后续真实字体程序绘制。"
                              "预留字形位置保持可读的自然背景；允许无字色带、刷痕、曲线和纸张层次形成完整广告设计，"
                              "设计元素与主体自然交织，避免大块空白墙面与产品摄影上方机械贴字。"
                              "不画文字占位符，不把提示词写到画面上。"
                              "素材中原有品牌标识保持原样，不新增标识。")
        canvas = torch.from_numpy(np.asarray(context["generation_canvas"]).copy()).float().div(255).unsqueeze(0)
        return canvas, context, context["prompt"], count


def _error_record(exc):
    if isinstance(exc, (PromoError, OCRGuardError)):
        error = {"code": exc.code, "message": exc.message,
                 "needs_adjustment": exc.needs_adjustment}
        if exc.details:
            error["details"] = exc.details
    else:
        error = {"code": "OUTPUT_FAILED", "message": "该张图片处理或保存失败。",
                 "needs_adjustment": ["查看ComfyUI后端日志及输出目录权限"], "error_type": type(exc).__name__}
    return error


def _failure(index, context, exc):
    return {"index": index, "status": "failed", "generation_status": "failed",
            "validation_status": "failed", "warnings": [], "url": None, "width": context["width"],
            "height": context["height"], "format": "png", "copy_text": context["copy_text"],
            "title_text": context.get("title_text"), "body_text": context.get("body_text"),
            "language": context["language"], "error": _error_record(exc)}


class PromoRenderAndSave:
    """Check each generated frame, typeset exact glyphs, save and report every item."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",), "context": ("PROMO_CONTEXT",),
            "filename_prefix": ("STRING", {"default": "PromoTypesetter"}),
            "ocr_mode": (["off", "reject", "repair_masked"],),
        }, "optional": {"allowed_text_mask": ("MASK", {"tooltip": "允许保留的素材原有品牌文字区域，不应覆盖新增乱码"}),
                        "cleanup_mask": ("MASK", {"tooltip": "仅repair_masked模式使用；不可覆盖主体或品牌"}),
                        "quality_policy": (["return_with_warnings", "strict"],
                                           {"tooltip": "默认照常出图，OCR只返回复核提醒；strict才拦截未通过校验的图片"})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result_json",)
    FUNCTION = "run"
    CATEGORY = "Promo Typesetter"
    OUTPUT_NODE = True
    DESCRIPTION = "无字底图→自动选区→真实字体排字→PNG与预览。默认不运行OCR，直接返回图片。"

    def run(self, images, context, filename_prefix="PromoTypesetter", ocr_mode="off",
            quality_policy="return_with_warnings", allowed_text_mask=None, cleanup_mask=None):
        from urllib.parse import urlencode
        import folder_paths

        if not isinstance(context, dict) or context.get("version") != "0.1.0":
            _error("INVALID_CONTEXT", "请连接本插件的PromoTextPrepare节点。", ["context"])
        if quality_policy not in ("return_with_warnings", "strict"):
            _error("INVALID_QUALITY_POLICY", "请选择 return_with_warnings 或 strict。", ["quality_policy"])
        width, height, count = context["width"], context["height"], context["count"]
        results, previews = [], []
        batch_ok = (isinstance(images, torch.Tensor) and images.ndim == 4
                    and images.shape[0] == count and images.shape[1:3] == (height, width)
                    and images.shape[-1] in (3, 4))
        if not batch_ok:
            exc = PromoError("BATCH_SIZE_MISMATCH", "解码图片的数量或尺寸与请求不一致；请连接count到RepeatLatentBatch.amount。",
                             ["RepeatLatentBatch.amount", "images", "width", "height"],
                             {"expected": [count, height, width, 3],
                              "actual": list(images.shape) if isinstance(images, torch.Tensor) else None})
            results = [_failure(i, context, exc) for i in range(count)]
        else:
            safe_prefix = bool(re.fullmatch(r"[\w\- /]{1,120}", filename_prefix)
                               and not Path(filename_prefix).is_absolute()
                               and ".." not in Path(filename_prefix).parts)
            for index in range(count):
                path = None
                try:
                    warnings = []
                    if not safe_prefix:
                        _error("INVALID_FILENAME", "保存前缀只能为输出目录内的名称或子目录，不能使用绝对路径或..。", ["filename_prefix"])
                    frame = images[index].detach().cpu().numpy()
                    if not np.isfinite(frame).all() or (frame < 0).any() or (frame > 1).any():
                        _error("INVALID_GENERATED_IMAGE", "模型返回的图片像素无效。", ["images"])
                    background = Image.fromarray(np.round(frame[:, :, :3] * 255).astype(np.uint8), "RGB")
                    allowed = _mask(allowed_text_mask, width, height, "allowed_text_mask", index, count)
                    cleanup = _mask(cleanup_mask, width, height, "cleanup_mask", index, count)
                    if ocr_mode == "off":
                        ocr = {"status": "not_run", "mode": "off"}
                    else:
                        try:
                            checked, ocr = inspect_background(background, allowed_text_mask=allowed,
                                                              cleanup_mask=cleanup, protected_mask=context.get("protected_mask"),
                                                              mode=ocr_mode)
                            background = checked
                        except OCRGuardError as exc:
                            if quality_policy == "strict":
                                raise
                            warning = _error_record(exc)
                            warnings.append(warning)
                            # Optional OCR never destroys a generated frame in the
                            # return-with-warnings policy. Keep its original pixels.
                            ocr = {**exc.details, "status": "review_required", "error": warning}
                            logging.getLogger(__name__).warning(
                                "Promo image %d: %s; returning image for review", index, exc.code)
                    layer, manifest = render_text_layer(
                        context, background, allow_low_contrast=quality_policy != "strict")
                    if manifest["checks"].get("contrast") == "review_required":
                        warnings.append({"code": "LOW_TEXT_CONTRAST", "message": "文字区域背景较复杂，已保留原文并返回图片，请复核可读性。",
                                         "needs_adjustment": ["text_color", "layout", "design_prompt"], "details": {}})
                    output = Image.alpha_composite(background.convert("RGBA"), layer).convert("RGB")
                    manifest["checks"]["background_incidental_text"] = (
                        "not_checked" if ocr_mode == "off" else
                        "review_required" if ocr.get("status") == "review_required" else "passed_detection")
                    manifest["ocr"] = ocr
                    manifest["warnings"] = warnings
                    manifest["quality_policy"] = quality_policy
                    manifest["copy_sha256"] = context["copy_sha256"]
                    manifest["image_generation"] = "qwen_external_native_workflow"
                    manifest["checks"]["identity_and_design"] = "requires_visual_review"
                    full_output, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
                        filename_prefix, folder_paths.get_output_directory(), width, height)
                    # Exclusive creation prevents overwrite if concurrent queues resolve the same counter.
                    for attempt in range(1000):
                        leaf = f"{filename}_{counter + attempt:05d}_.png"
                        path = Path(full_output) / leaf
                        try:
                            handle = path.open("xb")
                            break
                        except FileExistsError:
                            continue
                    else:
                        raise OSError("No available output filename")
                    metadata = PngImagePlugin.PngInfo()
                    metadata.add_text("promo", json.dumps(manifest, ensure_ascii=False))
                    try:
                        with handle:
                            output.save(handle, format="PNG", pnginfo=metadata)
                    except Exception:
                        path.unlink(missing_ok=True)
                        raise
                    file_info = {"filename": leaf, "subfolder": subfolder, "type": "output"}
                    item_status = "review_required" if warnings else "success"
                    result = {"index": index, "status": item_status, "generation_status": "success",
                              "validation_status": "review_required" if warnings else "passed",
                              "warnings": warnings, "url": "/view?" + urlencode(file_info),
                              "width": width, "height": height, "format": "png",
                              "copy_text": context["copy_text"], "language": context["language"],
                              "title_text": context.get("title_text"), "body_text": context.get("body_text"),
                              "copy_sha256": context["copy_sha256"], "file": file_info,
                              "typography": manifest, "error": None}
                    results.append(result)
                    previews.append(file_info)
                except Exception as exc:
                    if not isinstance(exc, (PromoError, OCRGuardError)):
                        logging.getLogger(__name__).exception("Promo image %d processing failed", index)
                    results.append(_failure(index, context, exc))
        passed = sum(item["status"] == "success" for item in results)
        review = sum(item["status"] == "review_required" for item in results)
        success = passed + review
        overall = ("partial_success" if success and success < count else "failed" if not success
                   else "review_required" if review else "success")
        payload = {"schema_version": "0.1.0", "status": overall,
                   "requested_count": count, "success_count": success, "produced_count": success,
                   "passed_count": passed, "review_count": review, "failed_count": count - success,
                   "copy_text": context["copy_text"], "language": context["language"], "images": results,
                   "title_text": context.get("title_text"), "body_text": context.get("body_text"),
                   "visual_acceptance": "requires_review"}
        encoded = json.dumps(payload, ensure_ascii=False)
        return {"ui": {"images": previews, "text": [encoded]}, "result": (encoded,)}


NODE_CLASS_MAPPINGS = {"PromoTextPrepare": PromoTextPrepare, "PromoRenderAndSave": PromoRenderAndSave}
NODE_DISPLAY_NAME_MAPPINGS = {"PromoTextPrepare": "宣传图 · 原文与设计预检", "PromoRenderAndSave": "宣传图 · 精准排字与逐图结果"}
