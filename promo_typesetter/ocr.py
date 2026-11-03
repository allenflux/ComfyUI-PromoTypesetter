"""Fail-closed text-region guard for a generated background, before typesetting.

RapidOCR is used in *detection-only* mode. Illegible invented letters must not
disappear from the results merely because a recognizer cannot read them. Models
come from the pinned rapidocr-onnxruntime wheel; this module makes no downloads.
"""

from __future__ import annotations

import json
import threading
from importlib import metadata
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


RAPIDOCR_VERSION = "1.4.4"
_ENGINE: Any = None
_ENGINE_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()
_REPAIR_DILATION_PX = 3


class OCRGuardError(ValueError):
    """A background cannot safely be handed to the final typography stage."""

    def __init__(
        self,
        code: str,
        message: str,
        needs_adjustment: list[str] | str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.needs_adjustment = (
            [needs_adjustment] if isinstance(needs_adjustment, str) else needs_adjustment
        )
        self.details = details or {}
        super().__init__(message)

    def __str__(self) -> str:
        return json.dumps(
            {
                "code": self.code,
                "message": self.message,
                "needs_adjustment": self.needs_adjustment,
                "details": self.details,
            },
            ensure_ascii=False,
        )


def _unavailable(exc: Exception) -> OCRGuardError:
    return OCRGuardError(
        "OCR_UNAVAILABLE",
        "背景文字检测不可用，不能将未检查的图片标记为成功。",
        [
            "运行 python scripts/setup_ocr.py 配置独立 OCR 环境，并确认本地 ONNX 模型可读取。",
            "修复检测器后重新运行；不要跳过背景文字检测。",
        ],
        {"cause": f"{type(exc).__name__}: {exc}"},
    )


def _get_engine() -> Any:
    """Load once, with explicit CPU-only providers and no network fallback."""
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is not None:
            return _ENGINE
        try:
            installed = metadata.version("rapidocr-onnxruntime")
            if installed != RAPIDOCR_VERSION:
                raise RuntimeError(
                    f"Expected rapidocr-onnxruntime=={RAPIDOCR_VERSION}, got {installed}"
                )
            import onnxruntime
            from rapidocr_onnxruntime import RapidOCR

            onnxruntime.disable_telemetry_events()

            engine = RapidOCR(
                det_use_cuda=False,
                det_use_dml=False,
                cls_use_cuda=False,
                cls_use_dml=False,
                rec_use_cuda=False,
                rec_use_dml=False,
                intra_op_num_threads=2,
                inter_op_num_threads=1,
                det_box_thresh=0.4,
                det_thresh=0.3,
            )
            providers = engine.text_det.infer.session.get_providers()
            if providers != ["CPUExecutionProvider"]:
                raise RuntimeError(f"Expected CPUExecutionProvider only, got {providers}")
            _ENGINE = engine
        except Exception as exc:
            raise _unavailable(exc) from exc
        return _ENGINE


def _detect_regions(image: Image.Image) -> tuple[list[list[list[float]]], list[float]]:
    """Return polygons, never recognition-filtered words or confidence scores."""
    try:
        engine = _get_engine()
        # RapidOCR's ndarray input is BGR; RGBA is composited onto white first.
        if "A" in image.getbands():
            rgb = Image.new("RGBA", image.size, (255, 255, 255, 255))
            rgb.alpha_composite(image.convert("RGBA"))
            rgb = rgb.convert("RGB")
        else:
            rgb = image.convert("RGB")
        bgr = np.ascontiguousarray(np.asarray(rgb)[:, :, ::-1])
        # The v1.4.4 detector updates preprocessing state per image.
        with _INFERENCE_LOCK:
            result, elapsed = engine(
                bgr, use_det=True, use_cls=False, use_rec=False
            )
        if result is None:
            return [], [] if elapsed is None else [float(v) for v in elapsed]
        points = np.asarray(result, dtype=np.float64)
        if points.size == 0:
            return [], [] if elapsed is None else [float(v) for v in elapsed]
        if points.ndim != 3 or points.shape[1:] != (4, 2) or not np.isfinite(points).all():
            raise RuntimeError("Invalid detection-only result; expected finite Nx4x2 polygons")
        width, height = image.size
        points[:, :, 0] = np.clip(points[:, :, 0], 0, width - 1)
        points[:, :, 1] = np.clip(points[:, :, 1], 0, height - 1)
        for polygon in points:
            x, y = polygon[:, 0], polygon[:, 1]
            if abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))) < 1:
                raise RuntimeError("Detector returned a degenerate polygon")
        return points.tolist(), [] if elapsed is None else [float(v) for v in elapsed]
    except OCRGuardError:
        raise
    except Exception as exc:
        raise _unavailable(exc) from exc


def _mask_array(mask: Any, size: tuple[int, int], name: str) -> np.ndarray:
    """Masks are explicit, same-sized regions; never stretch or broadcast them."""
    height, width = size[1], size[0]
    if mask is None:
        return np.zeros((height, width), dtype=bool)
    try:
        if isinstance(mask, Image.Image):
            if mask.mode not in ("1", "L", "I", "F"):
                raise ValueError("use a grayscale mask")
            arr = np.asarray(mask)
        else:
            # Accept a single ComfyUI MASK without importing torch.
            if hasattr(mask, "detach"):
                mask = mask.detach().cpu().numpy()
            arr = np.asarray(mask)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[:, :, 0]
        if arr.shape != (height, width):
            raise ValueError(f"expected {(height, width)}, got {arr.shape}")
        if not (np.issubdtype(arr.dtype, np.number) or arr.dtype == np.bool_):
            raise ValueError("mask must contain finite numeric values")
        if not np.isfinite(arr).all() or (arr < 0).any():
            raise ValueError("mask values must be finite and nonnegative")
        if arr.dtype == np.bool_:
            return arr.copy()
        if np.issubdtype(arr.dtype, np.floating):
            if (arr > 1).any():
                raise ValueError("floating masks must be in [0,1]")
            return arr >= 0.5
        # Binary integer masks use 0/1; 8-bit grayscale uses 0..255.
        if (arr > 255).any():
            raise ValueError("integer masks must be in [0,255]")
        return arr >= (1 if arr.max(initial=0) <= 1 else 128)
    except Exception as exc:
        raise OCRGuardError(
            "INVALID_OCR_MASK",
            f"{name} 无法读取或尺寸不匹配。",
            f"提供与输出图片同尺寸的单通道 {name}；白色为指定区域，黑色为其他区域。",
            {"mask": name, "cause": str(exc), "required_size": list(size)},
        ) from exc


def _polygon_mask(polygon: list[list[float]], size: tuple[int, int]) -> np.ndarray:
    canvas = Image.new("1", size)
    ImageDraw.Draw(canvas).polygon([tuple(point) for point in polygon], fill=1)
    return np.asarray(canvas, dtype=bool)


def _classify_regions(
    polygons: list[list[list[float]]], allowed: np.ndarray, size: tuple[int, int]
) -> tuple[list[dict[str, Any]], np.ndarray]:
    regions: list[dict[str, Any]] = []
    unwanted = np.zeros_like(allowed)
    for index, polygon in enumerate(polygons):
        area = _polygon_mask(polygon, size)
        is_allowed = bool(area.any() and np.all(allowed[area]))
        regions.append({"index": index, "polygon": polygon, "allowed": is_allowed})
        if not is_allowed:
            unwanted |= area
    return regions, unwanted


def _inspect_background_local(
    image: Image.Image,
    allowed_text_mask: Any = None,
    cleanup_mask: Any = None,
    protected_mask: Any = None,
    mode: str = "reject",
) -> tuple[Image.Image, dict[str, Any]]:
    """Inspect all background text before adding the caller's exact font layer.

    ``reject`` is the default. ``repair_masked`` may inpaint only detected text
    fully inside an explicit cleanup mask, including the 3 px dilation margin.
    Original brand text can be explicitly allowed. Protected subjects and allowed
    text must not be touched by repair. A second detector pass is mandatory.

    A pass means no *detected* unallowed regions, not proof that every possible
    tiny/stylized glyph is absent. The exact caller copy is checked separately.
    """
    if mode not in {"reject", "repair_masked"}:
        raise OCRGuardError(
            "INVALID_OCR_MODE", "背景文字检测模式无效。", "选择 reject 或 repair_masked。"
        )
    if not isinstance(image, Image.Image):
        raise OCRGuardError(
            "IMAGE_UNREADABLE", "背景图片无法读取。", "提供可读取的宣传素材图片。"
        )
    try:
        image.load()
        if min(image.size) < 4:
            raise ValueError("image is too small for text detection")
    except Exception as exc:
        raise OCRGuardError(
            "IMAGE_UNREADABLE", "背景图片无法读取或尺寸过小。", "提供可读取的宣传素材图片。",
            {"cause": str(exc)},
        ) from exc

    allowed = _mask_array(allowed_text_mask, image.size, "allowed_text_mask")
    cleanup = _mask_array(cleanup_mask, image.size, "cleanup_mask")
    protected = _mask_array(protected_mask, image.size, "protected_mask")
    polygons, elapsed = _detect_regions(image)
    regions, unwanted = _classify_regions(polygons, allowed, image.size)
    checks: dict[str, Any] = {
        "status": "inspected",
        "mode": mode,
        "detector": "rapidocr-onnxruntime",
        "detector_version": RAPIDOCR_VERSION,
        "provider": "CPUExecutionProvider",
        "recognition_used": False,
        "detected_count": len(regions),
        "allowed_count": sum(region["allowed"] for region in regions),
        "unexpected_count": sum(not region["allowed"] for region in regions),
        "boxes": regions,
        "elapsed_seconds": elapsed,
        "limitation": "检测模型可能漏检极小、低对比度或高度变形的文字；通过不等于绝对无字。",
    }
    if not unwanted.any():
        checks["status"] = "passed"
        return image.copy(), checks
    if mode == "reject":
        checks["status"] = "rejected"
        raise OCRGuardError(
            "BACKGROUND_TEXT_DETECTED",
            "背景中检测到未经允许的文字或疑似乱码，已阻止输出。",
            [
                "重新生成无文字背景，并仅在最终字体排版节点输入指定文案。",
                "保留原始品牌文字时提供 allowed_text_mask；需要修复时明确提供安全 cleanup_mask。",
            ],
            checks,
        )
    repair = np.asarray(
        Image.fromarray(unwanted.astype(np.uint8) * 255).filter(
            ImageFilter.MaxFilter(2 * _REPAIR_DILATION_PX + 1)
        ),
        dtype=bool,
    )
    unsafe = repair & (~cleanup | protected | allowed)
    if unsafe.any():
        checks["status"] = "unsafe_repair"
        raise OCRGuardError(
            "UNSAFE_TEXT_REPAIR",
            "疑似文字或修复边缘超出安全清理区域，或覆盖受保护主体/品牌文字。",
            [
                "调整 cleanup_mask，使其覆盖全部待清理文字及至少 3 像素边缘。",
                "确保清理区域不接触人物、产品、标识或 allowed_text_mask；否则重新生成背景。",
            ],
            {**checks, "unsafe_pixel_count": int(unsafe.sum()), "dilation_px": _REPAIR_DILATION_PX},
        )
    try:
        import cv2

        # RGB channel order is immaterial to inpaint; retain the original alpha.
        rgb = np.asarray(image.convert("RGB"))
        repaired_array = cv2.inpaint(rgb, repair.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
        repaired = Image.fromarray(repaired_array)
        if "A" in image.getbands():
            repaired.putalpha(image.getchannel("A"))
    except Exception as exc:
        checks["status"] = "repair_failed"
        raise OCRGuardError(
            "TEXT_REPAIR_FAILED", "安全区域的文字清理失败。", "检查 OpenCV 安装或重新生成无文字背景。",
            {**checks, "cause": f"{type(exc).__name__}: {exc}"},
        ) from exc
    final_polygons, final_elapsed = _detect_regions(repaired)
    final_regions, remaining = _classify_regions(final_polygons, allowed, image.size)
    checks.update(
        {
            "status": "rechecked",
            "repair_pixel_count": int(repair.sum()),
            "dilation_px": _REPAIR_DILATION_PX,
            "post_repair_boxes": final_regions,
            "post_repair_unexpected_count": sum(not region["allowed"] for region in final_regions),
            "post_repair_elapsed_seconds": final_elapsed,
        }
    )
    if remaining.any():
        checks["status"] = "rejected"
        raise OCRGuardError(
            "BACKGROUND_TEXT_REMAINS", "清理后仍检测到未经允许的文字，已阻止输出。",
            "重新生成无文字背景，或调整安全清理区域后重试。", checks,
        )
    checks["status"] = "repaired"
    return repaired, checks


def inspect_background(image, allowed_text_mask=None, cleanup_mask=None,
                       protected_mask=None, mode="reject"):
    """Inspect in the private interpreter without importing host cv2/ONNX.

    The worker calls ``_inspect_background_local``. A missing or failed private
    runtime is an explicit error; there is no host-library fallback.
    """
    from .ocr_process import inspect_isolated

    return inspect_isolated(image, allowed_text_mask=allowed_text_mask,
                            cleanup_mask=cleanup_mask, protected_mask=protected_mask,
                            mode=mode)
