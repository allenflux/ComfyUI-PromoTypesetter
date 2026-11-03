"""Private-runtime CLI. It imports OCR dependencies only in the private Python."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "1"
MASK_NAMES = {"allowed_text_mask", "cleanup_mask", "protected_mask"}


def _load_local_ocr():
    specification = importlib.util.spec_from_file_location("promo_ocr_private_local", ROOT / "promo_typesetter" / "ocr.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _write_response(path, result):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _self_check():
    from importlib.metadata import version
    from PIL import Image, ImageDraw, ImageFont
    ocr = _load_local_ocr()
    image = Image.new("RGB", (900, 260), "white")
    font = ImageFont.truetype(str(ROOT / "fonts" / "NotoSansCJKsc-Regular.otf"), 96)
    ImageDraw.Draw(image).text((40, 60), "Promo 2026", font=font, fill="black")
    polygons, _ = ocr._detect_regions(image)
    if not polygons:
        raise RuntimeError("OCR self-check did not detect the test text")
    return {"status": "ok", "protocol": PROTOCOL, "provider": "CPUExecutionProvider",
            "detector_version": ocr.RAPIDOCR_VERSION, "detection_regions": len(polygons),
            "versions": {"python": sys.version.split()[0], **{name: version(name) for name in
                         ("numpy", "opencv-python", "onnxruntime", "rapidocr-onnxruntime")}}}


def _inside(directory, name, expected):
    if name != expected:
        raise ValueError("Unexpected request filename")
    path = directory / expected
    if path.is_symlink():
        raise ValueError("Symlink paths are not accepted")
    return path


def process_request(request_path, response_path):
    """Process a host-created temporary request with no external file references."""
    directory = request_path.parent.resolve()
    if request_path.is_symlink() or response_path.parent.resolve() != directory or response_path.is_symlink():
        raise ValueError("Request and response must be private files in the same directory")
    if request_path.stat().st_size > 65536:
        raise ValueError("Request is too large")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(request, dict) or request.get("protocol") != PROTOCOL:
        raise ValueError("Unsupported OCR worker request protocol")
    if request.get("mode") not in ("reject", "repair_masked"):
        raise ValueError("Unsupported OCR mode")
    image_path = _inside(directory, request.get("image"), "image.png")
    output_path = _inside(directory, request.get("output"), "output.png")
    masks = request.get("masks", {})
    if not isinstance(masks, dict) or set(masks) - MASK_NAMES:
        raise ValueError("Unexpected mask names")

    from PIL import Image
    ocr = _load_local_ocr()
    try:
        with Image.open(image_path) as opened:
            if opened.format != "PNG":
                raise ValueError("Input must be PNG")
            image = opened.copy()
        mask_images = {}
        for name, leaf in masks.items():
            with Image.open(_inside(directory, leaf, name + ".png")) as opened:
                if opened.format != "PNG" or opened.mode != "L":
                    raise ValueError("Mask must be a grayscale PNG")
                mask_images[name] = opened.copy()
        # Never call a dispatcher here: only the private runtime's local detector.
        local_inspect = getattr(ocr, "_inspect_background_local", None)
        if local_inspect is None:
            raise RuntimeError("OCR local implementation is missing; update the full plugin")
        result, checks = local_inspect(image, mode=request["mode"], **mask_images)
        result.save(output_path, format="PNG")
        _write_response(response_path, {"protocol": PROTOCOL, "status": "ok",
                                        "output": "output.png", "checks": checks})
        return 0
    except ocr.OCRGuardError as exc:
        _write_response(response_path, {"protocol": PROTOCOL, "status": "error", "error": {
            "code": exc.code, "message": exc.message, "needs_adjustment": exc.needs_adjustment,
            "details": exc.details}})
        return 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--response", type=Path)
    args = parser.parse_args(argv)
    if args.self_check:
        try:
            print(json.dumps(_self_check(), ensure_ascii=False))
            return 0
        except Exception as exc:
            print(json.dumps({"status": "error", "protocol": PROTOCOL, "error_type": type(exc).__name__}))
            return 1
    if args.request is None or args.response is None:
        parser.error("--request and --response are required")
    try:
        return process_request(args.request, args.response)
    except Exception as exc:
        # No traceback or environment dump: runtime paths may contain private information.
        try:
            _write_response(args.response, {"protocol": PROTOCOL, "status": "error", "error": {
                "code": "OCR_WORKER_FAILED", "message": "独立 OCR 环境处理失败，不能放行图片。",
                "needs_adjustment": ["重新运行 python scripts/setup_ocr.py，并检查独立 OCR 环境。"],
                "details": {"error_type": type(exc).__name__}}})
        except OSError:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
