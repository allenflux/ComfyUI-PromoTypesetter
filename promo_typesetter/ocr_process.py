"""Run OCR in the plugin's private Python, without importing host cv2/ONNX."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "1"
TIMEOUT_SECONDS = 120
MAX_RESPONSE_BYTES = 4 * 1024 * 1024


def runtime_python():
    """Return the fixed private interpreter path; never use a caller shell value."""
    runtime = ROOT / ".venv-ocr"
    return runtime / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def runtime_available():
    return runtime_python().is_file()


def worker_environment():
    """Preserve ordinary proxy/certificate settings, exclude host Python paths."""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PROMO_OCR_WORKER"] = "1"
    return environment


def _problem(code, message, adjustment, **details):
    from .ocr import OCRGuardError
    return OCRGuardError(code, message, adjustment, details)


def _invalid_response(message):
    return _problem("OCR_WORKER_PROTOCOL_ERROR", message,
                    ["重新运行 python scripts/setup_ocr.py，然后重启 ComfyUI。"])


def _error_from_response(value):
    from .ocr import OCRGuardError
    if not isinstance(value, dict):
        raise _invalid_response("OCR 子进程返回的错误格式无效。")
    code, message = value.get("code"), value.get("message")
    adjustments, details = value.get("needs_adjustment"), value.get("details", {})
    if not (isinstance(code, str) and code and isinstance(message, str) and message
            and isinstance(adjustments, list) and all(isinstance(item, str) for item in adjustments)
            and isinstance(details, dict)):
        raise _invalid_response("OCR 子进程返回的错误字段不完整。")
    return OCRGuardError(code, message, adjustments, details)


def _validate_checks(checks, mode):
    if not isinstance(checks, dict):
        raise _invalid_response("OCR 子进程没有返回逐图检查结果。")
    status = checks.get("status")
    if (status not in ("passed", "repaired") or checks.get("mode") != mode
            or checks.get("provider") != "CPUExecutionProvider"
            or checks.get("detector_version") != "1.4.4"
            or checks.get("recognition_used") is not False):
        raise _invalid_response("OCR 子进程返回的检测状态不符合约定。")
    if status == "passed" and checks.get("unexpected_count") != 0:
        raise _invalid_response("OCR 子进程仍报告了未经允许的文字。")
    if status == "repaired" and (mode != "repair_masked" or checks.get("post_repair_unexpected_count") != 0):
        raise _invalid_response("OCR 子进程没有确认修复后文字检查通过。")
    return {**checks, "execution": "isolated_python_process", "protocol": PROTOCOL}


def inspect_isolated(image, allowed_text_mask=None, cleanup_mask=None,
                     protected_mask=None, mode="reject"):
    """Send pixel files to the private interpreter and preserve all OCR failures.

    No command shell, image model, API call or runtime package installation is
    involved. Each frame is checked in one CPU process with a bounded timeout.
    """
    from .ocr import OCRGuardError, _mask_array

    if mode not in ("reject", "repair_masked"):
        raise _problem("INVALID_OCR_MODE", "背景文字检测模式无效。", ["选择 reject 或 repair_masked。"])
    if not isinstance(image, Image.Image):
        raise _problem("IMAGE_UNREADABLE", "背景图片无法读取。", ["提供可读取的背景图片。"])
    try:
        image.load()
        if min(image.size) < 4:
            raise ValueError("image too small")
    except Exception as exc:
        raise _problem("IMAGE_UNREADABLE", "背景图片无法读取或尺寸过小。",
                       ["提供可读取的背景图片。"], error_type=type(exc).__name__) from exc

    masks = {"allowed_text_mask": allowed_text_mask, "cleanup_mask": cleanup_mask,
             "protected_mask": protected_mask}
    # Validation precedes serialization; invalid floats must not turn into a valid PNG mask.
    prepared_masks = {name: _mask_array(mask, image.size, name)
                      for name, mask in masks.items() if mask is not None}
    interpreter = runtime_python()
    if not interpreter.is_file():
        raise _problem("OCR_RUNTIME_MISSING", "独立 OCR 环境尚未安装，已阻止未检查图片输出。",
                       ["在插件目录运行 python scripts/setup_ocr.py，然后重启 ComfyUI。"])

    try:
        with tempfile.TemporaryDirectory(prefix="promo-ocr-") as temporary:
            directory = Path(temporary)
            image.save(directory / "image.png", format="PNG")
            mask_files = {}
            for name, array in prepared_masks.items():
                leaf = name + ".png"
                Image.fromarray(np.asarray(array, dtype=np.uint8) * 255, "L").save(directory / leaf, format="PNG")
                mask_files[name] = leaf
            request = {"protocol": PROTOCOL, "image": "image.png", "masks": mask_files,
                       "mode": mode, "output": "output.png"}
            request_file, response_file = directory / "request.json", directory / "response.json"
            request_file.write_text(json.dumps(request), encoding="utf-8")
            completed = subprocess.run(
                [str(interpreter), str(ROOT / "promo_typesetter" / "ocr_worker.py"),
                 "--request", str(request_file), "--response", str(response_file)],
                cwd=str(ROOT), env=worker_environment(), capture_output=True, timeout=TIMEOUT_SECONDS,
                check=False,
            )
            if not response_file.is_file() or response_file.stat().st_size > MAX_RESPONSE_BYTES:
                raise _problem("OCR_WORKER_FAILED", "OCR 子进程未返回有效结果，已阻止图片输出。",
                               ["重新运行 python scripts/setup_ocr.py，并检查独立 OCR 环境。"],
                               returncode=completed.returncode)
            try:
                response = json.loads(response_file.read_text(encoding="utf-8"))
            except (ValueError, UnicodeError) as exc:
                raise _invalid_response("OCR 子进程返回的结果不是有效 JSON。") from exc
            if not isinstance(response, dict) or response.get("protocol") != PROTOCOL:
                raise _invalid_response("OCR 子进程协议版本不匹配。")
            if response.get("status") == "error":
                raise _error_from_response(response.get("error"))
            if completed.returncode != 0 or response.get("status") != "ok" or response.get("output") != "output.png":
                raise _invalid_response("OCR 子进程未确认图片处理成功。")
            checks = _validate_checks(response.get("checks"), mode)
            # The response cannot select an arbitrary local path or a symlink target.
            output_file = directory / "output.png"
            if not output_file.is_file() or output_file.is_symlink():
                raise _invalid_response("OCR 子进程没有返回指定的输出图片。")
            try:
                with Image.open(output_file) as opened:
                    if opened.format != "PNG" or opened.size != image.size or opened.mode not in ("RGB", "RGBA"):
                        raise ValueError("unexpected image format, dimensions or mode")
                    result = opened.copy()
            except (OSError, ValueError) as exc:
                raise _invalid_response("OCR 子进程输出图片无法读取或尺寸改变。") from exc
            return result, checks
    except OCRGuardError:
        raise
    except subprocess.TimeoutExpired as exc:
        raise _problem("OCR_WORKER_TIMEOUT", "独立 OCR 检查超时，已阻止图片输出。",
                       ["检查 CPU 负载或重新生成图片后重试。"], timeout_seconds=TIMEOUT_SECONDS) from exc
    except (OSError, ValueError) as exc:
        raise _problem("OCR_WORKER_FAILED", "独立 OCR 进程无法运行或交换图片。",
                       ["重新运行 python scripts/setup_ocr.py，检查磁盘空间及运行权限。"],
                       error_type=type(exc).__name__) from exc
