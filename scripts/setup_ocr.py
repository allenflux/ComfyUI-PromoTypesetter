#!/usr/bin/env python3
"""Install and check OCR in this plugin's private Python environment.

Usage: the Python used by ComfyUI runs ``python scripts/setup_ocr.py``.
No package is installed into or removed from ComfyUI's Python environment.
Host runtime constraints/configuration apply to the host, so this installer
disables them only for pip inside the new private environment. Index, proxy
and certificate environment settings are retained. No TLS checks are disabled.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "1"
MARKER = "promo_ocr_runtime.json"
_PROBE = ("import json,sys,site; print(json.dumps({'prefix':sys.prefix,"
          "'base_prefix':sys.base_prefix,'user_site':site.ENABLE_USER_SITE}))")


class SetupError(RuntimeError):
    def __init__(self, code, message, details=None):
        super().__init__(message)
        self.code, self.message = code, message
        self.details = details or {}

    def as_dict(self):
        return {"status": "failed", "code": self.code, "message": self.message,
                "details": self.details}


def runtime_python(directory):
    return Path(directory) / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def private_environment(directory, environ=None):
    """Change only a copied child environment; preserve host configuration."""
    env = dict(os.environ if environ is None else environ)
    # A global NumPy pin cannot describe a separately versioned private runtime.
    # Keep PIP_BUILD_CONSTRAINT: --only-binary prevents isolated source builds.
    for name in ("PIP_CONSTRAINT", "PIP_TARGET", "PIP_PREFIX", "PIP_ROOT", "PIP_PYTHON",
                 "PYTHONHOME", "PYTHONPATH"):
        env.pop(name, None)
    env.update({"PIP_CONFIG_FILE": os.devnull, "PIP_USER": "0", "PIP_NO_DEPS": "0",
                "PIP_REQUIRE_VIRTUALENV": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                "VIRTUAL_ENV": str(directory)})
    return env


def _redacted_log(value):
    # Do not echo a private package-index URL, which may include credentials.
    return re.sub(r"(?:https?|ftp)://[^\s<>]+", "<URL hidden>", value or "")[-6000:]


def _run(runner, command, env, stage):
    try:
        result = runner([str(value) for value in command], env=env, text=True,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise SetupError("OCR_SETUP_FAILED", f"OCR 独立环境的{stage}未能启动。",
                         {"error_type": type(exc).__name__}) from exc
    if result.returncode:
        raise SetupError("OCR_SETUP_FAILED", f"OCR 独立环境的{stage}失败；未安装到 ComfyUI 主环境。",
                         {"returncode": result.returncode,
                          "log": _redacted_log((result.stdout or "") + "\n" + (result.stderr or ""))})
    return result.stdout or ""


def _json(raw, stage):
    try:
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("not an object")
        return value
    except (ValueError, TypeError) as exc:
        raise SetupError("OCR_SETUP_INVALID_RESULT", f"{stage}没有返回有效的 JSON 对象。") from exc


def setup_runtime(root=ROOT, *, python=None, version_info=None, runner=None,
                  environ=None, progress=print):
    """Return verified runtime metadata, installing exclusively into .venv-ocr."""
    version_info = sys.version_info if version_info is None else version_info
    if not (3, 10) <= tuple(version_info[:2]) < (3, 13):
        raise SetupError("OCR_SETUP_PYTHON_VERSION", "OCR 独立环境需要 Python 3.10–3.12。")
    runner = subprocess.run if runner is None else runner
    python = sys.executable if python is None else str(python)
    root = Path(root).resolve()
    directory = root / ".venv-ocr"
    requirements = root / "requirements-ocr.txt"
    worker = root / "promo_typesetter" / "ocr_worker.py"
    if not requirements.is_file() or not worker.is_file():
        raise SetupError("OCR_SETUP_FILES_MISSING", "缺少 OCR 安装或工作进程文件，请先更新完整节点仓库。")
    if directory.is_symlink():
        raise SetupError("OCR_SETUP_UNSAFE_DIRECTORY", ".venv-ocr 不能是指向其他环境的符号链接。")
    executable = runtime_python(directory)
    marker = directory / MARKER
    env = private_environment(directory, environ)
    if not executable.is_file():
        progress("创建插件专用 OCR 环境（不使用 ComfyUI 主环境的 site-packages）…")
        _run(runner, [python, "-I", "-m", "venv", directory], env, "创建")
    config = directory / "pyvenv.cfg"
    try:
        config_values = dict(line.strip().lower().split("=", 1) for line in config.read_text().splitlines()
                             if "=" in line)
        isolated = any(key.strip() == "include-system-site-packages" and value.strip() == "false"
                       for key, value in config_values.items())
    except (OSError, ValueError):
        isolated = False
    if not executable.is_file() or not isolated:
        raise SetupError("OCR_SETUP_NOT_ISOLATED", ".venv-ocr 不是有效的独立虚拟环境，已停止安装。")
    probe = _json(_run(runner, [executable, "-I", "-c", _PROBE], env, "隔离检查"), "隔离检查")
    if (Path(str(probe.get("prefix", ""))).resolve() != directory.resolve()
            or probe.get("prefix") == probe.get("base_prefix") or probe.get("user_site") is not False):
        raise SetupError("OCR_SETUP_NOT_ISOLATED", "OCR Python 没有与主环境隔离，已停止安装。")
    # An old success marker must not outlive a failed repair/install attempt.
    marker.unlink(missing_ok=True)
    progress("仅向 .venv-ocr 安装 OCR 依赖；主环境的 NumPy、OpenCV、Torch 保持原样…")
    _run(runner, [executable, "-I", "-m", "pip", "install", "--only-binary=:all:",
                  "-r", requirements], env, "依赖安装")
    _run(runner, [executable, "-I", "-m", "pip", "check"], env, "依赖检查")
    progress("检查本地 ONNX CPU 检测器和实际文字检测…")
    checked = _json(_run(runner, [executable, worker, "--self-check"], env, "检测器自检"), "检测器自检")
    if (checked.get("status") != "ok" or str(checked.get("protocol")) != PROTOCOL
            or checked.get("provider") != "CPUExecutionProvider"
            or checked.get("detector_version") != "1.4.4"
            or type(checked.get("detection_regions")) is not int or checked["detection_regions"] < 1):
        raise SetupError("OCR_SETUP_SELFCHECK_FAILED", "OCR 自检未通过，不能标记为安装成功。",
                         {key: checked.get(key) for key in ("status", "protocol", "provider", "detector_version", "detection_regions")})
    versions = checked.get("versions", {})
    numpy_version = re.match(r"^(\d+)\.(\d+)", str(versions.get("numpy", "")))
    numpy_pair = tuple(map(int, numpy_version.groups())) if numpy_version else None
    if (versions.get("opencv-python") != "4.12.0.88"
            or versions.get("rapidocr-onnxruntime") != "1.4.4"
            or numpy_pair is None or not (2, 0) <= numpy_pair < (2, 3)):
        raise SetupError("OCR_SETUP_VERSION_MISMATCH", "OCR 独立环境版本与依赖要求不一致。")
    payload = dict(checked, python=str(executable), isolated=True)
    temporary = marker.with_suffix(".json.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temporary.replace(marker)
    finally:
        temporary.unlink(missing_ok=True)
    progress("OCR 独立环境安装及自检通过。重启 ComfyUI 后使用新版工作流。")
    return payload


def main():
    try:
        setup_runtime()
    except SetupError as exc:
        print(json.dumps(exc.as_dict(), ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
