#!/usr/bin/env python3
"""Submit one native ComfyUI batch and return Promo Typesetter's result JSON.

Only Python's standard library is required. Images must already be uploaded to
the target ComfyUI server. A failed or timed-out request is never resubmitted.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import socket
import sys
import time
from urllib import error, parse, request


DEFAULT_WORKFLOW = Path(__file__).resolve().parents[1] / "examples" / "Qwen2.1_精准文案_API.json"
STYLES = ["自动", "古风典雅", "现代简约", "时尚潮流", "悬疑暗调", "科技未来"]


class ClientError(Exception):
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


def normalize_base_url(value):
    try:
        if not isinstance(value, str) or any(character.isspace() for character in value):
            raise ValueError("whitespace")
        parsed = parse.urlsplit(value)
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("credentials")
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("origin")
        if parsed.query or parsed.fragment or parsed.port == 0:
            raise ValueError("query/fragment/port")
        # Validate port syntax without ever echoing a potentially credentialed URL.
        parsed.port
    except (ValueError, TypeError):
        raise ClientError("INVALID_BASE_URL", "base-url 须为不含账号、密码、查询参数或片段的 HTTP(S) 地址。",
                          ["base-url"]) from None
    return value.rstrip("/") + "/"


def read_copy_file(path):
    try:
        # newline='' disables universal-newline conversion, retaining CRLF verbatim.
        with open(path, "r", encoding="utf-8", newline="") as handle:
            return handle.read()
    except (OSError, UnicodeError) as exc:
        raise ClientError("COPY_READ_FAILED", "无法按 UTF-8 读取指定文案文件。", ["copy-file"],
                          {"error_type": type(exc).__name__}) from exc


def _one_node(prompt, class_type):
    matches = [(str(node_id), node) for node_id, node in prompt.items()
               if isinstance(node, dict) and node.get("class_type") == class_type]
    if len(matches) != 1:
        raise ClientError("INVALID_WORKFLOW", f"API 模板必须恰好包含一个 {class_type} 节点。", ["workflow"])
    return matches[0]


def build_prompt(workflow_path, *, material, copy_text, language, allow_mixed_scripts,
                 style, layout, width, height, count, seed):
    try:
        with open(workflow_path, encoding="utf-8") as handle:
            prompt = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ClientError("WORKFLOW_READ_FAILED", "无法读取 API 工作流模板。", ["workflow"],
                          {"error_type": type(exc).__name__}) from exc
    if not isinstance(prompt, dict) or "nodes" in prompt:
        raise ClientError("INVALID_WORKFLOW", "需要 ComfyUI API 格式的节点字典，不能使用界面工作流文件。", ["workflow"])
    prompt = copy.deepcopy(prompt)
    prepare_id, prepare = _one_node(prompt, "PromoTextPrepare")
    render_id, _ = _one_node(prompt, "PromoRenderAndSave")
    inputs = prepare.setdefault("inputs", {})
    material_link = inputs.get("material")
    if not (isinstance(material_link, list) and len(material_link) == 2
            and str(material_link[0]) in prompt
            and prompt[str(material_link[0])].get("class_type") == "LoadImage"):
        raise ClientError("INVALID_WORKFLOW", "material 必须连接 API 模板中的 LoadImage。", ["workflow", "material"])
    if (not isinstance(material, str) or not material.strip() or "\x00" in material
            or "://" in material or material.startswith(("/", "\\"))
            or ".." in material.replace("\\", "/").split("/")):
        raise ClientError("INVALID_MATERIAL_NAME", "material 须为已上传到 ComfyUI 的文件名或相对路径，不是本地路径或 URL。",
                          ["material"])
    prompt[str(material_link[0])]["inputs"]["image"] = material
    copy_link = inputs.get("copy_text")
    if isinstance(copy_link, list):
        source = prompt.get(str(copy_link[0])) if len(copy_link) == 2 else None
        if not isinstance(source, dict) or source.get("class_type") != "PrimitiveStringMultiline":
            raise ClientError("INVALID_WORKFLOW", "连接的文案输入必须来自 PrimitiveStringMultiline。", ["workflow"])
        source.setdefault("inputs", {})["value"] = copy_text
    else:
        inputs["copy_text"] = copy_text
    inputs.update(language=language, allow_mixed_scripts=allow_mixed_scripts,
                  style=style, layout=layout, width=width, height=height, count=count)
    repeats = [node for node in prompt.values() if isinstance(node, dict)
               and node.get("class_type") == "RepeatLatentBatch"]
    if len(repeats) != 1 or repeats[0].get("inputs", {}).get("amount") != [prepare_id, 3]:
        raise ClientError("INVALID_WORKFLOW", "RepeatLatentBatch.amount 必须连接准备节点的 count 输出，保持数量单一来源。",
                          ["workflow"])
    samplers = [node for node in prompt.values() if isinstance(node, dict)
                and node.get("class_type") in ("KSampler", "KSamplerAdvanced")]
    if len(samplers) != 1:
        raise ClientError("INVALID_WORKFLOW", "API 模板必须包含一次 KSampler 采样。", ["workflow"])
    key = "noise_seed" if samplers[0]["class_type"] == "KSamplerAdvanced" else "seed"
    samplers[0].setdefault("inputs", {})[key] = seed
    return prompt, render_id


def failure_result(exc, count, copy_text, language, prompt_id=None):
    reason = exc.as_dict()
    payload = {"schema_version": "0.1.0", "status": "failed", "requested_count": count,
               "success_count": 0, "failed_count": count, "copy_text": copy_text,
               "language": language, "error": reason,
               "images": [{"index": index, "status": "failed", "url": None,
                           "copy_text": copy_text, "language": language, "error": reason}
                          for index in range(count)]}
    if prompt_id:
        payload["prompt_id"] = prompt_id
    return payload


def _queue_error(body, status=None):
    details = {key: body[key] for key in ("error", "node_errors") if body.get(key)}
    if status:
        details["http_status"] = status
    messages = []
    top = body.get("error")
    if isinstance(top, dict) and top.get("message"):
        messages.append(str(top["message"]))
    elif isinstance(top, str):
        messages.append(top)
    node_errors = body.get("node_errors") or {}
    for node_id, node in (node_errors.items() if isinstance(node_errors, dict) else []):
        if not isinstance(node, dict):
            continue
        for item in node.get("errors", []):
            if isinstance(item, dict):
                explanation = ": ".join(str(item[key]) for key in ("message", "details") if item.get(key))
                if explanation:
                    messages.append(f"节点 {node_id}: {explanation}")
    return ClientError("COMFY_QUEUE_REJECTED", "ComfyUI 未接受工作流。" + "；".join(messages),
                       ["material", "workflow", "请求参数"], details)


def _http_json(opener, url, deadline, clock, payload=None):
    remaining = deadline - clock()
    if remaining <= 0:
        raise ClientError("TIMEOUT", "等待 ComfyUI 结果超时；未重新提交，原任务可能仍在服务器运行。", ["timeout", "检查服务器队列"])
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"} if data is not None else {},
                          method="POST" if data is not None else "GET")
    try:
        with opener(req, timeout=remaining) as response:
            raw = response.read()
    except error.HTTPError as exc:
        try:
            body = json.loads(exc.read().decode("utf-8"))
        except (ValueError, UnicodeError):
            body = {}
        if data is not None:
            raise _queue_error(body if isinstance(body, dict) else {}, exc.code) from None
        raise ClientError("COMFY_HTTP_ERROR", "读取 ComfyUI 历史结果失败，未重新提交。", ["base-url", "检查服务器"],
                          {"http_status": exc.code}) from None
    except (TimeoutError, socket.timeout):
        raise ClientError("TIMEOUT", "ComfyUI 网络请求超时；未重新提交，原任务可能仍在服务器运行。",
                          ["timeout", "检查服务器队列"]) from None
    except (error.URLError, OSError) as exc:
        if isinstance(getattr(exc, "reason", None), (TimeoutError, socket.timeout)):
            raise ClientError("TIMEOUT", "ComfyUI 网络请求超时；未重新提交。", ["timeout", "检查服务器队列"]) from None
        raise ClientError("COMFY_CONNECTION_ERROR", "无法连接 ComfyUI；未重试或重新提交。", ["base-url", "检查服务器"],
                          {"error_type": type(exc).__name__}) from None
    try:
        result = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeError):
        raise ClientError("INVALID_COMFY_RESPONSE", "ComfyUI 返回的内容不是有效 JSON。", ["base-url", "检查服务器"])
    if not isinstance(result, dict):
        raise ClientError("INVALID_COMFY_RESPONSE", "ComfyUI 返回的 JSON 不是对象。", ["检查服务器"])
    return result


def _execution_error(record):
    status = record.get("status") or {}
    if not isinstance(status, dict):
        return ClientError("INVALID_COMFY_RESPONSE", "ComfyUI 历史状态格式无效。", ["检查服务器"])
    for entry in status.get("messages", []):
        if not isinstance(entry, (list, tuple)) or len(entry) != 2 or entry[0] not in ("execution_error", "execution_interrupted"):
            continue
        item = entry[1] if isinstance(entry[1], dict) else {}
        message = str(item.get("exception_message") or "上游执行中断。")
        fields = ["检查失败节点及输入参数"]
        details = {key: item[key] for key in ("node_id", "node_type", "exception_type") if key in item}
        try:
            structured = json.loads(message)
        except ValueError:
            structured = None
        if isinstance(structured, dict):
            message = str(structured.get("message", message))
            fields = structured.get("needs_adjustment") or fields
            details["upstream_error"] = structured
        return ClientError("COMFY_EXECUTION_ERROR", "ComfyUI 上游执行失败：" + message[:4000], fields, details)
    if status.get("status_str") == "error":
        return ClientError("COMFY_EXECUTION_ERROR", "ComfyUI 上游执行失败，但未返回具体异常。", ["查看服务器日志"])
    return None


def _extract_result(record, render_id, base_url):
    outputs = record.get("outputs") or {}
    output = outputs.get(render_id) if isinstance(outputs, dict) else None
    if not isinstance(output, dict):
        return None
    for text in output.get("text", []):
        if not isinstance(text, str):
            continue
        try:
            result = json.loads(text)
        except ValueError:
            continue
        if not isinstance(result, dict) or not isinstance(result.get("images"), list):
            continue
        result = copy.deepcopy(result)
        for image in result["images"]:
            if isinstance(image, dict) and image.get("url"):
                image["url"] = parse.urljoin(base_url, str(image["url"]).lstrip("/"))
        return result
    return None


def request_generation(*, base_url, material, copy_text, language="zh-Hans",
                       allow_mixed_scripts=False, style="自动", layout="auto",
                       width=1152, height=1536, count=1, seed=0, timeout=600,
                       workflow_path=DEFAULT_WORKFLOW, opener=None,
                       clock=time.monotonic, sleeper=time.sleep, poll_interval=0.5):
    """Return a result dict; at most one POST is made for the entire batch."""
    prompt_id = None
    try:
        if type(count) is not int or count not in (1, 2, 3, 4):
            raise ClientError("INVALID_COUNT", "count 必须是 1–4。", ["count"])
        if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ClientError("INVALID_TIMEOUT", "timeout 必须为正数。", ["timeout"])
        base_url = normalize_base_url(base_url)
        prompt, render_id = build_prompt(workflow_path, material=material, copy_text=copy_text,
                                         language=language, allow_mixed_scripts=allow_mixed_scripts,
                                         style=style, layout=layout, width=width, height=height,
                                         count=count, seed=seed)
        opener = opener or request.urlopen
        deadline = clock() + timeout
        queue = _http_json(opener, parse.urljoin(base_url, "prompt"), deadline, clock, {"prompt": prompt})
        if queue.get("error") or queue.get("node_errors") or not queue.get("prompt_id"):
            raise _queue_error(queue)
        prompt_id = str(queue["prompt_id"])
        history_url = parse.urljoin(base_url, "history/" + parse.quote(prompt_id, safe=""))
        while True:
            history = _http_json(opener, history_url, deadline, clock)
            record = history.get(prompt_id)
            if isinstance(record, dict):
                failure = _execution_error(record)
                if failure:
                    raise failure
                result = _extract_result(record, render_id, base_url)
                if result is not None:
                    result["prompt_id"] = prompt_id
                    return result
                if (record.get("status") or {}).get("completed"):
                    raise ClientError("RESULT_MISSING", "工作流已完成，但没有返回 PromoRenderAndSave 的结果 JSON。",
                                      ["workflow", "检查输出节点"])
            remaining = deadline - clock()
            if remaining <= 0:
                raise ClientError("TIMEOUT", "等待 ComfyUI 结果超时；未重新提交，原任务可能仍在服务器运行。",
                                  ["timeout", "检查服务器队列"])
            sleeper(min(poll_interval, remaining))
    except ClientError as exc:
        safe_count = count if type(count) is int and count in (1, 2, 3, 4) else 0
        return failure_result(exc, safe_count, copy_text, language, prompt_id)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="ComfyUI HTTP(S) 地址，不含凭据")
    parser.add_argument("--material", required=True, help="已上传到服务器的 ComfyUI 文件名；不执行上传")
    parser.add_argument("--copy-file", required=True, help="UTF-8 完整文案文件，保留 CRLF 与原文")
    parser.add_argument("--language", choices=["zh-Hans", "zh-Hant", "en"], default="zh-Hans")
    parser.add_argument("--allow-mixed-scripts", action="store_true")
    parser.add_argument("--style", choices=STYLES, default="自动")
    parser.add_argument("--layout", choices=["auto", "top", "left", "bottom"], default="auto")
    parser.add_argument("--width", type=int, default=1152)
    parser.add_argument("--height", type=int, default=1536)
    parser.add_argument("--count", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=600, help="总等待秒数；超时不重提")
    parser.add_argument("--output", type=Path, help="可选，将返回 JSON 另存为文件")
    args = parser.parse_args(argv)
    try:
        copy_text = read_copy_file(args.copy_file)
    except ClientError as exc:
        result = failure_result(exc, args.count, None, args.language)
    else:
        result = request_generation(base_url=args.base_url, material=args.material, copy_text=copy_text,
                                    language=args.language, allow_mixed_scripts=args.allow_mixed_scripts,
                                    style=args.style, layout=args.layout, width=args.width,
                                    height=args.height, count=args.count, seed=args.seed, timeout=args.timeout)
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + "\n", encoding="utf-8")
        except OSError as exc:
            result = {"status": "failed", "error": {"code": "RESULT_WRITE_FAILED",
                       "message": "生成结果已返回，但本地 JSON 文件保存失败；未重新提交。",
                       "needs_adjustment": ["output"], "error_type": type(exc).__name__},
                      "generation_result": result}
            encoded = json.dumps(result, ensure_ascii=False, indent=2)
    print(encoded)
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
