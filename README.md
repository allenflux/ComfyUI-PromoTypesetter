# ComfyUI Promo Typesetter

给 Qwen 宣传图工作流增加独立文案输入、自动选择文字区域和真实字体排版。Qwen 负责素材组合、无字背景和构图，指定文案在生成完成后绘制到原画布内。**默认不运行 OCR，也不需要安装 OCR 环境。**

## 工作方式

```text
宣传素材 + 视觉要求 → Qwen / KSampler 单次批量生成 1–4 张无字底图
指定文案 → 文案与字体校验 ──────────────────────────┐
底图 → 按留白、纹理和对比度选文字区 → 同画布透明排版 → 保存、预览 PNG
```

- 文案作为独立输入，保留原文、标点、换行和简繁字形，不生成、补写、翻译或改写。
- 使用随包提供的 Noto CJK 字体，绘制前检查字符覆盖和实际文字尺寸。排不下、缺字或语言要求冲突时返回调整提示。
- 在原尺寸画布内按留白、纹理和对比度选择文字区域，使用字体和颜色衔接画面；不在图片底部另接一块文字画布。
- 支持主体保护遮罩，避免文字覆盖人物、产品和品牌关键区域。
- 默认 `ocr_mode=off`，**有有效底图就排字、保存并返回图片**。OCR 仅作为可选高级检查；主动启用后，默认也只附复核提醒，不拦截图片。
- 一次接收 1–4 张底图并返回逐图状态。批量出图由同一次 KSampler 执行完成，不自动循环调用单张生成。

## 安装

默认工作流只需要主环境的 `requirements.txt`：字体、图片、排版和语言校验依赖，不包含 RapidOCR、OpenCV 或 ONNX Runtime，也不重新安装 PyTorch。不需要执行 `setup_ocr.py`，不要求降低宿主 NumPy 或替换宿主 OpenCV。已验证的运行环境为 **Python 3.10–3.12**。

### ComfyUI Manager

在允许 Git URL 安装的 Manager 中选择 **Install via Git URL**，粘贴以下仓库地址：

```text
https://github.com/allenflux/ComfyUI-PromoTypesetter.git
```

Git URL 安装不要求先发布到 Comfy Registry。字体随仓库安装，无需另外复制字体。安装依赖后重载 ComfyUI 即可使用默认工作流。

### 从目录或 Git 安装

将完整的 `ComfyUI-PromoTypesetter` 文件夹放入 `ComfyUI/custom_nodes/`；也可以在该目录下克隆仓库：

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/allenflux/ComfyUI-PromoTypesetter.git
```

已有安装先更新文件：

```bash
cd /path/to/ComfyUI/custom_nodes/ComfyUI-PromoTypesetter
git pull --ff-only
```

在插件目录内，使用**运行 ComfyUI 的同一个 Python**安装主进程依赖：

```bash
cd /path/to/ComfyUI/custom_nodes/ComfyUI-PromoTypesetter
/path/to/comfyui-python -m pip install -r requirements.txt
```

完成后**重启 ComfyUI**。启动后应能找到 `PromoTextPrepare` 和 `PromoRenderAndSave` 节点。原工作流中的 `ExactPromoTextPrepare` 是旧节点名称；请导入本包示例或替换为本包节点。

### 可选高级 OCR 检查与旧版本依赖

仅在需要主动启用 `ocr_mode=reject` 或 `repair_masked` 时，才使用 Python 3.10–3.12 创建独立 OCR 环境：

```bash
/path/to/comfyui-python scripts/setup_ocr.py
```

`setup_ocr.py` 在插件内的 `.venv-ocr` 安装 `requirements-ocr.txt`，只对该私有安装子进程停用继承的 pip 约束，不修改主环境约束。默认 `ocr_mode=off` 完全不启动这个工作进程。

主动开启 OCR 后，检测器故障或疑似文字默认随图片返回 `review_required` 提醒；不会把未完成的检测标成通过，也不会静默改用宿主 OCR。只有同时选择 `quality_policy=strict` 时，检查失败才拒绝该张输出。`OCR_RUNTIME_MISSING` 表示尚未建立上述可选环境。

0.1.1 曾将 OpenCV 4.12 与所需 NumPy 版本放入主环境，在平台固定 NumPy 2.3.5 时会发生安装冲突；0.1.2 将这些 OCR 依赖移入私有环境。早期安装已经造成的宿主 `lada` / OpenCV 等冲突仍需按该环境的实际情况处理，OCR 隔离不会自动修复它们，也不代表宿主 `pip check` 已全部通过。

## 接入 Qwen 工作流

1. 将真实宣传主体素材连接到 `PromoTextPrepare.material`。可再连接 `material_2`–`material_5`，并在设计要求中说明各素材用于主体、Logo 或参考。风格参考图不能替代必需的宣传主体素材；节点检查图片可读取性，不声称能自动判定图片的业务角色。
2. 在 `copy_text` 输入完整上图文案，另外填写视觉要求、目标语言、风格和目标尺寸。
3. 将 `generation_canvas` 作为 Qwen 的尺寸画布，将 `background_prompt` 作为视觉提示，并保留原有的宣传素材连接。图像模型应预留文字空间，禁止绘制标题、标语、按钮文字或伪文字。
4. 将准备节点的 `count` 输出连接到 `RepeatLatentBatch.amount`（先把 `amount` 转为输入），从同一个数量来源设置采样批量；同一批只执行一次 KSampler。将解码后的整批图片连接到 `PromoRenderAndSave.images`。
5. 将准备节点的 `context` 连接到排版节点的 `context`，执行一次工作流。排版节点默认继续处理每张有效底图，直接保存 PNG、返回 `result_json` 并显示图片预览，无需另接 Save Image。每张 PNG 内嵌排版清单，不另外生成 JSON 文件。

示例还带有两个原生预览节点：`PreviewImage` 显示解码后的**未校验底图**，`PreviewAny` 显示最终 `result_json`，包含逐图状态、复核提醒和失败原因。最终排字节点同时预览已保存的图片。默认 `ocr_mode=off`、`quality_policy=return_with_warnings`，直接选择文字区并排字。`Prompt executed` 表示流程结束，实际出图数量应查看 `success_count` 和逐图状态。添加原生预览节点本身不需要重启后端；更新 Python 节点代码后需要重载 ComfyUI。

指定文案不应再次拼入 Qwen 提示词。排版完成的图片直接保存，不再经过 Qwen 或扩散采样，否则字形仍可能被重画。

### 输入范围

支持 1–5 张素材，每个素材接口只接受单张图片（batch = 1）。建议素材短边至少 **512 像素**；低于 512 会给出清晰度提醒，短边低于 128 会拒绝。

目标尺寸支持以下五种精确比例，宽高均须为 **32 的倍数**，长边须在 **1024–2048 像素**之间。默认尺寸为 1152 × 1536（3:4），宽高直接作为最终画布尺寸，不通过拉伸、裁切或缩放适配。

| 比例 | 可用尺寸示例 |
| --- | --- |
| 1:1 | 1024 × 1024 |
| 3:4 | 1152 × 1536 |
| 4:3 | 1536 × 1152 |
| 9:16 | 1152 × 2048 |
| 16:9 | 2048 × 1152 |

若通过上传服务或 API 接入，单张素材 **20 MB** 的文件限制应在解码前执行。ComfyUI `IMAGE` 输入已经是解码后的像素张量，本节点不将其内存大小冒充原文件大小，也不声称已校验原上传文件的字节数。

### 节点参数

`PromoTextPrepare`：

| 参数 | 用途 |
| --- | --- |
| `material` | 必需的宣传主体图片；额外图片接 `material_2`–`material_5`。 |
| `copy_text` | 完整上图文案，多行输入；第一行可作为标题。 |
| `language` | `zh-Hans` 简体中文、`zh-Hant` 繁体中文、`en` 英文。 |
| `allow_mixed_scripts` | 允许保留简繁与英文混排；不会转换输入字符。 |
| `design_prompt` | 只描述素材用途、构图、背景、配色等视觉要求。 |
| `style` | `自动`（不预选风格）、`古风典雅`、`现代简约`、`时尚潮流`、`悬疑暗调`、`科技未来`。明确的风格冲突返回调整提示。 |
| `layout` | `auto`、`top`、`left`、`bottom`。 |
| `width`、`height` | 最终图片宽高，符合上方比例与长边范围。 |
| `count` | 请求数量 1–4；通过节点输出连接到 `RepeatLatentBatch.amount`。 |
| `first_line_title` | 是否将原文第一行按标题字号绘制。 |
| `title_size`、`body_size` | 标题与正文字号；放不下时明确报错，不自动改写文案。 |
| `font_family` | `auto` 按风格选择、`sans` 无衬线、`serif` 衬线。 |
| `text_color` | `auto` 按区域对比度选择，或填写颜色值。 |
| `protected_mask` | 可选，保护主体的遮罩，须与最终画布尺寸一致。 |

输出为 `generation_canvas`（IMAGE）、`context`（PROMO_CONTEXT）、`background_prompt`（STRING）、`count`（INT）。

`PromoRenderAndSave`：

| 参数 | 用途 |
| --- | --- |
| `images` | 同次采样并解码后的整批底图。 |
| `context` | 准备节点输出的文案、字体、尺寸与排版上下文。 |
| `filename_prefix` | 输出文件名前缀，默认为 `PromoTypesetter`。 |
| `ocr_mode` | 默认 `off`，不运行 OCR；高级选项 `reject` 检测底图疑似文字，`repair_masked` 只在明确的清理遮罩内处理。 |
| `quality_policy` | 默认 `return_with_warnings`：有效底图继续排字、预览和保存，检测或可读性问题附待复核提醒；`strict`：检查未通过时拒绝该张输出。 |
| `allowed_text_mask` | 可选，明确允许保留原有品牌等文字的区域；不将素材文字提取为新文案。 |
| `cleanup_mask` | 可选，允许处理多余文字的区域；不允许侵入主体保护范围。 |

返回 `result_json`（STRING），并通过 ComfyUI 预览显示已保存的图片；待复核图片同样有图片地址和预览。

### 文字区域与遮罩

支持自动选择区域，以及顶部、左侧、底部等构图方向。Qwen 提示词要求预留干净的文字空间，`auto` 再按候选区域的留白、纹理和对比度选择位置；该选择不识别人物或产品身份，需要严格保护主体时应连接 `protected_mask`。文字使用原画布内的空间，长文案不会被静默删减。区域无法容纳全文时，应缩短调用方输入、调整字号、调整文字区域或增大目标尺寸后重新提交。

主体保护遮罩用于保护人物、产品、Logo 等区域。默认流程不检测或清理底图文字，直接在选定区域排入指定原文。主动启用高级 OCR 检查后，允许区域之外的疑似文字会列入 `warnings`，默认保留图片并标为 `review_required`。只有选择 `repair_masked` 并明确提供 `cleanup_mask` 时才尝试在该范围内清理；清理失败时，默认仍使用原底图并附上提醒。清理范围不能侵入主体保护区域，也不自动扩大到整张图。需要保留的原品牌文字可用 `allowed_text_mask` 明确标出；它不会成为指定上图文案的来源。`strict` 策略仅在主动选择时拒收。

简体中文、繁体中文、英文以及混排均保留输入字符。语言选项用于校验和字体选择，不用于转换或翻译。例如，输入“優惠”不会被自动替换成“优惠”。

## 结果与错误

已生成图片保存为 PNG，并通过 `result_json` 返回逐图结果 JSON。每张包括图片地址、宽高、格式、实际使用文案、语言、字体及状态；待复核图片也包含这些信息。默认 OCR 状态明确记为关闭，不写成检测通过。启用高级检查后的提醒与详情同样保留在结果中。

- `success_count`：实际保存并返回的图片数量，包含待复核图片。
- `passed_count`：自动检查通过的图片数量。
- `review_count`：已经返回图片、需要人工复核的数量。
- `failed_count`：未能生成或保存图片的数量。

例如，请求 4 张且均完成保存时，`success_count=4`、`failed_count=0`。若主动启用 OCR 且四张都有提醒，仍返回 4 张图片，同时 `review_count=4`，总体与逐图状态为 `review_required`。只有部分图片未能输出时才标记 `partial_success`；失败项保留原因及调整信息。

以下情况仍需要调整输入或修复运行环境：

- 未填写指定文案、未提供必需素材，或输入无法读取。
- 文案与语言要求冲突、字体缺字，或全文无法放入可用文字区域。
- 请求的风格与视觉要求冲突，或尺寸、比例、数量不符合输入范围。
- 主动启用 OCR 并选择 `strict` 策略时，底图检测到未清理的多余文字，或清理会触及受保护主体。
- 图片生成、排版或保存失败。

“每次有图”的处理范围是 Qwen 已生成、可读取且符合尺寸的有效底图；缺失素材或文案等输入错误仍会在采样前提示调整。默认直接排字和保存，不经过 OCR 拦截。结果 JSON 按图片记录已返回、待复核或失败。底图实际批量或尺寸与请求不一致时仍返回明确错误。

文字层的内容来自输入字符串和真实字体，避免了图像模型生成中文笔画错误的问题。默认依靠无字底图提示词，不运行 OCR。模型偶发生成的底图伪文字、人物身份、产品结构、Logo 和整体设计效果仍需结合图片检查；本节点不会将这些视觉判断伪装成已经通过的自动验收。

## 一次 API 调用生成 1–4 张

提供标准库客户端 [scripts/call_api.py](scripts/call_api.py)，默认读取 [Qwen2.1_精准文案_API.json](examples/Qwen2.1_精准文案_API.json)。先将素材上传到目标 ComfyUI，并将完整文案保存为 UTF-8 文件，然后执行：

```bash
python scripts/call_api.py \
  --base-url http://127.0.0.1:8188 \
  --material product.png \
  --copy-file copy.txt \
  --language zh-Hans \
  --style 现代简约 \
  --layout auto \
  --width 1152 --height 1536 \
  --count 4 --seed 2026 \
  --timeout 600 \
  --output result.json
```

`--material` 是已经上传到服务器的文件名，不是本地图片路径；此脚本不上传文件。`--copy-file` 保留原始 UTF-8 文案，包括 CRLF 换行，不做删减、改写、翻译或简繁转换。混排时添加 `--allow-mixed-scripts`。

一次运行只发送 **一次 `POST /prompt`**，由工作流完成整个批量，随后通过 `GET /history/{prompt_id}` 查询结果。请求数量 1、2、3、4 均使用相同机制。脚本不重试提交、不自动补失败图片；超时也不会取消服务器上的原任务，应先核对返回的 `prompt_id` 和服务器队列。

标准输出为 JSON；`--output` 可另存同一份 JSON，并将逐图地址补全为基于 `--base-url` 的完整地址。保留节点返回的成功、待复核、部分成功和失败状态；ComfyUI 入队校验或上游执行错误也会转换为带原因、调整项的失败结果。请以返回图片数量及逐图状态判断结果，待复核图片不等于未出图。地址不能嵌入用户名或密码，脚本不提供凭据参数。

客户端测试使用网络桩验证提交次数、文案保留、错误和超时，不代表已在目标 GPU 服务器完成生成验收。

## 开发与验证

本版本的实际测试范围和未验证项见 [验证记录](VALIDATION.md)。

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

在 ComfyUI 外测试时还需安装兼容当前 Python 的 PyTorch。可选 OCR 的完整回归需先执行 `python scripts/setup_ocr.py` 建立私有环境。测试覆盖默认无 OCR 排版、文案保留、简繁英文混排、字体覆盖、区域溢出、批量状态及可选 OCR 检测；各版本的实际验证结果见验证记录，不沿用旧版本通过数量作为新版本结论。

仓库中的 `pyproject.toml` 提供节点包名称和版本。若另行发布到 Comfy Registry，需要填写实际注册的 Publisher ID；本包不预填未经验证的发布者身份。

## 许可

节点代码使用 [MIT License](LICENSE)。随包 Noto 字体使用 SIL Open Font License 1.1，见 [字体说明](fonts/README.md) 及字体目录中的许可文件。Qwen、ComfyUI、OCR 和其他第三方组件遵循各自许可。
