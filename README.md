# ComfyUI Promo Typesetter

给 Qwen 宣传图工作流增加独立文案输入、真实字体排版和逐图校验。Qwen 负责素材组合、背景和构图，指定文案在生成完成后绘制到原画布内，不再交给图像模型临摹字形。

## 工作方式

```text
宣传素材 + 视觉要求 → Qwen / KSampler 单次批量生成 1–4 张底图
指定文案 → 文案与字体校验 ──────────────────────────┐
底图 → 检查模型生成的多余文字 → 同画布透明排版 → 保存 PNG，返回结果 JSON
```

- 文案作为独立输入，保留原文、标点、换行和简繁字形，不生成、补写、翻译或改写。
- 使用随包提供的 Noto CJK 字体，绘制前检查字符覆盖和实际文字尺寸。排不下、缺字或语言要求冲突时返回调整提示。
- 在原尺寸画布内选择文字区域，使用字体、颜色、留白和对比度衔接画面；不在图片底部另接一块文字画布。
- 支持主体保护遮罩，避免文字覆盖人物、产品和品牌关键区域。
- 排版前对每张底图进行文字检测，避免指定文案之外还残留模型画出的错字。默认拒绝检测到多余文字的底图。
- 一次接收 1–4 张底图并返回逐图状态。批量出图由同一次 KSampler 执行完成，不自动循环调用单张生成。

## 安装

0.1.2 使用**独立 OCR 虚拟环境**：ComfyUI 主进程负责字体和排版，私有 CPU 工作进程在合成文字前检查底图。需要使用 **Python 3.10–3.12**，当前不支持 Python 3.13。

主环境的 `requirements.txt` 只安装字体、图片、排版和语言校验依赖，不包含 RapidOCR、OpenCV 或 ONNX Runtime，也不重新安装 PyTorch。OCR 依赖安装到插件目录下的 `.venv-ocr`，不要求降低宿主 NumPy，也不替换宿主 OpenCV 或 PyTorch。

### ComfyUI Manager

在允许 Git URL 安装的 Manager 中选择 **Install via Git URL**，粘贴以下仓库地址：

```text
https://github.com/allenflux/ComfyUI-PromoTypesetter.git
```

Git URL 安装不要求先发布到 Comfy Registry。字体随仓库安装，无需另外复制字体。安装文件后，仍需按下方步骤创建私有 OCR 环境，再重启 ComfyUI。

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

在插件目录内，使用**运行 ComfyUI 的同一个 Python 3.10–3.12**安装主进程依赖并建立 OCR 环境：

```bash
cd /path/to/ComfyUI/custom_nodes/ComfyUI-PromoTypesetter
/path/to/comfyui-python -m pip install -r requirements.txt
/path/to/comfyui-python scripts/setup_ocr.py
```

`setup_ocr.py` 在 `.venv-ocr` 内安装 `requirements-ocr.txt`。它仅在私有虚拟环境的安装子进程中停用继承的 pip 约束，避免平台的 `PIP_CONSTRAINT`（例如固定 NumPy 2.3.5）干扰 OCR 依赖解析；不修改主环境的约束文件或全局环境变量。

完成后**重启 ComfyUI**。启动后应能找到 `PromoTextPrepare` 和 `PromoRenderAndSave` 节点。原工作流中的 `ExactPromoTextPrepare` 是旧节点名称；请导入本包示例或替换为本包节点。

### OCR 环境与旧版本依赖

若返回 `OCR_RUNTIME_MISSING`，在插件目录执行 `python scripts/setup_ocr.py`（使用上述 ComfyUI Python），完成后重启。私有 OCR 运行时存在但启动或检测失败时会明确报错，不静默改用宿主 OCR，也不跳过底图检测后保存成功图片。

0.1.1 曾将 OpenCV 4.12 与所需 NumPy 版本放入主环境，在平台固定 NumPy 2.3.5 时会发生安装冲突；0.1.2 将这些 OCR 依赖移入私有环境。早期安装已经造成的宿主 `lada` / OpenCV 等冲突仍需按该环境的实际情况处理，OCR 隔离不会自动修复它们，也不代表宿主 `pip check` 已全部通过。

## 接入 Qwen 工作流

1. 将真实宣传主体素材连接到 `PromoTextPrepare.material`。可再连接 `material_2`–`material_5`，并在设计要求中说明各素材用于主体、Logo 或参考。风格参考图不能替代必需的宣传主体素材；节点检查图片可读取性，不声称能自动判定图片的业务角色。
2. 在 `copy_text` 输入完整上图文案，另外填写视觉要求、目标语言、风格和目标尺寸。
3. 将 `generation_canvas` 作为 Qwen 的尺寸画布，将 `background_prompt` 作为视觉提示，并保留原有的宣传素材连接。图像模型应预留文字空间，禁止绘制标题、标语、按钮文字或伪文字。
4. 将准备节点的 `count` 输出连接到 `RepeatLatentBatch.amount`（先把 `amount` 转为输入），从同一个数量来源设置采样批量；同一批只执行一次 KSampler。将解码后的整批图片连接到 `PromoRenderAndSave.images`。
5. 将准备节点的 `context` 连接到排版节点的 `context`，执行一次工作流。排版节点直接保存成功 PNG，返回 `result_json` 并显示成功图预览，无需另接 Save Image。每张 PNG 内嵌排版清单，不另外生成 JSON 文件。

示例还带有两个原生预览节点：`PreviewImage` 显示解码后的**未校验底图**，`PreviewAny` 显示最终 `result_json`，包含逐图状态和失败原因。底图预览不代表验收通过；只有排字输出节点的成功图片才是成品。日志出现 `Prompt executed` 只说明工作流执行完成，若全部图片被 OCR 或排版校验拒收，成功数量仍为 0，成品预览为空。添加这两个原生节点不需要重启后端。

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
| `ocr_mode` | `reject` 拒绝底图多余文字；`repair_masked` 只允许在明确的清理遮罩内处理。 |
| `allowed_text_mask` | 可选，明确允许保留原有品牌等文字的区域；不将素材文字提取为新文案。 |
| `cleanup_mask` | 可选，允许处理多余文字的区域；不允许侵入主体保护范围。 |

返回 `result_json`（STRING），并通过 ComfyUI 预览显示成功图片。

### 文字区域与遮罩

支持自动选择区域，以及顶部、左侧、底部等构图方向。`auto` 按区域纹理和对比度选择位置，不是人物或产品检测器；需要严格保护主体时应连接 `protected_mask`。文字使用原画布内的空间，长文案不会被静默删减。区域无法容纳全文时，应缩短调用方输入、调整字号、调整文字区域或增大目标尺寸后重新提交。

主体保护遮罩用于保护人物、产品、Logo 等区域。底图若检测到允许区域之外的文字，默认返回失败；只有选择 `repair_masked` 并明确提供 `cleanup_mask` 时才允许在该范围内处理。清理范围不能侵入主体保护区域，也不自动扩大到整张图。需要保留的原品牌文字可用 `allowed_text_mask` 明确标出；它不会成为指定上图文案的来源。

简体中文、繁体中文、英文以及混排均保留输入字符。语言选项用于校验和字体选择，不用于转换或翻译。例如，输入“優惠”不会被自动替换成“优惠”。

## 结果与错误

成功图片保存为 PNG，并通过 `result_json` 返回逐图结果 JSON。结果包含图片位置、宽高、格式、实际使用文案、语言、字体及状态。批量结果同时返回请求数量、成功数量和各图片的处理结果；部分成功会明确标记，失败项包含原因及建议调整内容。

以下情况不会作为成功图返回：

- 未填写指定文案、未提供必需素材，或输入无法读取。
- 文案与语言要求冲突、字体缺字，或全文无法放入可用文字区域。
- 请求的风格与视觉要求冲突，或尺寸、比例、数量不符合输入范围。
- 底图检测到未清理的多余文字，或清理会触及受保护主体。
- OCR 依赖、检测或图片保存失败。

准备阶段的输入错误，以及 Load Image 等上游节点的文件读取错误，会通过 ComfyUI 报错并中止采样，不会伪造一份成功结果。进入排版阶段后，结果 JSON 按图片分别记录成功或失败。底图实际批量与请求 `count` 不一致时返回 `BATCH_SIZE_MISMATCH`，整批失败且不保存图片。

文字层的内容来自输入字符串和真实字体，避免了图像模型生成中文笔画错误的问题。底图文字检测仍可能漏检或误检，人物身份、产品结构、Logo 和整体设计效果也需要结合输出检查；本节点不会将这些视觉判断伪装成已经通过的自动验收。

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

标准输出为 JSON；`--output` 可另存同一份 JSON，并将逐图地址补全为基于 `--base-url` 的完整地址。保留节点返回的成功、部分成功和失败状态；ComfyUI 入队校验或上游执行错误也会转换为带原因、调整项的失败结果。完全成功退出码为 0，部分成功或失败为 1。地址不能嵌入用户名或密码，脚本不提供凭据参数。

客户端测试使用网络桩验证提交次数、文案保留、错误和超时，不代表已在目标 GPU 服务器完成生成验收。

## 开发与验证

本版本的实际测试范围和未验证项见 [验证记录](VALIDATION.md)。

```bash
python -m pip install -r requirements.txt
python scripts/setup_ocr.py
python -m unittest discover -s tests -v
```

在 ComfyUI 外测试时还需安装兼容当前 Python 的 PyTorch。完整 OCR 回归需先建立私有 OCR 环境。测试覆盖文案保留、简繁英文混排、字体覆盖、区域溢出、批量状态及真实 OCR 检测；0.1.2 新增运行时隔离检查的实际结果见验证记录，不沿用旧版本通过数量作为本版本结论。

仓库中的 `pyproject.toml` 提供节点包名称和版本。若另行发布到 Comfy Registry，需要填写实际注册的 Publisher ID；本包不预填未经验证的发布者身份。

## 许可

节点代码使用 [MIT License](LICENSE)。随包 Noto 字体使用 SIL Open Font License 1.1，见 [字体说明](fonts/README.md) 及字体目录中的许可文件。Qwen、ComfyUI、OCR 和其他第三方组件遵循各自许可。
