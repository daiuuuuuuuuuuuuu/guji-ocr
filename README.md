# 古籍识别项目 — 三 Agent 流水线

> **English**: A 3-agent pipeline for OCR and text correction of classical Chinese texts. Input PDFs or images of ancient Chinese books, and get clean, corrected text output. Uses Qwen3.7-Plus (multimodal vision) for OCR recognition and DeepSeek-V4-Flash for page reordering and typo correction. Total cost is approximately ¥0.016 per page.
> 
> **中文**: 古籍 PDF/图片 → OCR 识别 → 页序重排 → 错别字纠正 → 干净文本输出

---

## 目录

- [项目概览](#项目概览)
- [完整使用流程](#完整使用流程)
- [安装](#安装)
- [傻瓜版操作指南](#傻瓜版操作指南不需要懂编程照着做就行) ← 不看别的也请看这个
  - [你需要准备什么](#你需要准备什么)
  - [第一步：配置 API Key](#第一步配置-api-key1-分钟)
  - [第二步：安装](#第二步安装1-分钟第一次用才做)
  - [第三步：放文件](#第三步放文件30-秒)
  - [第四步：跑](#第四步跑可能很久看页数)
  - [第五步：看结果](#第五步看结果)
  - [常见问题](#常见问题)
- [Agent 详解](#agent-详解)
  - [agent_qw — OCR 识别](#1-agent_qw--ocr-识别)
  - [agent_verify — 页序重排](#2-agent_verify--页序重排)
  - [agent_c — 错别字纠正](#3-agent_c--错别字纠正)
- [配置](#配置)
  - [配置优先级](#配置优先级agent_verify-和-agent_c)
  - [各 Agent 使用的模型](#各-agent-使用的模型)
  - [完整配置示例](#完整配置示例)
- [目录结构](#目录结构)
- [测试](#测试)

---

## 项目概览 / Project Overview

A 3-agent sequential pipeline that turns raw page images into clean, corrected classical Chinese text.

项目包含三个独立的 Agent，按顺序组成处理流水线：

```
古籍图片/PDF
    │
    ▼
┌──────────────────────────────────────────────┐
│  agent_qw (OCR 识别)                          │
│  输入: PDF/图片文件或目录                       │
│  输出: out/<书名>/{txt,json}/       │
│  模型: Qwen3.7-Plus (多模态视觉)                │
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│  agent_verify (页序重排)                       │
│  输入: out/<书名>/txt/page_*.txt               │
│  输出: out_v/<书名>/{txt,json}/     │
│  模型: DeepSeek-V4-Flash                       │
│  原理: LLM 判断页面首尾文字连贯性               │
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│  agent_c (错别字纠正)                          │
│  输入: out_v/<书名>/txt/page_*.txt             │
│  输出: out_c/<书名>/{txt,json}/     │
│  模型: DeepSeek-V4-Flash                       │
│  原理: LLM 校勘 OCR 常见形近字错误             │
└──────────────────────────────────────────────┘
```

---

## 完整使用流程 / Complete Workflow

Run the three agents in order. Each step feeds its output into the next.

```bash
# 激活虚拟环境
.venv\Scripts\activate

# 第一步：OCR 识别（PDF/图片 → 文本）
python -m agent_qw.cli run -i 古籍图片/ -o out

# 第二步：页序重排（LLM 判断文字连贯性，纠正页码错乱）
python -m agent_verify.cli verify -i out -o out_v

# 第三步：错别字纠正（LLM 校勘 OCR 识别错误）
python -m agent_c.cli correct -i out_v -o out_c
```

最终干净文本在 `out_c/<书名>/txt/` 目录中。

---

## 安装 / Installation

Create a virtual environment and install in editable mode.

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -e .
```

---

## 傻瓜版操作指南（不需要懂编程，照着做就行）

> 每页成本约 **1.6 分钱**，一本 500 页的古籍跑完全部流程约 **8 元**。

---

### 你需要准备什么

| 序号 | 准备项 | 说明 |
|:----:|--------|------|
| 1 | 一台 Windows 电脑 | 已装 Python 3.8+（没有的话去 [python.org](https://python.org) 下载） |
| 2 | 一个 **DashScope API Key** | 去 [阿里云百炼](https://bailian.console.aliyun.com/) 注册 → 开通 DashScope → 创建 API Key |
| 3 | 一个 **DeepSeek API Key** | 去 [DeepSeek 开放平台](https://platform.deepseek.com/) 注册 → API Keys → 创建 |
| 4 | 你的古籍 PDF 或图片 | 支持 `.pdf` `.png` `.jpg` `.tiff` `.bmp` |

> **两把 Key 的区别**：OCR 识别（看图识字）用阿里云的 Qwen 模型，必须用 DashScope Key；页序重排和错别字纠正用 DeepSeek 模型，必须用 DeepSeek Key。

---

### 第一步：配置 API Key（1 分钟）

打开项目里的 `config.yaml`，找到这三行，把等号右边的 Key 换掉：

```yaml
# 找到 llm 这一段，把 api_key 换成你的 DashScope Key ↓
llm:
  api_key: "sk-你的DashScope的Key"        # ← 改这里！

# 找到 agent_verify 这一段，换成你的 DeepSeek Key ↓
agent_verify:
  api_key: "sk-你的DeepSeek的Key"         # ← 改这里！

# 找到 agent_c 这一段，换成你的 DeepSeek Key ↓
agent_c:
  api_key: "sk-你的DeepSeek的Key"         # ← 改这里！(和上面同一个就行)
```

改完保存，其他配置**都不用动**。

> 不会找 `config.yaml`？它就在你下载的项目文件夹的根目录，和 `README.md` 同一个目录。

---

### 第二步：安装（1 分钟，第一次用才做）

打开 **PowerShell**（右键开始菜单 → Windows PowerShell），复制下面一整段，回车：

```powershell
cd "E:\005 ai\zuopinji_web\project_a\7_古籍识别\7.项目资产"
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

看到 `Successfully installed` 就说明装好了。

> **之后每次打开 PowerShell 要跑项目，只需要先输入这一行**：
> ```powershell
> .venv\Scripts\activate
> ```

---

### 第三步：放文件（30 秒）

把你的古籍 PDF 或图片丢进 **`古籍图片/`** 文件夹。

```
古籍图片/
├── 论语卷一.pdf
├── 论语卷二.pdf
└── 某页手稿.png
```

> 如果没有这个文件夹，自己建一个，就叫 `古籍图片`。

---

### 第四步：跑（可能很久，看页数）

确保已经执行过 `.venv\Scripts\activate`，然后一行一行来：

```powershell
# 第一步：OCR 识别（把图片变成文字）
python -m agent_qw.cli run -i 古籍图片/ -o out

# 第二步：页序重排（修正页码错乱）
python -m agent_verify.cli verify -i out -o out_v

# 第三步：错别字纠正（校对 OCR 识别错误）
python -m agent_c.cli correct -i out_v -o out_c
```

**每步跑完会打印总结**，像这样：

```
==================================================
  古籍识别完成
==================================================
  处理页数: 50
  输出目录: E:\...\out
==================================================
```

如果出现 `Error` 就看下面的常见问题。出现 `401 / invalid_api_key` 说明 Key 没写对，回去检查第一步。

---

### 第五步：看结果

打开 `out_c` 文件夹，找到你的书名，进去 `txt` 目录：

```
out_c/
└── 论语卷一/
    └── txt/
        ├── page_001.txt    ← 第一页，双击用记事本打开就能读
        ├── page_002.txt
        └── page_003.txt
```

每个 `.txt` 就是对应页面的最终干净文本。

---

### 常见问题

<details>
<summary><b>Q: 提示 "No API key configured"？</b></summary>

→ `config.yaml` 里 `api_key` 没填或填错了。回去看第一步。
</details>

<details>
<summary><b>Q: 提示 401 / invalid_api_key？</b></summary>

→ Key 错了。检查两点：
1. `llm` 段的 Key 是不是 DashScope 的（不是 DeepSeek 的）
2. `agent_verify` 和 `agent_c` 的 Key 是不是 DeepSeek 的（不是 DashScope 的）
</details>

<details>
<summary><b>Q: OCR 一页都没识别出来？</b></summary>

→ 检查 DashScope 账户里有没有余额（[控制台](https://bailian.console.aliyun.com/) 可查）。新用户有免费额度，但用完了要充值。
</details>

<details>
<summary><b>Q: 跑一半报网络错误？</b></summary>

→ 再跑一次，项目已内置自动重试（3 次）。如果连续失败可能是网络代理问题。
</details>

<details>
<summary><b>Q: 能只跑其中一步吗？</b></summary>

→ 能。三步各自独立：

```powershell
# 只跑 OCR：
python -m agent_qw.cli run -i 古籍图片/ -o out

# 只跑错别字纠正（从 OCR 输出直接纠正，跳过页序重排）：
python -m agent_c.cli correct -i out -o out_c
```
</details>

<details>
<summary><b>Q: 页序重排结果不对？</b></summary>

→ 核心原理是用每页首尾 80 字的连贯性来判断顺序。如果页面字数太少（不足 80 字），LLM 可能判断不准。试试加 `--batch-size 50`。
</details>

<details>
<summary><b>Q: 一本古籍拆成了多个 PDF，怎么处理？</b></summary>

→ 把多个 PDF 当成一本书处理，需要先把它们合并成一个 PDF（推荐用 [PDF24](https://tools.pdf24.org/zh/merge-pdf) 免费在线工具），再放进 `古籍图片/`。
</details>

<details>
<summary><b>Q: 要花多少钱？</b></summary>

→ 每页约 **¥0.016**（1 分 6 厘）：

| 步骤 | 每页成本 | 模型 |
|------|:------:|------|
| OCR 识别 | ¥0.014 | Qwen3.7-Plus（视觉） |
| 页序重排 | < ¥0.0001 | DeepSeek V4 Flash |
| 错别字纠正 | ¥0.0022 | DeepSeek V4 Flash |
| **合计** | **¥0.016** | |

一本 100 页的书全程约 ¥1.60，500 页约 ¥8。
</details>

---

## Agent 详解 / Agent Details

Each agent handles one stage of the pipeline: OCR recognition, page reordering, and typo correction.

### 1. agent_qw — OCR 识别

**功能**：将古籍 PDF 或图片转为结构化文本。

**输入**：PDF 文件（`.pdf`）或图片文件（`.png`, `.jpg`, `.jpeg`, `.tiff`, `.tif`, `.bmp`），可以是单个文件或整个目录。

**处理流程**：
1. PDF → 渲染为 300 DPI 图片（PyMuPDF）
2. 图片预处理：自适应二值化 → 可选锐化 → 可选纠偏
3. OCR 识别：将二值化图片以 base64 发给 Qwen-VL，直接看图识字
4. 跨页页眉页脚检测与去除（出现频率 ≥40% 的行视为页眉页脚）
5. 文本规范化（合并空白、去除控制字符）

**输出**：

```
out/<书名>/
├── txt/
│   └── page_001.txt           # 纯文本（已去页眉页脚）
├── json/
│   └── page_001.json          # 结构化结果（含 raw_text、bbox、置信度）
└── preprocess/
    └── page_001_binary.png    # 二值化预处理图片
```

JSON 字段说明：
- `raw_text`：OCR 原始输出（保留页眉页脚）
- `full_text`：去页眉页脚后的干净文本
- `blocks[]`：每块文字的 bbox、文本、行级置信度

**命令行参数**：

| 参数 | 说明 |
|------|------|
| `--input, -i` | 输入 PDF/图片文件或目录（必填） |
| `--output, -o` | 输出目录（默认: `out`） |
| `--config, -c` | 配置文件路径 |
| `--debug, -d` | Debug 模式，保存原始页面图片到 `pages/` |
| `--workers, -w` | 并行线程数（默认: 2） |

---

### 2. agent_verify — 页序重排

**功能**：用 LLM 判断页面文字连贯性，纠正 OCR 后的页码错乱。

**核心原理**：古籍每页文字连续——第 N 页结尾应与第 N+1 页开头连成一句话。将每页首尾各 80 字发给 DeepSeek，由 LLM 找出最连贯的页面排列顺序。

**输入**：`out/<书名>/txt/page_*.txt`（直接读清洗后的 txt，不依赖 JSON）

**批处理策略**：
- 每批最多 **100 页**，一次 LLM 调用
- 跨批校验：处理第 N+1 批时，传入第 N 批最后一页的尾部文字，确保页 100→101 的连贯性
- LLM 返回 JSON：`{"order": [3, 1, 2], "notes": "排序依据"}`

**输出**：

```
out_v/<书名>/
├── txt/page_001.txt           # 重排后的纯文本
└── json/page_001.json         # 对应 JSON（复制自输入）
```

**命令行参数**：

| 参数 | 说明 |
|------|------|
| `--input, -i` | 输入目录（如 `out/` 或 `out_c/`，必填） |
| `--output, -o` | 输出目录（默认: `out_v`） |
| `--debug, -d` | 启用 debug 日志 |
| `--model` | 覆盖 LLM 模型名 |
| `--base-url` | 覆盖 API 地址 |
| `--batch-size` | 每批最大页数（默认: 100） |

---

### 3. agent_c — 错别字纠正

**功能**：用 LLM 校勘 OCR 输出中的识别错误，专为 `llm_vision` 引擎设计。

**核心原理**：将整页文本发给 LLM，LLM 根据古籍用词习惯和上下文判断并修正 OCR 错误。重点关注形近字误识（如 己/已/巳、日/曰、未/末 等）。

**输入**：`out_v/<书名>/txt/page_*.txt`（或任意包含 `txt/` 子目录的书籍目录）

**输出**：

```
out_c/<书名>/
├── txt/page_001.txt           # 校勘后的文本
└── json/page_001.json         # 对应 JSON（复制自输入）
```

**命令行参数**：

| 参数 | 说明 |
|------|------|
| `--input, -i` | 输入目录（如 `out_v/`，必填） |
| `--output, -o` | 输出目录（默认: `out_c`） |
| `--debug, -d` | 启用 debug 日志 |
| `--model` | 覆盖 LLM 模型名 |
| `--base-url` | 覆盖 API 地址 |

---

## 配置 / Configuration

All settings live in `config.yaml` at the project root. Each agent can override model and API settings independently.

所有配置文件为项目根目录下的 `config.yaml`。

### 配置优先级（agent_verify 和 agent_c）

```
CLI 参数 --model / --base-url           （最高）
    ↓ 覆盖
环境变量 GUJI_LLM_MODEL / GUJI_LLM_BASE_URL / GUJI_LLM_API_KEY
    ↓ 覆盖
config.yaml 对应段（agent_verify 或 agent_c）
    ↓ 回退
config.yaml llm 段
    ↓ 回退
硬编码默认值                            （最低）
```

### 各 Agent 使用的模型

| Agent | 模型 | API 地址 | 用途 |
|-------|------|----------|------|
| agent_qw | Qwen3.7-Plus | DashScope (`dashscope.aliyuncs.com`) | 多模态视觉 OCR，直接看图识字 |
| agent_verify | DeepSeek-V4-Flash | DeepSeek (`api.deepseek.com`) | 页序连贯性判断 |
| agent_c | DeepSeek-V4-Flash | DeepSeek (`api.deepseek.com`) | 错别字校勘 |

### 完整配置示例

```yaml
# OCR 模型配置（agent_qw 使用）
ocr:
  model: qwen3.7-plus
  temperature: 0.1
  max_tokens: 4096

# LLM API 配置（agent_qw 使用）
llm:
  base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
  api_key: "sk-xxx"
  model: qwen3.7-plus
  temperature: 0.1
  max_tokens: 4096

# agent_verify 页序重排配置
agent_verify:
  base_url: https://api.deepseek.com/v1
  api_key: "sk-xxx"
  model: deepseek-v4-flash
  temperature: 0.1
  max_tokens: 4096

# agent_c 错别字纠正配置
agent_c:
  base_url: https://api.deepseek.com/v1
  api_key: "sk-xxx"
  model: deepseek-v4-flash
  temperature: 0.1
  max_tokens: 4096

# 图片预处理
layout:
  binarize: true
  sharpen: false
  sharpen_amount: 1.5
  deskew: false

# 全局设置
output_dir: out
debug: false
max_workers: 2
```

---

## 目录结构 / Directory Structure

```
项目根目录/
├── agent_qw/                  # OCR 识别 Agent（核心）
│   ├── ocr/                   #   OCR 引擎（多模态 LLM 视觉）
│   ├── vision/                #   图片预处理（二值化/锐化/纠偏）
│   ├── llm/                   #   LLM 客户端 + 提示词模板
│   ├── postprocess/           #   文本规范化
│   ├── cli.py                 #   命令行入口
│   └── pipeline.py            #   主流水线编排
├── agent_verify/              # 页序重排 Agent
│   ├── verifier.py            #   LLM 文字连贯性校验核心
│   ├── detector.py            #   旧版本地页码检测（保留，未使用）
│   └── cli.py                 #   命令行入口
├── agent_c/                   # 错别字纠正 Agent
│   ├── corrector.py           #   LLM 校勘核心
│   └── cli.py                 #   命令行入口
├── config.yaml                # 统一配置文件
├── pyproject.toml             # 包元数据和依赖
├── samples/                   # 测试样本
├── tests/                     # 单元测试
├── out/                       # agent_qw 输出
├── out_v/                     # agent_verify 输出
└── out_c/                     # agent_c 输出
```

---

## 测试 / Testing

```bash
cd tests
python test_schemas.py
python test_export.py
```
