# 高熵合金论文结构化抽取 Pipeline

从高熵合金（HEA）领域学术 PDF 中抽取「成分–工艺–性能」结构化数据，输出 JSONL，供下游机器学习或知识库使用。

基于 [LangExtract](https://github.com/google/langextract) ,适配星河社区的所有模型及Gemini和openai等模型，支持文本清洗、分块抽取、单块超时与解析失败重试。

---

## 功能概览

| 功能 | 说明 |
|------|------|
| **多模型** | ERNIE 4.5/5.0、DeepSeek V3、Qwen3 Coder、Kimi K2（星河 API）、Gemini 2.0 Flash |
| **文本清洗** | 去除出版商声明；在正文后 30% 内截断致谢 / 利益冲突 / 参考文献，节省 Token |
| **分块抽取** | 按字符数分块 + 重叠，单块失败可切半重试；单块超时（默认 120s）跳过，不拖死整程 |
| **结构化输出** | 扁平 Extraction → 按 `material_id` 聚合为 MaterialEntity → 转目标 JSON 模板写入 JSONL |

---

## 项目结构

```
AM/
├── main.py              # 入口：argparse、分块、lx.extract、聚合、写 JSONL
├── config_manager.py    # 模型工厂：星河 / Gemini 的 ModelConfig + max_tokens / timeout
├── pdf_utils.py         # PDF 提文本、clean_and_truncate_text、chunk_text
├── schemas.py           # Pydantic 模型（Element / Property / Processing / MaterialEntity）
│                        # + build_prompt_description、group_extractions_to_entities、entity_to_target_json
├── .env                 # 本地 API Key
├── AMpdf/               # 待处理 PDF，程序扫描该目录下 *.pdf
├── output/              # 输出 he_data_{model}.jsonl
├── requirements.txt     # 依赖
└── README.md            # 本文件
```

可选：仓库内可含 `langextract-main/` 作为 LangExtract 子模块或本地参考，运行时不依赖该目录。

---

## 环境要求

- **Python** ≥ 3.10
- 建议使用虚拟环境：`python -m venv .venv` 后激活再安装依赖

---

## 安装

```bash
git clone https://github.com/jiushiaaa/langextract-AM.git
cd AM
pip install -r requirements.txt
```

---

## 配置

在项目根目录创建 `.env` 文件：

```env
# 星河社区（用于 ernie4.5/ ernie5 / deepseek / qwen / kimi）
AI_STUDIO_API_KEY=你的星河API密钥

# 仅在使用 --model gemini 时需要
GOOGLE_API_KEY=你的Gemini密钥
```

星河 API 密钥在 [飞桨 AI Studio](https://aistudio.baidu.com/) 获取；Gemini 在 Google AI Studio 获取。

---

## 使用方法

```bash
# 默认：ERNIE 4.5，处理 AMpdf/ 下全部 PDF，chunk=6000，串行
python main.py

# 指定模型
python main.py --model qwen
python main.py --model deepseek
python main.py --model kimi
python main.py --model ernie5
python main.py --model gemini

# 限制篇数、分块大小、并发（当前默认串行，--workers 主要影响后续可扩展）
python main.py --model ernie4 --max 2 --chunk 12000
```

### 参数说明

| 参数 | 默认 | 说明 |
|------|------|------|
| `--model` | ernie4.5 | 模型：ernie5, ernie4.5, deepseek, qwen, kimi, gemini |
| `--max` | 0 | 最多处理 PDF 数量，0 表示全部 |
| `--chunk` | 6000 | 分块大小（字符），单块失败会切半重试 |
| `--workers` | 1 | 分块并发数，1 为串行（便于排查卡住） |

---

## 输出

- 路径：`output/he_data_{model}.jsonl`
- 格式：每行一条 JSON，包含 `source_pdf`、`chunk_id` 相关信息及抽取出的成分/工艺/性能等字段（见 `schemas.py` 中目标模板）。
- 每次运行会**覆盖**该模型对应的 JSONL 文件；多篇 PDF 时按篇追加写入（线程安全）。

---

## 常见问题与优化方向

1. **403 访问过于频繁**  
   星河社区 API 限流：保持 `--workers 1` 或改为 2，并适当增大 `--chunk` 减少请求次数。

2. **某块一直卡在 “HTTP 200 OK” 之后**  
   已加单块超时（默认 240 秒），超时会自动跳过该块并继续下一块；可在 `main.py` 中调整 `CHUNK_TIMEOUT`。ERNIE5 Thinking 推理慢，建议保持 240 或更大。

3. **JSON 解析失败（Expecting value: line 1 column 1 / Unterminated string）**  
   单块会先切半重试；若仍失败则跳过该块。**ERNIE 5.0 Thinking** 常在 JSON 前输出推理内容，易触发 “Expecting value”，属已知现象，可改用 ernie4.5 或增大 chunk 减少请求次数。

4. **Connection error / Server disconnected**  
   网络或服务端断开时，该块会跳过并打简短日志，不打断整程；可稍后重跑或换网络。

5. **可优化方向（供同伴扩展）**  
   - 增加更多 Few-shot 示例或细化 `schemas.py` 中 Field 的 description，提升抽取质量。  
   - 对星河社区内的模型做简单请求频率限制（如 token bucket），避免 403。  
   - 支持从本地 `langextract-main` 安装开发版：`pip install -e ./langextract-main`。  
   - 输出层增加去重、与已有 JSONL 的 merge 策略。  
   - 将 `CHUNK_TIMEOUT`、`MIN_CHUNK_RETRY` 等做成命令行或配置文件项。

---

## 依赖（见 requirements.txt）

- `langextract`：结构化抽取
- `pydantic`：数据模型与校验
- `pymupdf`：PDF 文本提取（首选）
- `pdfplumber`：PDF 提取备选
- `python-dotenv`：加载 `.env`

---

