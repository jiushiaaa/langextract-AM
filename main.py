# -*- coding: utf-8 -*-
"""
高熵合金论文抽取 Pipeline（基于 LangExtract）。

用法：
  python main.py                          # 默认 ernie4，处理全部 PDF
  python main.py --model deepseek          # DeepSeek V3
  python main.py --model qwen              # Qwen3 Coder 30B
  python main.py --model kimi              # Kimi K2
  python main.py --model ernie5            # ERNIE 5.0 Thinking
  python main.py --model gemini            # Gemini 2.0 Flash
  python main.py --max 2 --chunk 6000 --workers 5

流程：
  PDF → PyMuPDF 提文本 → 轻量清洗 → 手动分块 → 每块 lx.extract()（单块失败跳过/切半重试）
  → 扁平 Extraction → 聚合
  → group_extractions_to_entities()（按 material_id 聚合）
  → MaterialEntity（Pydantic 验证）
  → entity_to_target_json()（转目标模板）
  → output/he_data_{model}.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import langextract as lx
from langextract.resolver import ResolverParsingError

from config_manager import get_model_config
from pdf_utils import (
    extract_text_from_pdf,
    list_pdfs,
    chunk_text,
    clean_and_truncate_text,
)
from schemas import (
    build_prompt_description,
    entity_to_target_json,
    group_extractions_to_entities,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)

# ============================================================
# 常量
# ============================================================

ROOT = Path(__file__).resolve().parent
AMPDF_DIR = ROOT / "AMpdf"
OUTPUT_DIR = ROOT / "output"
# 某块解析失败时，若块长大于此值则自动切半重试一次
MIN_CHUNK_RETRY = 2000
# 单块最大等待时间（秒），超时则跳过该块；ERNIE5 Thinking 很慢，建议 240+
CHUNK_TIMEOUT = 240
# 分块级并发线程数（1=串行便于排查卡住，2+ 可能触发限流）
DEFAULT_CHUNK_WORKERS = 1
# 多线程写入同一 JSONL 时使用的锁
_write_lock = threading.Lock()

# ============================================================
# Few-shot 示例（LangExtract ExampleData）
#
# 注意：
#   - extraction_text 必须是 text 的逐字子串
#   - 按出现顺序排列，不重叠
#   - attributes 值全部为 str 或 list[str]（LangExtract 类型限制）
# ============================================================

EXAMPLES = [
    # ---- 示例 1：DED 制备 RHEA，多温度力学性能 ----
    lx.data.ExampleData(
        text=(
            "The Ti42Hf21Nb21V16 refractory high-entropy alloy was fabricated "
            "using directed energy deposition with a laser power of 550 W and "
            "a scanning speed of 5 mm/s. The alloy exhibited a yield strength "
            "of 1030 MPa and total elongation of 22.5% at room temperature. "
            "At 873 K, the yield strength was 636 MPa."
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="composition",
                extraction_text="Ti42Hf21Nb21V16",
                attributes={
                    "material_id": "T42",
                    "formula": "Ti42Hf21Nb21V16",
                    "elements_json": '{"Ti": 42, "Hf": 21, "Nb": 21, "V": 16}',
                    "unit": "at.%",
                },
            ),
            lx.data.Extraction(
                extraction_class="process",
                extraction_text=(
                    "directed energy deposition with a laser power of 550 W "
                    "and a scanning speed of 5 mm/s"
                ),
                attributes={
                    "material_id": "T42",
                    "method": "DED",
                    "heat_treatment": "",
                    "details": "laser power 550 W, scanning speed 5 mm/s",
                },
            ),
            lx.data.Extraction(
                extraction_class="property",
                extraction_text="yield strength of 1030 MPa",
                attributes={
                    "material_id": "T42",
                    "property_type": "Yield_Strength",
                    "value": "1030",
                    "unit": "MPa",
                    "test_temperature": "298 K",
                },
            ),
            lx.data.Extraction(
                extraction_class="property",
                extraction_text="total elongation of 22.5%",
                attributes={
                    "material_id": "T42",
                    "property_type": "Elongation_Total",
                    "value": "22.5",
                    "unit": "%",
                    "test_temperature": "298 K",
                },
            ),
            lx.data.Extraction(
                extraction_class="property",
                extraction_text="yield strength was 636 MPa",
                attributes={
                    "material_id": "T42",
                    "property_type": "Yield_Strength",
                    "value": "636",
                    "unit": "MPa",
                    "test_temperature": "873 K",
                },
            ),
        ],
    ),

    # ---- 示例 2：电弧熔炼 HEA，含 balance 元素 ----
    lx.data.ExampleData(
        text=(
            "FeCoCrNiMo0.3 high entropy alloy was prepared by arc melting in "
            "argon atmosphere, followed by homogenization at 1200C for 24h. "
            "Tensile tests showed an ultimate tensile strength of 853 MPa and "
            "elongation of 35.2% at 298 K."
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="composition",
                extraction_text="FeCoCrNiMo0.3",
                attributes={
                    "material_id": "FeCoCrNiMo0.3",
                    "formula": "FeCoCrNiMo0.3",
                    "elements_json": (
                        '{"Fe": 23.26, "Co": 23.26, "Cr": 23.26, '
                        '"Ni": 23.26, "Mo": 6.98}'
                    ),
                    "unit": "at.%",
                },
            ),
            lx.data.Extraction(
                extraction_class="process",
                extraction_text=(
                    "arc melting in argon atmosphere, followed by "
                    "homogenization at 1200C for 24h"
                ),
                attributes={
                    "material_id": "FeCoCrNiMo0.3",
                    "method": "Arc Melting",
                    "heat_treatment": "homogenization at 1200C for 24h",
                    "details": "argon atmosphere",
                },
            ),
            lx.data.Extraction(
                extraction_class="property",
                extraction_text="ultimate tensile strength of 853 MPa",
                attributes={
                    "material_id": "FeCoCrNiMo0.3",
                    "property_type": "UTS",
                    "value": "853",
                    "unit": "MPa",
                    "test_temperature": "298 K",
                },
            ),
            lx.data.Extraction(
                extraction_class="property",
                extraction_text="elongation of 35.2%",
                attributes={
                    "material_id": "FeCoCrNiMo0.3",
                    "property_type": "Elongation_Total",
                    "value": "35.2",
                    "unit": "%",
                    "test_temperature": "298 K",
                },
            ),
        ],
    ),
]


# ============================================================
# 文本预处理（仅移除出版商声明，保留论文主体）
# ============================================================

_REMOVE_PATTERNS = [
    re.compile(r".*all rights (are )?reserved.*", re.IGNORECASE),
    re.compile(r".*text and data mining.*", re.IGNORECASE),
    re.compile(r".*ai training.*and similar technologies.*", re.IGNORECASE),
    re.compile(r"^Contents lists available at ScienceDirect$", re.IGNORECASE),
    re.compile(r"^journal homepage:.*$", re.IGNORECASE),
    re.compile(r"^Available online.*$", re.IGNORECASE),
]


def clean_paper_text(text: str) -> str:
  """轻量清洗：仅移除出版商声明行，保留论文主体。"""
  lines = text.splitlines()
  keep = []
  for line in lines:
    stripped = line.strip()
    if not stripped:
      keep.append(line)
      continue
    if any(pat.match(stripped) for pat in _REMOVE_PATTERNS):
      continue
    keep.append(line)
  return "\n".join(keep)


# ============================================================
# 单篇 PDF 处理（按块抽取，单块失败不拖垮整篇）
# ============================================================

def process_chunk(
    chunk: str,
    pdf_name: str,
    profile,
    chunk_label: str,
    prompt: str,
    *,
    show_progress: bool = False,
) -> list:
  """
  对单块文本调用 lx.extract，返回该块的 Extraction 列表。
  并行调用时请设 show_progress=False 避免进度条错乱。
  """
  result = lx.extract(
      text_or_documents=chunk,
      prompt_description=prompt,
      examples=EXAMPLES,
      config=profile.config,
      use_schema_constraints=profile.use_schema_constraints,
      max_char_buffer=len(chunk) + 500,
      max_workers=1,
      batch_length=1,
      extraction_passes=1,
      show_progress=show_progress,
      prompt_validation_level=lx.prompt_validation.PromptValidationLevel.OFF,
  )
  return list(result.extractions or [])


def _process_one_chunk(
    idx: int,
    chunk: str,
    pdf_name: str,
    profile,
    prompt: str,
) -> tuple[int, list]:
  """
  单块处理（供线程池调用）：执行 lx.extract，失败时尝试切半重试，返回 (chunk_idx, extractions)。
  """
  label = str(idx)
  log.info("Chunk %s: 开始 lx.extract (约 %d 字符)", label, len(chunk))
  try:
    ex = process_chunk(
        chunk, pdf_name, profile, label, prompt, show_progress=False
    )
    if ex:
      log.info("Chunk %s: %d 条 Extraction", label, len(ex))
    return (idx, ex)
  except (ResolverParsingError, json.JSONDecodeError) as e:
    log.warning("Chunk %s JSON 解析失败: %s", label, str(e)[:100])
    if len(chunk) > MIN_CHUNK_RETRY:
      try:
        half = len(chunk) // 2
        ex_a = process_chunk(
            chunk[:half], pdf_name, profile, f"{label}a", prompt,
            show_progress=False,
        )
        ex_b = process_chunk(
            chunk[half:], pdf_name, profile, f"{label}b", prompt,
            show_progress=False,
        )
        combined = ex_a + ex_b
        log.info("Chunk %s 降级重试成功: %sa=%d, %sb=%d", label, label, len(ex_a), label, len(ex_b))
        return (idx, combined)
      except Exception as e2:
        log.warning("Chunk %s 降级重试仍失败: %s", label, e2)
    return (idx, [])
  except Exception as e:
    err_msg = str(e)
    if "Connection" in err_msg or "Connection error" in err_msg or "disconnected" in err_msg.lower():
      log.warning("Chunk %s 网络/连接错误，跳过: %s", label, err_msg[:80])
      return (idx, [])
    log.exception("Chunk %s 未预期错误: %s", label, e)
    return (idx, [])


def save_results_to_jsonl(
    records: list[dict],
    output_jsonl: Path,
    lock: threading.Lock,
) -> None:
  """线程安全地追加写入多条记录到 JSONL 文件。"""
  if not records:
    return
  with lock:
    try:
      with open(output_jsonl, "a", encoding="utf-8") as f:
        for rec in records:
          f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
      log.error("写入 JSONL 失败: %s", e)


def process_one_pdf(
    pdf_path: Path,
    profile,
    chunk_size: int,
    chunk_workers: int = DEFAULT_CHUNK_WORKERS,
) -> list[dict]:
  """
  对单篇 PDF：提文本 → 清洗并截断致谢/参考文献 → 手动分块 → 分块并发 lx.extract → 聚合 → 转 JSON。
  """
  # ---- 1. 提取纯文本并清洗、截断 ----
  log.info("[1/4] 提取正文: %s", pdf_path.name)
  raw_text = extract_text_from_pdf(pdf_path)
  text = clean_paper_text(raw_text)
  text = clean_and_truncate_text(text)
  log.info("        原始: %d 字符 → 清洗+截断后: %d 字符", len(raw_text), len(text))

  prompt = build_prompt_description()
  chunks = chunk_text(text, chunk_size, overlap=500)
  log.info(
      "[2/4] lx.extract 分块并发 (共 %d 块, chunk=%d, workers=%d)",
      len(chunks), chunk_size, chunk_workers,
  )

  all_extractions = []
  # 串行逐块跑 + 单块超时，避免某块在 200 OK 后解析卡死导致整程挂起
  for idx, ch in enumerate(chunks, 1):
    with ThreadPoolExecutor(max_workers=1) as executor:
      future = executor.submit(
          _process_one_chunk, idx, ch, pdf_path.name, profile, prompt
      )
      try:
        _, ex = future.result(timeout=CHUNK_TIMEOUT)
        all_extractions.append((idx, ex))
      except TimeoutError:
        log.warning(
            "Chunk %d 超过 %d 秒未返回（可能卡在 200 OK 后解析），跳过该块",
            idx, CHUNK_TIMEOUT,
        )
        all_extractions.append((idx, []))
      except Exception as e:
        log.exception("Chunk %s 异常: %s", idx, e)
        all_extractions.append((idx, []))

  # 按 chunk 顺序合并，保证结果可复现
  all_extractions.sort(key=lambda x: x[0])
  merged = []
  for _, ex in all_extractions:
    merged.extend(ex)
  all_extractions = merged

  log.info("        合计 %d 条 Extraction", len(all_extractions))

  if not all_extractions:
    return []

  # ---- 3. 聚合为 MaterialEntity (Pydantic) ----
  log.info("[3/4] 聚合 MaterialEntity")
  entities, evidence = group_extractions_to_entities(all_extractions)
  log.info("        识别出 %d 种材料", len(entities))

  # ---- 4. 转目标 JSON 模板 ----
  log.info("[4/4] 转目标 JSON")
  records = []
  for entity in entities:
    rec = entity_to_target_json(
        entity=entity,
        source_pdf=pdf_path.name,
        evidence=evidence,
    )
    records.append(rec)
  return records


# ============================================================
# 主函数
# ============================================================

def main():
  parser = argparse.ArgumentParser(
      description="高熵合金论文结构化抽取 Pipeline (LangExtract)",
  )
  parser.add_argument(
      "--model", default="ernie4",
      choices=["ernie5", "ernie4", "deepseek", "qwen", "kimi", "gemini"],
      help="选择模型 (default: ernie4)",
  )
  parser.add_argument(
      "--max", type=int, default=0, dest="max_pdfs",
      help="最多处理几篇 PDF (0=全部, default: 0)",
  )
  parser.add_argument(
      "--chunk", type=int, default=6000,
      help="分块大小/字符，单块失败会跳过或切半重试 (default: 6000)",
  )
  parser.add_argument(
      "--workers", type=int, default=DEFAULT_CHUNK_WORKERS,
      help="分块并发线程数 (default: %d)" % DEFAULT_CHUNK_WORKERS,
  )
  args = parser.parse_args()

  # ---- 获取模型配置 ----
  profile = get_model_config(args.model)
  log.info("模型: %s (%s)", profile.config.model_id, profile.label)
  log.info("schema_constraints: %s, chunk_size: %s, workers: %s",
           profile.use_schema_constraints, args.chunk, args.workers)

  # ---- 列出 PDF ----
  pdfs = list_pdfs(AMPDF_DIR)
  if not pdfs:
    log.error("未在 %s 下找到 PDF，退出", AMPDF_DIR)
    return 1

  if args.max_pdfs > 0:
    pdfs = pdfs[: args.max_pdfs]
    log.info("仅处理前 %d 篇 PDF", len(pdfs))

  # ---- 输出路径（含模型名），本次运行先清空 ----
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
  output_jsonl = OUTPUT_DIR / f"he_data_{profile.label}.jsonl"
  try:
    with open(output_jsonl, "w", encoding="utf-8") as _:
      pass
  except Exception as e:
    log.error("无法创建输出文件 %s: %s", output_jsonl, e)
    return 1

  total_records = 0
  log.info("共 %d 篇 PDF", len(pdfs))
  log.info("=" * 60)

  for i, pdf_path in enumerate(pdfs, 1):
    log.info("[%d/%d] %s", i, len(pdfs), pdf_path.name)
    try:
      records = process_one_pdf(
          pdf_path, profile, args.chunk, chunk_workers=args.workers
      )
      save_results_to_jsonl(records, output_jsonl, _write_lock)
      total_records += len(records)
      log.info("        本篇: %d 条材料记录", len(records))
    except Exception as e:
      log.exception("本篇失败: %s", e)
      continue

  log.info("=" * 60)
  log.info("写入: %s", output_jsonl)
  log.info("总材料记录数: %d", total_records)
  return 0


if __name__ == "__main__":
  sys.exit(main())
