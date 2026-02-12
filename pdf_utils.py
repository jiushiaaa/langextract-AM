# -*- coding: utf-8 -*-
"""从 PDF 提取纯文本、清洗截断、分块，供后续 LLM 分析。"""

import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# 在正文末尾截断：在文本后 30% 区域内匹配以下独立行标题（不区分大小写），取最早位置截断
_TRUNCATE_PATTERN = re.compile(
    r"(?im)^\s*("
    r"Acknowledgements|Acknowledgments|"
    r"Declaration of Competing Interest|"
    r"Conflict of interest|"
    r"CRediT authorship contribution statement|"
    r"References|REFERENCES"
    r")\s*$"
)


def extract_text_from_pdf(pdf_path: str | Path) -> str:
  """从单个 PDF 提取全文，优先 PyMuPDF，失败时尝试 pdfplumber。"""
  pdf_path = Path(pdf_path)
  if not pdf_path.is_file():
    raise FileNotFoundError(f"PDF 不存在: {pdf_path}")

  try:
    import pymupdf
    doc = pymupdf.open(pdf_path)
    parts = []
    for page in doc:
      parts.append(page.get_text())
    doc.close()
    return "\n".join(parts)
  except Exception as e1:
    try:
      import pdfplumber
      with pdfplumber.open(pdf_path) as doc:
        parts = [p.extract_text() or "" for p in doc.pages]
      return "\n".join(parts)
    except Exception as e2:
      raise RuntimeError(
          f"PyMuPDF 与 pdfplumber 均无法解析 {pdf_path}: pymupdf={e1}, pdfplumber={e2}"
      ) from e2


def list_pdfs(dir_path: str | Path) -> list[Path]:
  """列出目录下所有 .pdf 文件（按文件名排序）。"""
  dir_path = Path(dir_path)
  if not dir_path.is_dir():
    return []
  return sorted(dir_path.glob("*.pdf"))


def clean_and_truncate_text(text: str) -> str:
  """
  在分块前截断：丢弃论文末尾的致谢、利益冲突、参考文献等非正文内容。
  仅在文本**后 30%** 区域内查找独立行标题，取最早匹配位置截断，避免误删正文中的词汇。

  Returns:
    截断后的文本（可能为原文本若未匹配到任何标题）。
  """
  if not text or len(text) < 100:
    return text
  try:
    search_start = max(0, int(len(text) * 0.7))
    search_zone = text[search_start:]
    m = _TRUNCATE_PATTERN.search(search_zone)
    if m:
      truncate_pos = search_start + m.start()
      out = text[:truncate_pos].rstrip()
      logger.info(
          "clean_and_truncate: 在位置 %d 截断 (匹配: %r), 长度 %d -> %d",
          truncate_pos, m.group(1).strip(), len(text), len(out),
      )
      return out
  except Exception as e:
    logger.warning("clean_and_truncate 异常，保留原文: %s", e)
  return text


def chunk_text(
  text: str,
  chunk_size: int,
  overlap: int = 500,
) -> list[str]:
  """
  将长文本按固定大小分块，块间保留 overlap 字符重叠以保持上下文连贯。

  Args:
    text: 全文
    chunk_size: 每块目标长度（字符）
    overlap: 块与块之间的重叠长度（字符）

  Returns:
    文本块列表
  """
  if not text or chunk_size <= 0:
    return []
  if len(text) <= chunk_size:
    return [text]

  chunks = []
  start = 0
  while start < len(text):
    end = min(start + chunk_size, len(text))
    chunks.append(text[start:end])
    if end >= len(text):
      break
    start = end - overlap
    if start >= end:
      start = end
  return chunks
