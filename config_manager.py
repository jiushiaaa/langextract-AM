# -*- coding: utf-8 -*-
"""
模型工厂 —— 根据名称返回 LangExtract ModelConfig + 运行时标志。

支持模型（星河社区 OpenAI 兼容 API，除 gemini 外）：
  deepseek — DeepSeek V3          (Max Output 12k，max_tokens=8192)
  ernie4   — ERNIE 4.5 Turbo 128k (Max Output 12k，强制 max_tokens=8192)
  qwen     — Qwen3 Coder 30B       (Max Output 32k，max_tokens=16384)
  kimi     — Kimi K2 Instruct      (Max Output 32k，max_tokens=16384)
  ernie5   — ERNIE 5.0 Thinking   (max_tokens=8192)
  gemini   — Google Gemini 2.0 Flash (独立 API，含 schema 约束)

关键设计：
  - 星河模型统一 provider="OpenAILanguageModel"，base_url 星河 API；
  - provider_kwargs 中 max_output_tokens 映射为 API 的 max_tokens，防止 JSON 截断；
  - Gemini 使用 LangExtract 原生 GeminiLanguageModel。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from langextract import factory


@dataclass
class ModelProfile:
  """模型配置 + 运行时标志。"""
  config: factory.ModelConfig
  use_schema_constraints: bool   # Gemini 支持，ERNIE 不支持
  label: str                      # 用于输出文件名，如 "ernie5"


def _load_env() -> None:
  """尝试从项目根 .env 加载环境变量。"""
  try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.is_file():
      load_dotenv(env_path)
  except Exception:
    pass


def get_model_config(model_name: str) -> ModelProfile:
  """
  根据名称返回模型配置。

  Args:
    model_name: 'ernie5' | 'ernie4' | 'deepseek' | 'qwen' | 'kimi' | 'gemini'

  Returns:
    ModelProfile 实例。
  """
  _load_env()

  api_key = os.environ.get("AI_STUDIO_API_KEY")
  base_url = "https://aistudio.baidu.com/llm/lmapi/v3"

  # ---------- 星河社区 OpenAI 兼容模型（统一 base_url + api_key） ----------
  # 请求超时（秒），超时后报错便于重试，避免无限等待
  REQUEST_TIMEOUT = 60.0

  def _openai_profile(model_id: str, max_output_tokens: int, label: str) -> ModelProfile:
    if not api_key:
      raise RuntimeError("未设置 AI_STUDIO_API_KEY")
    return ModelProfile(
        config=factory.ModelConfig(
            model_id=model_id,
            provider="OpenAILanguageModel",
            provider_kwargs={
                "api_key": api_key,
                "base_url": base_url,
                "temperature": 0.1,
                "max_output_tokens": max_output_tokens,
                "timeout": REQUEST_TIMEOUT,
            },
        ),
        use_schema_constraints=False,
        label=label,
    )

  if model_name == "deepseek":
    # Max Output 12k，设 8192 留安全余量
    return _openai_profile("deepseek-v3", 8192, "deepseek")

  if model_name == "ernie4":
    # Max Output 12k，必须强制 8192（覆盖默认 2k）
    return _openai_profile("ernie-4.5-turbo-128k-preview", 8192, "ernie4")

  if model_name == "qwen":
    # Max Output 32k，设 16384 利用长输出、防长表格截断
    return _openai_profile("qwen3-coder-30b-a3b-instruct", 16384, "qwen")

  if model_name == "kimi":
    # Max Output 32k，设 16384
    return _openai_profile("kimi-k2-instruct", 16384, "kimi")

  if model_name == "ernie5":
    return _openai_profile("ernie-5.0-thinking-preview", 8192, "ernie5")

  if model_name == "gemini":
    api_key_gemini = os.environ.get(
        "GOOGLE_API_KEY",
        os.environ.get("LANGEXTRACT_API_KEY", ""),
    )
    if not api_key_gemini:
      raise RuntimeError("未设置 GOOGLE_API_KEY 或 LANGEXTRACT_API_KEY")
    return ModelProfile(
        config=factory.ModelConfig(
            model_id="gemini-2.0-flash",
            # 不指定 provider —— LangExtract 自动匹配 ^gemini → GeminiLanguageModel
            provider_kwargs={
                "api_key": api_key_gemini,
            },
        ),
        # Gemini 原生支持 schema 约束（controlled generation）
        use_schema_constraints=True,
        label="gemini",
    )

  raise ValueError(
      f"未知模型: {model_name!r}。支持: ernie5, ernie4, deepseek, qwen, kimi, gemini"
  )
