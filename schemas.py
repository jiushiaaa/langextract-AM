# -*- coding: utf-8 -*-
"""
高熵合金论文抽取 —— Pydantic 数据模型 + Prompt 定义 + 转换工具。

核心设计：
  - Pydantic 模型的 Field(description=...) 同时充当「给人看的文档」和「给 LLM 看的指令」；
  - analysis_thought 字段专为 ERNIE 5.0 Thinking 设计：
    在 JSON 内部留一个"思维链缓存"位，防止 CoT 吐在 JSON 结构外导致解析失败；
  - LangExtract 使用扁平 Extraction → 本模块负责聚合成嵌套 MaterialEntity → 再转目标模板。
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from pydantic import BaseModel, Field


# ============================================================
# 1. 基础组件
# ============================================================

class Element(BaseModel):
  """单个合金元素及其含量。"""
  symbol: str = Field(
      ...,
      description="元素符号 (e.g., 'Ni', 'Al'). 必须是标准化学符号。",
  )
  value: float = Field(
      ...,
      description="数值。如果是范围值(20-30)，请取平均值。",
  )
  unit: str = Field(
      "at.%",
      description="单位 (e.g., 'at.%', 'wt.%').",
  )
  is_balance: bool = Field(
      False,
      description="如果文中提到该元素是 'balance' 或 'rem.', 设为 True。",
  )


class Property(BaseModel):
  """单条力学性能测量。"""
  property_type: str = Field(
      ...,
      description=(
          "性能类别 (e.g., 'Yield_Strength', 'UTS', "
          "'Elongation_Total', 'Elongation_Uniform', 'Hardness')."
      ),
  )
  value: float = Field(..., description="具体数值。")
  unit: str = Field(..., description="单位 (e.g., 'MPa', '%', 'HV').")
  test_temperature: Optional[str] = Field(
      None,
      description=(
          "测试温度 (e.g., 'Room Temperature', '298K', '600C'). "
          "如未提及则留空。"
      ),
  )


class Processing(BaseModel):
  """制备工艺信息。"""
  method: str = Field(
      ...,
      description="主要制备方法 (e.g., 'Arc Melting', 'DED', 'SLM').",
  )
  heat_treatment: Optional[str] = Field(
      None,
      description="热处理条件描述 (e.g., 'Annealed at 1100C for 2h').",
  )


# ============================================================
# 2. 顶层聚合对象
# ============================================================

class MaterialEntity(BaseModel):
  """一种材料的完整记录。"""
  # 思维链缓存 —— 引导 ERNIE 5.0 先分析再输出
  analysis_thought: Optional[str] = Field(
      None,
      description=(
          "请先简要分析这段文本中包含了几种材料，以及它们的关键特征，"
          "然后再填充后续字段。"
      ),
  )
  material_name: str = Field(
      ...,
      description="文中使用的材料标识符 (e.g., 'RHEA-1', 'Sample A', 'T42').",
  )
  formula: str = Field(
      ...,
      description="完整的化学式 (e.g., 'Ti42Hf21Nb21V16').",
  )
  composition: list[Element] = Field(
      default_factory=list,
      description="该材料的详细成分列表。",
  )
  process: Processing = Field(
      default_factory=lambda: Processing(method="Unknown"),
      description="该材料的制备与处理工艺。",
  )
  properties: list[Property] = Field(
      default_factory=list,
      description="该材料对应的所有力学性能数据。",
  )


# ============================================================
# 3. 证据（溯源）记录
# ============================================================

class EvidenceSpan(BaseModel):
  """LangExtract 提供的原文溯源区间。"""
  extraction_class: str
  text: str
  char_start: Optional[int] = None
  char_end: Optional[int] = None
  alignment: Optional[str] = None


# ============================================================
# 4. LangExtract Prompt 描述（嵌入 Pydantic description）
# ============================================================

def build_prompt_description() -> str:
  """
  自动从 Pydantic 模型的 Field description 生成 LangExtract 用的 prompt_description，
  确保 Prompt 与 Schema 始终一致。
  """
  return """\
Extract ALL materials, their compositions, processing methods, and mechanical \
properties from this materials science text.

Extraction classes and required attributes:

1. "composition" — Chemical composition of each alloy.
   Attributes:
     material_id       — short identifier for this alloy (e.g. T42, Mo3)
     formula           — """ + MaterialEntity.model_fields["formula"].description + """
     elements_json     — a JSON string mapping element symbols to numeric values,
                         e.g. '{"Ti": 42, "Hf": 21, "Nb": 21, "V": 16}'.
                         """ + Element.model_fields["is_balance"].description + """
                         If balance, set value to -1.
     unit              — """ + Element.model_fields["unit"].description + """

2. "process" — Fabrication / processing method.
   Attributes:
     material_id       — same id as the related composition
     method            — """ + Processing.model_fields["method"].description + """
     heat_treatment    — """ + (Processing.model_fields["heat_treatment"].description or "") + """
     details           — other key parameters: power, speed, layer thickness, etc.

3. "property" — Each individual mechanical property measurement.
   Attributes:
     material_id       — same id as the related composition
     property_type     — """ + Property.model_fields["property_type"].description + """
     value             — """ + Property.model_fields["value"].description + """ (as a string)
     unit              — """ + Property.model_fields["unit"].description + """
     test_temperature  — """ + (Property.model_fields["test_temperature"].description or "") + """

Rules:
- Use EXACT text spans from the source. Do NOT paraphrase.
- List extractions in order of appearance. Do NOT overlap spans.
- Use the SAME material_id across composition / process / property.
- Only extract data EXPLICITLY stated. Do NOT guess or calculate.
- If multiple materials are studied, extract ALL of them.
- For range values (e.g. 20-30), take the midpoint.
- For 'balance' elements, set value to -1.
"""


# ============================================================
# 5. 扁平 Extraction → MaterialEntity 聚合
# ============================================================

def _parse_elements_json(raw: str) -> list[Element]:
  """把 '{"Ti": 42, "Hf": 21}' 解析为 Element 列表。"""
  try:
    d = json.loads(raw)
    return [
        Element(
            symbol=str(k),
            value=float(v) if float(v) != -1 else 0,
            unit="at.%",
            is_balance=(float(v) == -1),
        )
        for k, v in d.items()
    ]
  except (json.JSONDecodeError, TypeError, ValueError):
    return []


def group_extractions_to_entities(
    extractions: list,
) -> tuple[list[MaterialEntity], list[EvidenceSpan]]:
  """
  将 LangExtract 返回的扁平 Extraction 按 material_id 聚合为 MaterialEntity。

  分组策略：
    - attributes 中有 material_id 的，按其值分组；
    - 缺少 material_id 的，归入最近一次 composition 的组。
  """
  all_evidence: list[EvidenceSpan] = []
  for ext in extractions:
    ci = ext.char_interval
    all_evidence.append(EvidenceSpan(
        extraction_class=ext.extraction_class,
        text=ext.extraction_text,
        char_start=ci.start_pos if ci else None,
        char_end=ci.end_pos if ci else None,
        alignment=ext.alignment_status.value if ext.alignment_status else None,
    ))

  groups: dict[str, dict[str, list]] = {}
  current_mid = "__default__"

  for ext in extractions:
    attrs = ext.attributes or {}
    mid = attrs.get("material_id", "")
    cls = ext.extraction_class

    if cls == "composition":
      mid = mid or attrs.get("formula", ext.extraction_text[:40])
      current_mid = mid
    if not mid:
      mid = current_mid

    if mid not in groups:
      groups[mid] = {"compositions": [], "processes": [], "properties": []}

    if cls == "composition":
      groups[mid]["compositions"].append(attrs)
    elif cls == "process":
      groups[mid]["processes"].append(attrs)
    elif cls == "property":
      groups[mid]["properties"].append(attrs)

  entities: list[MaterialEntity] = []
  for mid, g in groups.items():
    # --- composition ---
    formula = ""
    elements: list[Element] = []
    unit = "at.%"
    for c in g["compositions"]:
      formula = c.get("formula", formula)
      unit = c.get("unit", unit)
      elems = _parse_elements_json(c.get("elements_json", "{}"))
      if elems:
        elements = elems
        for e in elements:
          e.unit = unit

    # --- process ---
    method = ""
    heat_treatment = None
    details_parts = []
    for p in g["processes"]:
      method = p.get("method", method)
      ht = p.get("heat_treatment")
      if ht:
        heat_treatment = ht
      det = p.get("details")
      if det:
        details_parts.append(det)

    proc = Processing(
        method=method or "Unknown",
        heat_treatment=heat_treatment,
    )

    # --- properties ---
    props: list[Property] = []
    for pr in g["properties"]:
      try:
        val = float(pr.get("value", ""))
      except (ValueError, TypeError):
        continue
      props.append(Property(
          property_type=pr.get("property_type", pr.get("name", "Unknown")),
          value=val,
          unit=pr.get("unit", ""),
          test_temperature=pr.get("test_temperature", pr.get("condition")),
      ))

    entity = MaterialEntity(
        material_name=mid,
        formula=formula or mid,
        composition=elements,
        process=proc,
        properties=props,
    )
    entities.append(entity)

  return entities, all_evidence


# ============================================================
# 6. MaterialEntity → 用户目标 JSON 模板
# ============================================================

def _temperature_to_kelvin(temp_str: str | None) -> int:
  """'298 K' / '600C' / 'Room Temperature' → 数值 (K)。"""
  if not temp_str:
    return 298
  t = temp_str.lower().strip()
  if "room" in t or t == "rt":
    return 298
  nums = re.findall(r"[\d.]+", temp_str)
  if not nums:
    return 298
  val = float(nums[0])
  if "c" in t and "k" not in t:
    return int(val + 273)
  return int(val)


def entity_to_target_json(
    entity: MaterialEntity,
    source_pdf: str,
    evidence: list[EvidenceSpan] | None = None,
) -> dict[str, Any]:
  """MaterialEntity → Composition_Info / Process_Info / Properties_Info 目标模板。"""
  mid = entity.material_name
  sid = f"S_{mid}_AsBuilt"
  elements_dict = {}
  for e in entity.composition:
    elements_dict[e.symbol] = -1 if e.is_balance else e.value

  composition_info = {
      "Mat_ID": f"M_{mid}",
      "Alloy_Name_Raw": mid,
      "Formula_Normalized": entity.formula,
      "Composition_JSON": json.dumps(elements_dict, ensure_ascii=False),
      "Source_DOI": "",
  }

  process_info = {
      "Sample_ID": sid,
      "Mat_ID": f"M_{mid}",
      "Process_Category": entity.process.method,
      "Process_Text_For_AI": entity.process.heat_treatment or "",
      "Key_Params_JSON": "{}",
      "Main_Phase": "",
      "Microstructure_Text_For_AI": "",
      "Has_Precipitates": False,
      "Grain_Size_avg_um": None,
  }

  properties_info = []
  for i, prop in enumerate(entity.properties, start=1):
    properties_info.append({
        "Test_ID": f"T_{mid}_{i:02d}",
        "Sample_ID": sid,
        "Test_Temperature_K": _temperature_to_kelvin(prop.test_temperature),
        "Property_Type": prop.property_type,
        "Property_Value": prop.value,
        "Property_Unit": prop.unit,
    })

  result: dict[str, Any] = {
      "_source_pdf": source_pdf,
      "Composition_Info": composition_info,
      "Process_Info": process_info,
      "Properties_Info": properties_info,
  }
  if evidence:
    result["_evidence"] = [e.model_dump() for e in evidence]

  return result
