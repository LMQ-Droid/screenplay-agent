# -*- coding: utf-8 -*-
"""Agent 主流程：拆任务、串行编排、逐场质控。

任务拆分方式（这是"Agent"的实质，不是玄学）：

    全文
     ├─ ① 分场       一次调用，只返回标题列表（输出短、便宜）
     ├─ ② 逐场抽取   每场一次调用，只喂这一场原文（上下文聚焦）
     ├─ ③ 人物合并   一次调用，把各场人物去重（全局一致性）
     └─ ④ 抽样复核   随机抽几场做漏抽检查（质控闭环）

为什么不用「一次调用处理整篇」：
- 整篇塞进去会超上下文，长剧本直接崩；
- 一次输出几十场，模型容易中途丢失结构、尾部截断；
- 逐场调用可并行、单场失败只重试那一场，成本可控。

代价是丢失跨场上下文，所以补一个③做全局人物归一。
"""
from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass, field
from typing import Any

from . import prompts
from .llm import LLMClient
from .schema import (
    REQUIRED_LINE_FIELDS,
    REQUIRED_SCENE_FIELDS,
    Emotion,
    Scene,
    Screenplay,
    parse_model,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- 统计


@dataclass
class AgentStats:
    """一次运行的完整统计，用于质控报告。"""

    scenes_total: int = 0
    scenes_ok: int = 0
    scenes_failed: int = 0
    scenes_retried: int = 0
    retry_reasons: dict[str, int] = field(default_factory=dict)
    drop_reasons: dict[str, int] = field(default_factory=dict)
    character_conflicts: list[str] = field(default_factory=list)
    qa_checked: int = 0
    qa_missing_dialogue: int = 0
    qa_missing_characters: int = 0

    def note_retry(self, reason: str) -> None:
        self.scenes_retried += 1
        self.retry_reasons[reason] = self.retry_reasons.get(reason, 0) + 1

    def note_drop(self, reason: str) -> None:
        self.drop_reasons[reason] = self.drop_reasons.get(reason, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenes_total": self.scenes_total,
            "scenes_ok": self.scenes_ok,
            "scenes_failed": self.scenes_failed,
            "scenes_retried": self.scenes_retried,
            "retry_reasons": self.retry_reasons,
            "drop_reasons": self.drop_reasons,
            "character_conflicts": self.character_conflicts,
            "qa_checked": self.qa_checked,
            "qa_missing_dialogue": self.qa_missing_dialogue,
            "qa_missing_characters": self.qa_missing_characters,
        }


# ---------------------------------------------------------------- 异常


class QualityGateError(Exception):
    """质控未通过。调用方决定是丢弃该场还是整批中止。"""


# ---------------------------------------------------------------- Agent


class ScreenplayAgent:
    """剧本结构化 Agent。

    Args:
        client: LLM 客户端
        max_retries: 单场抽取最多重试次数
        qa_sample: 抽样复核的场次数，0 表示不做
        strict: True 时人物冲突直接报错，False 时只记录
    """

    def __init__(
        self,
        client: LLMClient | None = None,
        max_retries: int = 2,
        qa_sample: int = 3,
        strict: bool = False,
    ) -> None:
        self.client = client or LLMClient()
        self.max_retries = max_retries
        self.qa_sample = qa_sample
        self.strict = strict
        self.stats = AgentStats()

    # ------------------------------------------------------------ 主流程

    def parse(self, text: str, title: str = "未命名剧本") -> Screenplay:
        """把整篇剧本解析成结构化结果。"""
        if not text or not text.strip():
            raise ValueError("剧本内容为空")

        self.stats = AgentStats()

        # ① 分场
        spans = self.split_scenes(text)
        self.stats.scenes_total = len(spans)
        logger.info("识别到 %d 场", len(spans))

        # ② 逐场抽取
        scenes: list[Scene] = []
        for i, span in enumerate(spans, start=1):
            scene_id = f"S{i:02d}"
            scene = self.extract_scene(span, scene_id)
            if scene is None:
                self.stats.scenes_failed += 1
                continue
            scenes.append(scene)
            self.stats.scenes_ok += 1

        if not scenes:
            raise QualityGateError("所有场次都抽取失败，请检查模型输出或提示词")

        # ③ 人物合并
        characters = self.merge_characters(scenes)

        # ④ 抽样复核
        if self.qa_sample > 0 and spans:
            self.run_qa(spans, scenes)

        return Screenplay(
            title=title,
            characters=characters,
            scenes=scenes,
            meta={
                "stats": self.stats.as_dict(),
                "usage": self.client.usage.as_dict(),
                "offline": self.client.offline,
            },
        )

    # ------------------------------------------------------------ ① 分场

    def split_scenes(self, text: str) -> list[str]:
        """切分场次，返回每场的原文片段。

        失败兜底很重要：模型分场失败时不能整篇丢弃，退化成
        「按空行切块」或「整篇一场」，至少保证流程能继续。
        """
        # 先做规则预切分：如果原文本身有清晰场标，直接用规则结果。
        # 这样能省一次模型调用，且比模型更可靠。
        rule_spans = _rule_split_by_heading(text)
        if len(rule_spans) >= 2:
            logger.info("规则识别到 %d 场，跳过分场模型调用", len(rule_spans))
            return rule_spans

        try:
            data = self.client.complete_json(
                prompts.SCENE_SPLIT_SYSTEM,
                prompts.SCENE_SPLIT_USER.format(payload=text[:20000]),
                temperature=0.0,
            )
            headings = [s.get("heading", "") for s in data.get("scenes", [])]
            spans = _locate_spans(text, headings)
            if spans:
                return spans
            logger.warning("分场结果无法在原文中定位，退回规则分场")
        except Exception as exc:  # noqa: BLE001
            logger.warning("分场模型调用失败: %s，退回规则分场", exc)

        return rule_spans or [text]

    # ------------------------------------------------------------ ② 逐场

    def extract_scene(self, span: str, scene_id: str) -> Scene | None:
        """抽取单场结构，带校验与重试。"""
        last_error = ""
        for attempt in range(self.max_retries + 1):
            try:
                data = self.client.complete_json(
                    prompts.SCENE_EXTRACT_SYSTEM,
                    prompts.SCENE_EXTRACT_USER.format(
                        scene_id=scene_id, payload=span
                    ),
                    temperature=0.2,
                )

                # 强制覆盖 scene_id，避免模型自己编号错乱
                data["scene_id"] = scene_id
                # 原文片段以我们的切片为准，模型可能截错
                data["source_span"] = span[:2000]

                scene = parse_model(Scene, data)

                # 质控
                self._validate_scene(scene, span)

                return scene

            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                reason = _classify_error(exc)
                if attempt < self.max_retries:
                    self.stats.note_retry(reason)
                    logger.info(
                        "%s 第%d次校验未过（%s），重试", scene_id, attempt + 1, reason
                    )
                else:
                    logger.warning("%s 抽取失败: %s", scene_id, last_error)
                    self.stats.note_drop(reason)

        return None

    def _validate_scene(self, scene: Scene, span: str) -> None:
        """单场质控。任一项不通过就抛异常触发重试。"""
        # 1. 必填字段
        for f in REQUIRED_SCENE_FIELDS:
            if not getattr(scene, f, None):
                raise QualityGateError(f"缺少必填字段 {f}")

        # 2. 台词必填
        for i, line in enumerate(scene.dialogue):
            for f in REQUIRED_LINE_FIELDS:
                if not getattr(line, f, None):
                    raise QualityGateError(f"第{i+1}句台词缺少 {f}")

        # 3. 反编造：台词必须在原文里能找到
        fabricated = [
            ln.text for ln in scene.dialogue
            if ln.text and not _text_in_span(ln.text, span)
        ]
        if fabricated:
            raise QualityGateError(
                f"台词不在原文中（疑似编造）: {fabricated[:2]}"
            )

        # 4. 反编造：人物名必须在原文里能找到
        ghost = [c for c in scene.characters if c not in span]
        if ghost:
            raise QualityGateError(f"人物不在原文中（疑似编造）: {ghost[:3]}")

        # 5. 说话人必须是本场人物
        unknown_speakers = {
            ln.speaker for ln in scene.dialogue if ln.speaker not in scene.characters
        }
        if unknown_speakers:
            # 自动补进去而不是报错——这是可以安全修复的问题
            scene.characters.extend(sorted(unknown_speakers))

    # ------------------------------------------------------------ ③ 人物合并

    def merge_characters(self, scenes: list[Scene]) -> list[str]:
        """合并全剧人物，检测冲突。

        冲突定义：同一场里出现两个名字，其中一个是另一个的子串
        （如「林小满」与「小满」）。这通常是同一人物被抽成两个，
        也可能是真的两个不同角色——所以只报告，不自动合并。
        """
        all_names: dict[str, set[int]] = {}
        for idx, scene in enumerate(scenes, start=1):
            for name in scene.characters:
                all_names.setdefault(name, set()).add(idx)

        # 检测疑似同一人
        conflicts: list[str] = []
        names = sorted(all_names, key=len, reverse=True)
        for i, long_name in enumerate(names):
            for short_name in names[i + 1 :]:
                if short_name != long_name and short_name in long_name:
                    # 只有出现在同一场时才认为可疑
                    if all_names[long_name] & all_names[short_name]:
                        conflicts.append(f"{long_name} / {short_name}")

        self.stats.character_conflicts = conflicts
        if conflicts:
            msg = f"发现疑似人物重复 {len(conflicts)} 组: {conflicts[:5]}"
            if self.strict:
                raise QualityGateError(msg)
            logger.warning(msg)

        # 用模型做一次归一（离线模式下列表原样返回）
        if len(all_names) <= 1:
            return list(all_names)

        try:
            payload = "\n".join(
                f"{name}（出现在第 {sorted(idx)} 场）"
                for name, idx in sorted(all_names.items())
            )
            data = self.client.complete_json(
                prompts.CHARACTER_MERGE_SYSTEM,
                prompts.CHARACTER_MERGE_USER.format(payload=payload),
                temperature=0.0,
            )
            merged = [
                str(c.get("name", "")).strip()
                for c in data.get("characters", [])
                if c.get("name")
            ]
            # 安全阀：合并结果不能凭空多出名字
            allowed = set(all_names)
            merged = [m for m in merged if m in allowed]
            if merged:
                # 补回被模型漏掉的
                for name in all_names:
                    if name not in merged:
                        merged.append(name)
                return merged
        except Exception as exc:  # noqa: BLE001
            logger.warning("人物合并失败: %s，使用原始列表", exc)

        return list(all_names)

    # ------------------------------------------------------------ ④ 抽样复核

    def run_qa(self, spans: list[str], scenes: list[Scene]) -> None:
        """随机抽几场做漏抽检查。"""
        n = min(self.qa_sample, len(scenes))
        if n <= 0:
            return

        indices = random.sample(range(len(scenes)), n)
        for idx in indices:
            scene = scenes[idx]
            span = spans[idx] if idx < len(spans) else scene.source_span
            try:
                data = self.client.complete_json(
                    prompts.QA_CHECK_SYSTEM,
                    prompts.QA_CHECK_USER.format(
                        payload=span[:6000],
                        extracted_dialogue="\n".join(
                            f"{ln.speaker}: {ln.text}" for ln in scene.dialogue
                        ) or "（无）",
                        extracted_characters="、".join(scene.characters) or "（无）",
                    ),
                    temperature=0.0,
                )
                self.stats.qa_checked += 1
                miss_d = data.get("missing_dialogue") or []
                miss_c = data.get("missing_characters") or []
                if miss_d:
                    self.stats.qa_missing_dialogue += len(miss_d)
                    logger.info("%s 复核发现漏抽台词 %d 条", scene.scene_id, len(miss_d))
                if miss_c:
                    self.stats.qa_missing_characters += len(miss_c)
                    logger.info("%s 复核发现漏抽人物: %s", scene.scene_id, miss_c)
            except Exception as exc:  # noqa: BLE001
                logger.debug("%s 复核跳过: %s", scene.scene_id, exc)


# ---------------------------------------------------------------- 工具


def _classify_error(exc: Exception) -> str:
    """把异常归类，用于统计重试原因。"""
    msg = str(exc)
    if "JSON" in msg or "json" in msg or "解析" in msg:
        return "json_parse"
    if "编造" in msg:
        return "fabrication"
    if "必填" in msg or "缺少" in msg:
        return "missing_field"
    if "validation" in msg.lower() or "Validation" in type(exc).__name__:
        return "schema_invalid"
    return "other"


def _text_in_span(text: str, span: str) -> bool:
    """判断台词是否来自原文。

    做归一化比较：去空白、去标点，避免因为模型调整了空格而误判为编造。
    """
    if text in span:
        return True
    norm = lambda s: re.sub(r"[\s，。！？、；：""''（）()…—\-\.\,\!\?]", "", s)  # noqa: E731
    return norm(text) in norm(span)


_SCENE_HEADING_RE = re.compile(
    r"^\s*(?:(\d+)[\.、\)]\s*|第[一二三四五六七八九十百]+场\s*[：:]?\s*|"
    r"((?:内景|外景|内外景|INT|EXT)[\.．]?\s*.{0,40}?"
    r"(?:日|夜|黄昏|清晨|傍晚|白天|晚上|DAY|NIGHT|DAWN|DUSK))\s*$)",
    re.M,
)


def _rule_split_by_heading(text: str) -> list[str]:
    """按场景标题规则切分，返回每场原文。"""
    matches = list(_SCENE_HEADING_RE.finditer(text))
    if len(matches) < 2:
        return []

    spans: list[str] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        span = text[m.start() : end].strip()
        if span:
            spans.append(span)
    return spans


def _locate_spans(text: str, headings: list[str]) -> list[str]:
    """把模型给的标题映射回原文位置，切出各场片段。

    模型可能改写了标题文字，所以用"模糊定位"：先精确找，找不到就
    退化为按标题行本身切分。定位失败返回空，让调用方退回规则分场。
    """
    positions: list[int] = []
    for h in headings:
        if not h:
            continue
        pos = text.find(h)
        if pos == -1:
            # 退一步：只取标题里最关键的部分（地点词）
            core = re.sub(r"(内景|外景|INT|EXT|日|夜)\s*", "", h).strip(" .．:：")
            pos = text.find(core) if core else -1
        if pos != -1:
            positions.append(pos)

    if len(positions) < 2:
        return []

    positions = sorted(set(positions))
    spans: list[str] = []
    for i, pos in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        span = text[pos:end].strip()
        if span:
            spans.append(span)
    return spans
