# -*- coding: utf-8 -*-
"""结构化剧本的数据契约。

设计原则：**结构即校验**。
模型输出的自由度越高，下游越难用。所以这里把「一场戏应该长什么样」
写死成 schema，模型必须按这个形状返回，缺字段能被自动发现并重试。

字段选择依据下游内容生产（分镜/配音/剪辑）的实际需要：
- shot_hint: 镜头提示（景别、运动、氛围）——给分镜师
- emotion / intensity: 情绪与强度——给配音与配乐
- speaker_text 与 action: 台词与动作分离——给字幕与画面
"""
from __future__ import annotations

from enum import Enum
from typing import Any

import pydantic
from pydantic import BaseModel, Field

# pydantic v1 与 v2 的校验器 API 不同。这里做一层兼容，
# 避免换环境（或部署到云端）时因版本差异直接崩掉。
_PYDANTIC_V2 = int(pydantic.VERSION.split(".")[0]) >= 2

if _PYDANTIC_V2:
    from pydantic import field_validator as _validator

    def validate_field(*fields: str):
        return _validator(*fields)

    def parse_model(model: type[BaseModel], data: dict[str, Any]) -> Any:
        return model.model_validate(data)

    def dump_model(model: BaseModel) -> dict[str, Any]:
        return model.model_dump(mode="json")
else:
    from pydantic import validator as _validator  # type: ignore

    def validate_field(*fields: str):
        return _validator(*fields, allow_reuse=True)

    def parse_model(model: type[BaseModel], data: dict[str, Any]) -> Any:
        return model.parse_obj(data)  # type: ignore[attr-defined]

    def dump_model(model: BaseModel) -> dict[str, Any]:
        return model.dict()  # type: ignore[attr-defined]


class ShotSize(str, Enum):
    """景别。限定枚举值，避免模型自由发挥写出「中全景」这种不可用的词。"""

    EXTREME_WIDE = "大远景"
    WIDE = "远景"
    FULL = "全景"
    MEDIUM = "中景"
    MEDIUM_CLOSE = "中近景"
    CLOSE = "近景"
    EXTREME_CLOSE = "特写"
    UNKNOWN = "未指定"


class CameraMove(str, Enum):
    """运镜方式。"""

    STATIC = "固定"
    PAN = "摇"
    TILT = "俯仰"
    DOLLY = "推拉"
    TRACK = "跟"
    HANDHELD = "手持"
    AERIAL = "航拍"
    UNKNOWN = "未指定"


class Emotion(str, Enum):
    """情绪标签。枚举而非自由文本，下游才能做筛选和统计。"""

    CALM = "平静"
    HAPPY = "喜悦"
    SAD = "悲伤"
    ANGRY = "愤怒"
    FEAR = "恐惧"
    SURPRISE = "惊讶"
    ANXIOUS = "焦虑"
    TENDER = "温情"
    COLD = "冷漠"
    EXCITED = "激昂"
    UNKNOWN = "未判定"


class DialogueLine(BaseModel):
    """一句台词。"""

    speaker: str = Field(..., min_length=1, description="说话人，必须来自剧本原文")
    text: str = Field(..., min_length=1, description="台词内容，逐字来自原文")
    emotion: Emotion = Field(default=Emotion.UNKNOWN, description="该句的情绪")
    intensity: int = Field(default=3, ge=1, le=5, description="情绪强度 1-5")
    parenthetical: str | None = Field(
        default=None, description="台词前的表演提示，如（低声）"
    )


class ShotHint(BaseModel):
    """镜头提示，供分镜师使用。"""

    shot_size: ShotSize = Field(default=ShotSize.UNKNOWN)
    camera_move: CameraMove = Field(default=CameraMove.UNKNOWN)
    description: str = Field(default="", description="画面描述，不超过 60 字")
    lighting: str | None = Field(default=None, description="光线/氛围提示")


class Scene(BaseModel):
    """一场戏。"""

    scene_id: str = Field(..., min_length=1, description="场次编号，如 S01")
    heading: str = Field(
        ..., min_length=1,
        description="场景标题，如「内景 咖啡馆 日」，逐字来自原文或标准化后"
    )
    location: str = Field(default="未指定", description="地点")
    time_of_day: str = Field(default="未指定", description="日/夜/黄昏等")
    is_interior: bool | None = Field(default=None, description="内景为 True")
    characters: list[str] = Field(
        default_factory=list, description="本场出场人物，必须来自原文"
    )
    action: str = Field(default="", description="本场动作/画面描述")
    dialogue: list[DialogueLine] = Field(default_factory=list)
    emotion: Emotion = Field(default=Emotion.UNKNOWN, description="本场主导情绪")
    shot_hints: list[ShotHint] = Field(default_factory=list, description="镜头提示")
    source_span: str = Field(
        default="", description="本场对应的原文片段，用于溯源与核查漏抽"
    )

    @validate_field("characters")
    @classmethod
    def _dedupe_characters(cls, v: list[str]) -> list[str]:
        """人物去重但保持顺序——顺序影响下游「谁先出场」的判断。"""
        seen: set[str] = set()
        out: list[str] = []
        for name in v:
            name = name.strip()
            if name and name not in seen:
                seen.add(name)
                out.append(name)
        return out

    @validate_field("heading")
    @classmethod
    def _heading_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("heading 不能为空")
        return v.strip()


class Screenplay(BaseModel):
    """整部剧本的结构化结果。"""

    title: str = Field(default="未命名剧本")
    characters: list[str] = Field(
        default_factory=list, description="全剧人物，由各场合并而来"
    )
    scenes: list[Scene] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def dialogue_count(self) -> int:
        return sum(len(s.dialogue) for s in self.scenes)

    @property
    def total_chars(self) -> int:
        return sum(len(s.source_span) for s in self.scenes)


# 必填字段清单：质控阶段据此判断「模型是否漏了关键信息」。
# 只列真正影响下游生产的字段——heading 缺了就没法分镜，intensity 缺了
# 可以给默认值，不必重试。区分这一点能显著减少无谓的重试次数。
REQUIRED_SCENE_FIELDS = ("scene_id", "heading")
REQUIRED_LINE_FIELDS = ("speaker", "text")
