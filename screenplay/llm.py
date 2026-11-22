# -*- coding: utf-8 -*-
"""LLM 客户端：OpenAI 兼容接口 + 离线规则回退。

为什么做成双模式：
1. 没有 API key 时项目仍然能完整跑通（演示、测试、CI 都不需要网络）；
2. mock 不是"假装"，它是一套规则解析器，正好可以当作**对照组基线**——
   用来量化"模型比规则强多少"，这本身就是一个可写进简历的指标；
3. 真实调用必然遇到超时、限流、返回非 JSON，这些处理逻辑在离线模式下
   由 mock 的可控故障注入来覆盖测试。
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any

import requests

logger = logging.getLogger(__name__)

# 默认配置。全部可由环境变量覆盖，密钥不进代码。
DEFAULT_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
DEFAULT_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "90"))
DEFAULT_RETRIES = int(os.getenv("LLM_RETRIES", "3"))


@dataclass
class LLMUsage:
    """累计 token 与调用统计，用于成本核算。"""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    failures: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "failures": self.failures,
        }


@dataclass
class LLMResponse:
    text: str
    raw: dict[str, Any] = field(default_factory=dict)


class LLMClient:
    """统一的模型调用入口。

    Args:
        api_key: 为空则自动进入离线模式
        base_url: 兼容 OpenAI 的接口地址（DeepSeek/通义/Kimi 等都可用）
        model: 模型名
        offline: 强制离线，优先级最高
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        offline: bool = False,
        timeout: int = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
    ) -> None:
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.usage = LLMUsage()

        self.offline = offline or not self.api_key
        if self.offline:
            logger.warning(
                "LLM 未配置 API key，进入离线规则模式（结果仅用于演示与测试）"
            )

    # ------------------------------------------------------------ 核心调用

    def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        json_mode: bool = True,
        max_tokens: int = 4000,
    ) -> LLMResponse:
        """发起一次对话补全。

        Args:
            json_mode: 要求模型返回严格 JSON（OpenAI 兼容的
                       response_format={"type": "json_object"}）
        """
        if self.offline:
            return LLMResponse(text=_offline_complete(system, user))

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            # 结构化抽取任务需要稳定输出，温度调低
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        last_exc: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                )
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise RuntimeError(
                        f"HTTP {resp.status_code}: {resp.text[:200]}"
                    )
                resp.raise_for_status()
                data = resp.json()

                self.usage.calls += 1
                usage = data.get("usage") or {}
                self.usage.prompt_tokens += int(usage.get("prompt_tokens", 0))
                self.usage.completion_tokens += int(usage.get("completion_tokens", 0))

                text = (data["choices"][0]["message"]["content"] or "").strip()
                return LLMResponse(text=text, raw=data)

            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                self.usage.failures += 1
                if attempt == self.retries:
                    break
                # 指数退避 + 抖动，避免多进程同时重试打爆接口
                wait = min(2 ** attempt, 10) + random.uniform(0, 0.5)
                logger.warning(
                    "LLM 调用失败(第%d次): %s，%.1fs 后重试", attempt, exc, wait
                )
                time.sleep(wait)

        raise RuntimeError(f"LLM 调用失败，已重试 {self.retries} 次") from last_exc

    def complete_json(self, system: str, user: str, **kwargs: Any) -> dict[str, Any]:
        """要求返回 JSON 对象，并做容错解析。"""
        resp = self.complete(system, user, json_mode=True, **kwargs)
        return extract_json(resp.text)


# ---------------------------------------------------------------- JSON 容错


def extract_json(text: str) -> dict[str, Any]:
    """从模型输出里抠出 JSON 对象。

    真实模型经常不老实：
    - 包一层 ```json 代码块
    - 前面加一句"好的，以下是结果："
    - 末尾多一个逗号

    这三类都在这处理，避免因为格式问题白白重试（每次重试都是钱）。
    """
    if not text or not text.strip():
        raise ValueError("模型返回为空")

    raw = text.strip()

    # 去掉 markdown 代码围栏
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.S)
    if fence:
        raw = fence.group(1).strip()

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # 退一步：截取第一个 { 到最后一个 } 之间的内容
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        candidate = raw[start : end + 1]
        # 顺手修掉常见的尾逗号
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError as exc:
            raise ValueError(f"无法解析 JSON: {exc}; 原文前200字={raw[:200]}") from exc

    raise ValueError(f"响应中找不到 JSON 对象；原文前200字={raw[:200]}")


# ---------------------------------------------------------------- 离线实现


def _offline_complete(system: str, user: str) -> str:
    """离线规则引擎：用正则从原文抽取结构。

    这不是"假数据"——它读的是真实输入的剧本文本，只是用规则代替模型。
    因此可以作为**基线对照**：模型的效果必须优于它才有意义。

    它刻意只做最基础的事：
    - 按场景标题切场
    - 按「人名：台词」切台词（含中文冒号与全角引号）
    - 情绪一律给默认值（规则判断不了，这正是模型该赢的地方）
    """
    # 从 user prompt 里找出被标记包裹的剧本正文
    body = _extract_payload(user)
    if not body:
        return json.dumps({"scenes": []}, ensure_ascii=False)

    # 如果要的是分场列表
    if "仅返回场次列表" in user or "scene_list" in system:
        scenes = _rule_split_scenes(body)
        return json.dumps(
            {"scenes": [{"index": i + 1, "heading": s["heading"]} for i, s in enumerate(scenes)]},
            ensure_ascii=False,
        )

    # 否则解析单场结构
    scene = _rule_parse_scene(body)
    return json.dumps(scene, ensure_ascii=False)


def _extract_payload(user: str) -> str:
    """从 prompt 中提取 <<<PAYLOAD>>> 包裹的正文。"""
    m = re.search(r"<<<PAYLOAD>>>(.*?)<<<END>>>", user, re.S)
    if m:
        return m.group(1).strip()

    # 兼容：prompt 里直接给了整段文本
    m = re.search(r"【剧本原文】\s*(.+)$", user, re.S)
    if m:
        return m.group(1).strip()
    return ""


# 场景标题的常见写法：1. 内景 咖啡馆 日 / INT. CAFE - DAY / 第一场
_SCENE_HEADING_RE = re.compile(
    r"^\s*(?:(\d+)[\.、\)]|第([一二三四五六七八九十百]+)场|"
    r"((?:内景|外景|内外景|INT|EXT)[\.．]?\s*.+?"
    r"(?:日|夜|黄昏|清晨|傍晚|白天|晚上|DAY|NIGHT|DAWN|DUSK))\s*$)",
    re.M,
)


def _rule_split_scenes(text: str) -> list[dict[str, str]]:
    """按场景标题把全文切成多场。"""
    matches = list(_SCENE_HEADING_RE.finditer(text))
    if not matches:
        # 没有明确场标，整篇当一场
        return [{"heading": "场景 1", "body": text}]

    scenes: list[dict[str, str]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        heading = m.group(0).strip()
        body = text[m.end() : end].strip()
        scenes.append({"heading": heading, "body": body})
    return scenes


# 台词：支持「张三：台词」「张三（低声）：台词」「张三:"台词"」
_DIALOGUE_RE = re.compile(
    r"^\s*([\u4e00-\u9fa5A-Za-z][\u4e00-\u9fa5A-Za-z0-9·]{0,11})"
    r"(?:[（(]([^）)]{1,20})[）)])?"
    r"\s*[：:]\s*(.+?)\s*$",
    re.M,
)

# 动作/描述行：以括号开头，或明显不是台词的行
_ACTION_RE = re.compile(r"^\s*[（(].*[）)]\s*$")


def _rule_parse_scene(span: str) -> dict[str, Any]:
    """把一场戏的文本解析成结构化 dict。"""
    lines = [ln for ln in span.splitlines() if ln.strip()]
    heading = "场景 1"
    body_lines: list[str] = []

    for ln in lines:
        if _SCENE_HEADING_RE.match(ln):
            heading = ln.strip()
        else:
            body_lines.append(ln)

    body = "\n".join(body_lines)

    dialogue: list[dict[str, Any]] = []
    actions: list[str] = []
    speakers: list[str] = []
    seen: set[str] = set()

    for ln in body_lines:
        m = _DIALOGUE_RE.match(ln)
        if m and not _ACTION_RE.match(ln):
            name, paren, text = m.group(1), m.group(2), m.group(3).strip()
            if len(text) < 1:
                continue
            # 台词里可能还嵌着表演提示，如「（低声）你先坐」
            # 把前缀括号剥离到 parenthetical，让 text 只留台词本体。
            # 不剥的话下游做字幕会带上括号，且动作行会重复记录这段提示。
            inner = re.match(r"^[（(]([^）)]{1,20})[）)]\s*(.+)$", text)
            if inner:
                paren = paren or inner.group(1)
                text = inner.group(2).strip()

            dialogue.append(
                {
                    "speaker": name,
                    "text": text,
                    "emotion": "未判定",
                    "intensity": 3,
                    "parenthetical": paren,
                }
            )
            if name not in seen:
                seen.add(name)
                speakers.append(name)
        else:
            actions.append(ln.strip())

    # 从标题里抠地点与时间
    location, time_of_day, is_interior = _parse_heading(heading)

    return {
        "scene_id": "S01",
        "heading": heading,
        "location": location,
        "time_of_day": time_of_day,
        "is_interior": is_interior,
        "characters": speakers,
        "action": " ".join(actions)[:500],
        "dialogue": dialogue,
        "emotion": "未判定",
        "shot_hints": [],
        "source_span": span[:2000],
    }


def _parse_heading(heading: str) -> tuple[str, str, bool | None]:
    """从「内景 咖啡馆 日」解析出地点、时间、内外景。"""
    interior: bool | None = None
    if "内景" in heading or heading.upper().startswith("INT"):
        interior = True
    elif "外景" in heading or heading.upper().startswith("EXT"):
        interior = False

    time_map = {
        "日": "日", "夜": "夜", "黄昏": "黄昏", "清晨": "清晨",
        "傍晚": "傍晚", "白天": "日", "晚上": "夜",
        "DAY": "日", "NIGHT": "夜", "DAWN": "清晨", "DUSK": "黄昏",
    }
    time_of_day = "未指定"
    for key, val in time_map.items():
        if key in heading:
            time_of_day = val
            break

    cleaned = re.sub(
        r"(内景|外景|内外景|INT|EXT)[\.．]?", "", heading, flags=re.I
    )
    for key in time_map:
        cleaned = cleaned.replace(key, "")
    cleaned = re.sub(r"^\s*\d+[\.、\)]\s*", "", cleaned)
    cleaned = cleaned.replace("-", " ").strip(" .．:：")
    location = cleaned or "未指定"

    return location, time_of_day, interior
