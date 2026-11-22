# -*- coding: utf-8 -*-
"""A/B 对比：规则引擎 vs 真实模型。

这是本项目最有说服力的一个数字来源——不是"模型效果不错"，
而是"在同一个剧本上，规则引擎做不到的三件事，模型做到了几件"。

用法：
    python compare.py                 # 离线 vs 在线（需要有 key）
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from screenplay import qa_check  # noqa: E402
from screenplay.agent import ScreenplayAgent  # noqa: E402
from screenplay.llm import LLMClient  # noqa: E402
from screenplay.schema import dump_model  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

SAMPLE = ROOT / "samples" / "demo.txt"


def measure(sp, source: str, label: str) -> dict:
    """跑一次结构化，收集各项指标。"""
    data = dump_model(sp)
    report = qa_check.compare(source, data)
    stats = (sp.meta or {}).get("stats") or {}
    usage = (sp.meta or {}).get("usage") or {}

    # 情绪判定率：非「未判定」的台词占比
    lines = [ln for s in sp.scenes for ln in s.dialogue]
    judged = [ln for ln in lines if ln.emotion.value != "未判定"]
    emotion_rate = len(judged) / len(lines) * 100 if lines else 0.0

    # 情绪多样性：用了多少种不同情绪（单一标签说明没真正分析）
    distinct = len({ln.emotion.value for ln in judged}) if judged else 0

    shots = sum(len(s.shot_hints) for s in sp.scenes)

    return {
        "label": label,
        "scenes": len(sp.scenes),
        "dialogue": sp.dialogue_count,
        "characters": len(sp.characters),
        "missing_rate": report["missing_rate"],
        "extra_count": report["extra_count"],
        "emotion_judged_rate": round(emotion_rate, 1),
        "emotion_distinct": distinct,
        "shot_hints": shots,
        "retries": stats.get("scenes_retried", 0),
        "calls": usage.get("calls", 0),
        "tokens": usage.get("total_tokens", 0),
    }


def main() -> int:
    if not SAMPLE.exists():
        print(f"示例剧本不存在: {SAMPLE}")
        return 1

    source = SAMPLE.read_text(encoding="utf-8")

    api_key = os.getenv("LLM_API_KEY", "")
    if not api_key:
        print("未设置 LLM_API_KEY，只能跑规则引擎。")
        print("用法: $env:LLM_API_KEY='sk-xxx'; python compare.py")
        return 1

    results = []

    print("=" * 68)
    print("跑规则引擎（基线）...")
    print("=" * 68)
    rule_agent = ScreenplayAgent(client=LLMClient(offline=True), qa_sample=0)
    rule_sp = rule_agent.parse(source, title="demo")
    results.append(measure(rule_sp, source, "规则引擎"))

    print("\n" + "=" * 68)
    print("跑真实模型...")
    print("=" * 68)
    llm_agent = ScreenplayAgent(
        client=LLMClient(api_key=api_key), qa_sample=0
    )
    llm_sp = llm_agent.parse(source, title="demo")
    results.append(measure(llm_sp, source, "真实模型"))

    # 并排展示
    print("\n" + "=" * 68)
    print("对比结果")
    print("=" * 68)

    keys = [
        ("场次识别", "scenes", ""),
        ("台词抽取", "dialogue", ""),
        ("人物识别", "characters", ""),
        ("漏抽率", "missing_rate", "%"),
        ("编造内容", "extra_count", ""),
        ("情绪判定率", "emotion_judged_rate", "%"),
        ("情绪种类数", "emotion_distinct", ""),
        ("镜头提示条数", "shot_hints", ""),
        ("重试次数", "retries", ""),
        ("模型调用", "calls", ""),
        ("token 消耗", "tokens", ""),
    ]

    header = f"{'指标':<16} {'规则引擎':>14} {'真实模型':>14}   说明"
    print(header)
    print("-" * 68)

    r0, r1 = results[0], results[1]
    notes = {
        "missing_rate": "原文有但没抽到的台词占比",
        "extra_count": "抽出的原文不存在的内容（必须为 0）",
        "emotion_judged_rate": "能判定情绪的台词占比",
        "emotion_distinct": "单一标签说明没真正分析",
        "shot_hints": "原文没有的信息，只能推理得出",
        "tokens": "按实际接口返回统计",
    }

    for name, key, unit in keys:
        v0, v1 = r0[key], r1[key]
        s0 = f"{v0}{unit}" if v0 is not None else "-"
        s1 = f"{v1}{unit}" if v1 is not None else "-"

        # 标记改善
        mark = ""
        if key == "emotion_judged_rate" and v1 > v0:
            mark = " ↑模型强"
        elif key == "shot_hints" and v1 > v0:
            mark = " ↑模型强"
        elif key == "emotion_distinct" and v1 > v0:
            mark = " ↑模型强"
        elif key == "missing_rate" and v1 < v0:
            mark = " ↓模型更好"
        elif key == "extra_count" and v1 > v0:
            mark = " ↑模型更差"

        note = notes.get(key, "")
        print(f"{name:<16} {s0:>14} {s1:>14}   {note}{mark}")

    print("=" * 68)

    # 结论
    print("\n结论：")
    if r1["extra_count"] > 0:
        print(f"  ⚠️ 模型输出了 {r1['extra_count']} 条原文不存在的内容，需人工核查")
    if r1["shot_hints"] > 0 and r0["shot_hints"] == 0:
        print(f"  ✓ 镜头提示：规则引擎无法生成，模型产出 {r1['shot_hints']} 条")
    if r1["emotion_judged_rate"] > r0["emotion_judged_rate"]:
        print(
            f"  ✓ 情绪判定：{r0['emotion_judged_rate']}% → "
            f"{r1['emotion_judged_rate']}%"
            f"（用了 {r1['emotion_distinct']} 种不同标签）"
        )
    if r1["missing_rate"] <= 5 and r0["missing_rate"] > 5:
        print(
            f"  ✓ 完整度：漏抽率 {r0['missing_rate']}% → {r1['missing_rate']}%"
        )
    if r1["tokens"] > 0:
        print(f"  · 成本：{r1['calls']} 次调用 / {r1['tokens']:,} tokens")

    out = ROOT / "out" / "compare.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n明细已保存: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
