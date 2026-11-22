# -*- coding: utf-8 -*-
"""抽样校验：量化完整度与漏抽率。

为什么需要独立的校验工具：
结构化抽取的"看起来对"和"实际完整"是两件事。没有量化指标，
就无法判断改提示词是变好了还是变差了——只能凭感觉。

这里用**不依赖模型**的确定性方法做基准：
- 规则法从原文里数出台词行（"人名：内容" 模式）
- 与结构化结果做集合比对，算出漏抽率

它比人眼快、比模型稳，适合放进 CI 做回归。

用法：
    python -m screenplay.qa_check samples/demo.txt out/screenplay_xxx.json
    python -m screenplay.qa_check samples/demo.txt out/xxx.json --verbose
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# 台词行的识别规则：与 llm.py 的规则引擎保持独立实现，
# 避免"用抽取器自己的规则验证它自己"这种循环论证。
_DIALOGUE_LINE_RE = re.compile(
    r"^\s*([\u4e00-\u9fa5A-Za-z][\u4e00-\u9fa5A-Za-z0-9·]{0,11})"
    r"(?:[（(][^）)]{1,20}[）)])?"
    r"\s*[：:]\s*(.+?)\s*$",
    re.M,
)

_SCENE_HEADING_RE = re.compile(
    r"^\s*(?:\d+[\.、\)]\s*|第[一二三四五六七八九十百]+场|"
    r"(?:内景|外景|内外景|INT|EXT)[\.．]?)",
    re.M,
)

# 内嵌在叙述行里的台词：叙述。（提示）台词。
# 例：「林小满在对面坐下，没有急着喝。（轻声）我以为你不会来。」
# 这种写法校验器必须能识别，否则会把模型正确抽出的台词误报成"多抽"。
_INLINE_DIALOGUE_RE = re.compile(r"[。！？…][（(][^）)]{1,20}[）)]\s*(.+?)\s*$")

_PUNCT = re.compile(r"[\s，。！？、；：""''（）()…—\-\.\,\!\?\:;\"']")


def normalize(s: str) -> str:
    """归一化：去标点空白，用于宽松匹配。

    注意这里还要去掉句首的表演提示「（低声）」——结构化结果会把提示
    单独存进 parenthetical 字段，台词本体不含括号。如果校验时不剥离，
    每一条带提示的台词都会被误判成「漏抽 + 编造」。
    """
    s = re.sub(r"^[（(][^）)]{1,20}[）)]\s*", "", s.strip())
    return _PUNCT.sub("", s).strip()


def count_source_dialogue(text: str) -> list[tuple[str, str]]:
    """从原文中提取 (说话人, 台词) 基准集合。

    识别两种写法：
    1. 标准格式「人名：台词」
    2. 叙述行内嵌的说法「叙述。（提示）台词。」——这种没有冒号，
       需要靠行尾的括号提示定位。不处理它会把正确台词误报成编造。
    """
    out: list[tuple[str, str]] = []
    seen_texts: set[str] = set()

    for m in _DIALOGUE_LINE_RE.finditer(text):
        speaker, content = m.group(1), m.group(2).strip()
        if len(normalize(content)) < 1:
            continue
        if speaker in {"内景", "外景", "INT", "EXT"}:
            continue
        out.append((speaker, content))
        seen_texts.add(normalize(content))

    # 补充：识别叙述行内嵌的台词
    for line in text.splitlines():
        line = line.strip()
        if not line or _DIALOGUE_LINE_RE.match(line):
            continue
        m = _INLINE_DIALOGUE_RE.search(line)
        if not m:
            continue
        content = m.group(1).strip()
        norm = normalize(content)
        if len(norm) < 2 or norm in seen_texts:
            continue
        # 说话人未知时用占位符——校验只比对台词内容，不依赖说话人
        out.append(("", content))
        seen_texts.add(norm)

    return out


def count_source_scenes(text: str) -> int:
    """数原文中的场景标题数。"""
    n = len(_SCENE_HEADING_RE.findall(text))
    return n if n >= 2 else 1


def load_result(json_path: Path) -> dict[str, Any]:
    return json.loads(json_path.read_text(encoding="utf-8"))


def compare(source_text: str, result: dict[str, Any]) -> dict[str, Any]:
    """比对原文与结构化结果，算漏抽率。"""
    src_pairs = count_source_dialogue(source_text)
    src_scenes = count_source_scenes(source_text)

    extracted: list[tuple[str, str]] = []
    for scene in result.get("scenes", []):
        for ln in scene.get("dialogue", []):
            extracted.append((ln.get("speaker", ""), ln.get("text", "")))

    src_norm = {(sp, normalize(tx)) for sp, tx in src_pairs}
    ext_norm = {(normalize(sp), normalize(tx)) for sp, tx in extracted}

    # 按台词内容匹配（说话人可能有别名，内容才是硬证据）
    src_by_text = {normalize(tx): (sp, tx) for sp, tx in src_pairs}
    ext_texts = {normalize(tx) for _, tx in extracted}

    missing = [v for k, v in src_by_text.items() if k not in ext_texts]
    extra = [tx for _, tx in extracted if normalize(tx) not in src_by_text]

    src_chars = {sp for sp, _ in src_pairs if sp}  # 跳过内嵌台词的空说话人
    ext_chars = set(result.get("characters", []))
    missing_chars = sorted(src_chars - ext_chars)
    extra_chars = sorted(ext_chars - src_chars)

    total = len(src_pairs) or 1
    return {
        "source_scenes": src_scenes,
        "result_scenes": len(result.get("scenes", [])),
        "scene_match": src_scenes == len(result.get("scenes", [])),
        "source_dialogue": len(src_pairs),
        "result_dialogue": len(extracted),
        "missing_count": len(missing),
        "missing_rate": round(len(missing) / total * 100, 2),
        "completeness": round((total - len(missing)) / total * 100, 2),
        "missing_samples": [f"{sp}: {tx[:30]}" for sp, tx in missing[:8]],
        "extra_count": len(extra),
        "extra_samples": [tx[:40] for tx in extra[:8]],
        "missing_characters": missing_chars,
        "extra_characters": extra_chars,
        "source_characters": sorted(src_chars),
        "result_characters": sorted(ext_chars),
    }


def format_report(r: dict[str, Any]) -> str:
    lines = ["=" * 60, "结构化质量校验报告", "=" * 60]

    def row(k: str, v: Any, flag: str = "") -> None:
        lines.append(f"  {k:16s}: {v} {flag}")

    row("原文场次数", r["source_scenes"])
    row("结果场次数", r["result_scenes"], "OK" if r["scene_match"] else "!! 不一致")
    lines.append("")
    row("原文台词数", r["source_dialogue"])
    row("抽取台词数", r["result_dialogue"])
    row("漏抽条数", r["missing_count"], "!!" if r["missing_count"] else "OK")
    row("漏抽率", f"{r['missing_rate']}%",
        "!!" if r["missing_rate"] > 5 else "OK")
    row("完整度", f"{r['completeness']}%")
    lines.append("")
    row("多抽条数", r["extra_count"], "!! 疑似编造" if r["extra_count"] else "OK")

    if r["missing_samples"]:
        lines.append("\n  漏抽样例:")
        for s in r["missing_samples"]:
            lines.append(f"    - {s}")
    if r["extra_samples"]:
        lines.append("\n  多抽样例（原文不存在，需核查）:")
        for s in r["extra_samples"]:
            lines.append(f"    - {s}")

    lines.append("")
    row("原文人物", "、".join(r["source_characters"]) or "无")
    row("结果人物", "、".join(r["result_characters"]) or "无")
    if r["missing_characters"]:
        row("漏抽人物", "、".join(r["missing_characters"]), "!!")
    if r["extra_characters"]:
        row("多出人物", "、".join(r["extra_characters"]), "!! 疑似编造")

    lines.append("=" * 60)

    # 给出可执行的判定
    if r["missing_rate"] <= 5 and not r["extra_count"] and not r["extra_characters"]:
        lines.append("结论: 通过（漏抽率 ≤5%，无编造）")
    else:
        lines.append("结论: 需复核（见上方标 !! 的项）")
    lines.append("=" * 60)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="抽样校验：完整度与漏抽率")
    parser.add_argument("source", help="原始剧本文件")
    parser.add_argument("result", help="结构化结果 JSON")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出")
    parser.add_argument("--max-missing-rate", type=float, default=5.0,
                        help="漏抽率上限百分比，超过则返回非零")
    args = parser.parse_args(argv)

    src_path, res_path = Path(args.source), Path(args.result)
    if not src_path.is_file():
        print(f"ERROR: 原文不存在: {src_path}", file=sys.stderr)
        return 1
    if not res_path.is_file():
        print(f"ERROR: 结果不存在: {res_path}", file=sys.stderr)
        return 1

    text = src_path.read_text(encoding="utf-8")
    result = load_result(res_path)
    report = compare(text, result)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(format_report(report))

    return 0 if report["missing_rate"] <= args.max_missing_rate else 2


if __name__ == "__main__":
    raise SystemExit(main())
