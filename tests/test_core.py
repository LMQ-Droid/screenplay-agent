# -*- coding: utf-8 -*-
"""核心逻辑单元测试。

跑法：python tests/test_core.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from screenplay.agent import (  # noqa: E402
    ScreenplayAgent,
    _rule_split_by_heading,
    _text_in_span,
)
from screenplay.export import dialogue_frame, scenes_frame  # noqa: E402
from screenplay.llm import LLMClient, extract_json  # noqa: E402
from screenplay.schema import Emotion, Scene, Screenplay, parse_model  # noqa: E402

SAMPLE = (ROOT / "samples" / "demo.txt").read_text(encoding="utf-8")


# ---------------------------------------------------------------- JSON 容错


def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_with_fence():
    """模型常把 JSON 包在 markdown 代码块里。"""
    text = '```json\n{"a": 1, "b": [2]}\n```'
    assert extract_json(text) == {"a": 1, "b": [2]}


def test_extract_json_with_preamble():
    """模型常在前面加一句客气话。"""
    text = '好的，以下是结构化结果：\n{"scenes": []}'
    assert extract_json(text) == {"scenes": []}


def test_extract_json_trailing_comma():
    """尾逗号是模型最常犯的 JSON 错误。"""
    text = '{"a": 1, "b": 2,}'
    assert extract_json(text) == {"a": 1, "b": 2}


def test_extract_json_empty_raises():
    try:
        extract_json("")
    except ValueError:
        return
    raise AssertionError("空输入应当抛 ValueError")


def test_extract_json_no_object_raises():
    try:
        extract_json("这是一段没有 JSON 的文字")
    except ValueError:
        return
    raise AssertionError("无 JSON 时应抛 ValueError")


# ---------------------------------------------------------------- 分场


def test_rule_split_finds_three_scenes():
    spans = _rule_split_by_heading(SAMPLE)
    assert len(spans) == 3, f"expected 3 scenes, got {len(spans)}"
    assert "内景 咖啡馆 日" in spans[0]


def test_rule_split_keeps_dialogue_in_scene():
    """切分不能把台词切丢——这是最容易被忽略的漏抽来源。"""
    spans = _rule_split_by_heading(SAMPLE)
    joined = "".join(spans)
    for line in ["你等了很久吗", "我相信过", "你为什么要来"]:
        assert line in joined, f"台词丢失: {line}"


# ---------------------------------------------------------------- 反编造


def test_text_in_span_exact():
    assert _text_in_span("你等了很久吗？", SAMPLE)


def test_text_in_span_ignores_punctuation():
    """模型可能微调标点，不应因此判定为编造。"""
    assert _text_in_span("你等了很久吗", SAMPLE)


def test_text_in_span_rejects_fabrication():
    assert not _text_in_span("这杯咖啡是我爷爷留下的秘方", SAMPLE)


# ---------------------------------------------------------------- 场景校验


def test_validate_rejects_fabricated_dialogue():
    """编造的台词必须被质控拦下。"""
    agent = ScreenplayAgent(client=LLMClient(offline=True))
    span = _rule_split_by_heading(SAMPLE)[0]

    scene = parse_model(Scene, {
        "scene_id": "S01",
        "heading": "内景 咖啡馆 日",
        "characters": ["林小满"],
        "dialogue": [
            {"speaker": "林小满", "text": "我其实是外星人派来的卧底"}
        ],
        "source_span": span,
    })

    try:
        agent._validate_scene(scene, span)
    except Exception as exc:
        assert "编造" in str(exc)
        return
    raise AssertionError("编造台词未被拦截")


def test_validate_rejects_fabricated_character():
    agent = ScreenplayAgent(client=LLMClient(offline=True))
    span = _rule_split_by_heading(SAMPLE)[0]

    scene = parse_model(Scene, {
        "scene_id": "S01",
        "heading": "内景 咖啡馆 日",
        "characters": ["林小满", "张三丰"],
        "dialogue": [],
        "source_span": span,
    })

    try:
        agent._validate_scene(scene, span)
    except Exception as exc:
        assert "编造" in str(exc)
        return
    raise AssertionError("编造人物未被拦截")


def test_validate_accepts_faithful_scene():
    """忠实于原文的场景必须通过校验。"""
    agent = ScreenplayAgent(client=LLMClient(offline=True))
    span = _rule_split_by_heading(SAMPLE)[0]

    scene = parse_model(Scene, {
        "scene_id": "S01",
        "heading": "内景 咖啡馆 日",
        "characters": ["林小满", "陈默"],
        "dialogue": [
            {"speaker": "林小满", "text": "你等了很久吗？", "emotion": "焦虑", "intensity": 3},
        ],
        "source_span": span,
    })
    agent._validate_scene(scene, span)


def test_validate_missing_heading_rejected():
    agent = ScreenplayAgent(client=LLMClient(offline=True))
    scene = parse_model(Scene, {
        "scene_id": "S01", "heading": "x", "characters": [],
    })
    scene.heading = ""  # 模拟模型漏了标题
    try:
        agent._validate_scene(scene, "x")
    except Exception as exc:
        assert "heading" in str(exc) or "必填" in str(exc)
        return
    raise AssertionError("缺 heading 未被拦截")


# ---------------------------------------------------------------- 人物冲突


def test_character_conflict_detected():
    """同一场里「林小满」和「小满」应被识别为疑似重复。"""
    scenes = [
        Scene(scene_id="S01", heading="内景 咖啡馆 日",
              characters=["林小满", "小满", "陈默"]),
    ]
    agent = ScreenplayAgent(client=LLMClient(offline=True))
    agent.merge_characters(scenes)
    assert agent.stats.character_conflicts, "未检测到人物冲突"
    assert any("林小满" in c for c in agent.stats.character_conflicts)


# ---------------------------------------------------------------- 端到端


def test_offline_parse_produces_scenes():
    """离线模式必须能跑通全流程。"""
    agent = ScreenplayAgent(
        client=LLMClient(offline=True), qa_sample=0
    )
    sp = agent.parse(SAMPLE, title="demo")

    assert len(sp.scenes) == 3, f"expected 3 scenes, got {len(sp.scenes)}"
    assert sp.dialogue_count > 0, "未抽取到台词"
    assert len(sp.characters) >= 2, "人物抽取不足"


def test_offline_parse_extracts_known_dialogue():
    """关键台词必须被抽到——这是漏抽率的核心检验。"""
    agent = ScreenplayAgent(client=LLMClient(offline=True), qa_sample=0)
    sp = agent.parse(SAMPLE, title="demo")

    all_text = "".join(ln.text for s in sp.scenes for ln in s.dialogue)
    for expect in ["你等了很久吗", "我相信过", "你为什么要来"]:
        assert expect in all_text, f"漏抽台词: {expect}"


def test_offline_parse_no_fabrication():
    """离线结果的每句台词都应能在原文找到。"""
    agent = ScreenplayAgent(client=LLMClient(offline=True), qa_sample=0)
    sp = agent.parse(SAMPLE, title="demo")

    for s in sp.scenes:
        for ln in s.dialogue:
            assert _text_in_span(ln.text, SAMPLE), (
                f"抽取出原文不存在的台词: {ln.text}"
            )


def test_offline_characters_all_from_source():
    agent = ScreenplayAgent(client=LLMClient(offline=True), qa_sample=0)
    sp = agent.parse(SAMPLE, title="demo")
    for name in sp.characters:
        assert name in SAMPLE, f"人物 {name} 不在原文中"


# ---------------------------------------------------------------- 导出


def test_scenes_frame_shape():
    agent = ScreenplayAgent(client=LLMClient(offline=True), qa_sample=0)
    sp = agent.parse(SAMPLE, title="demo")
    df = scenes_frame(sp)
    assert len(df) == len(sp.scenes)
    assert "场次" in df.columns and "出场人物" in df.columns


def test_dialogue_frame_shape():
    agent = ScreenplayAgent(client=LLMClient(offline=True), qa_sample=0)
    sp = agent.parse(SAMPLE, title="demo")
    df = dialogue_frame(sp)
    assert len(df) == sp.dialogue_count
    for col in ["说话人", "台词", "情绪", "强度"]:
        assert col in df.columns, f"缺少列 {col}"


def test_dialogue_frame_intensity_range():
    """强度必须在 1-5 之间，否则配音无法按强度分级。"""
    agent = ScreenplayAgent(client=LLMClient(offline=True), qa_sample=0)
    sp = agent.parse(SAMPLE, title="demo")
    df = dialogue_frame(sp)
    if not df.empty:
        assert df["强度"].between(1, 5).all()


# ---------------------------------------------------------------- schema


def test_scene_dedupes_characters():
    s = Scene(scene_id="S01", heading="x",
              characters=["林小满", "陈默", "林小满"])
    assert s.characters == ["林小满", "陈默"], "人物未去重或顺序错乱"


def test_emotion_enum_rejects_unknown():
    try:
        Emotion("暴怒")
    except ValueError:
        return
    raise AssertionError("枚举应拒绝自创情绪值")


def test_screenplay_counts():
    sp = Screenplay(
        title="t",
        scenes=[
            Scene(scene_id="S01", heading="a", source_span="12345"),
            Scene(scene_id="S02", heading="b", source_span="678"),
        ],
    )
    assert len(sp.scenes) == 2
    assert sp.total_chars == 8


# ---------------------------------------------------------------- 抽样校验


def test_qa_normalize_strips_parenthetical():
    """校验器必须剥离句首表演提示，否则带提示的台词会被误判为漏抽。"""
    from screenplay import qa_check

    assert qa_check.normalize("（低声）你先坐") == "你先坐"
    assert qa_check.normalize("（快步走近）你等了很久吗？") == "你等了很久吗"


def test_qa_completeness_of_rule_engine():
    """规则引擎的完整度应当很高，但**允许漏掉叙述行内嵌的台词**。

    原文里有一句「林小满在对面坐下，没有急着喝。（轻声）我以为你不会来。」
    这句没有「人名：」前缀，规则引擎按冒号切分抓不到，是真模型的强项。
    所以这里断言「漏抽率低但不是 0」，并显式检查漏的就是这一句。

    这条测试的意义：把"规则引擎的能力边界"固化下来，
    以后接模型跑出更高完整度时，对比基线就是它。
    """
    import tempfile

    from screenplay import qa_check
    from screenplay.schema import dump_model

    agent = ScreenplayAgent(client=LLMClient(offline=True), qa_sample=0)
    sp = agent.parse(SAMPLE, title="demo")

    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(dump_model(sp), f, ensure_ascii=False)
        tmp = f.name

    data = json.loads(Path(tmp).read_text(encoding="utf-8"))
    Path(tmp).unlink(missing_ok=True)

    report = qa_check.compare(SAMPLE, data)

    # 规则引擎不应有编造内容
    assert report["extra_count"] == 0, "不应有编造台词"
    assert not report["extra_characters"], "不应有编造人物"
    assert report["scene_match"], "场次数应一致"

    # 完整度高，但预期会漏掉内嵌台词（1/16 条）
    assert report["missing_rate"] < 10, (
        f"规则引擎漏抽率过高: {report['missing_rate']}%"
    )
    if report["missing_samples"]:
        assert "我以为你不会来" in report["missing_samples"][0], (
            f"漏抽的应当是内嵌台词，实际: {report['missing_samples']}"
        )


def test_qa_recognizes_inline_dialogue():
    """校验器必须能识别叙述行内嵌的台词，否则会误报模型正确输出。"""
    from screenplay import qa_check

    text = "林小满坐下，没有急着喝。（轻声）我以为你不会来。"
    pairs = qa_check.count_source_dialogue(text)
    assert any("我以为你不会来" in content for _, content in pairs), (
        f"未识别内嵌台词: {pairs}"
    )


def test_qa_detects_fabrication():
    """人为加入一句原文没有的台词，校验器必须抓到。"""
    from screenplay import qa_check

    fake = {
        "characters": ["林小满", "陈默"],
        "scenes": [
            {
                "scene_id": "S01",
                "heading": "内景 咖啡馆 日",
                "dialogue": [
                    {"speaker": "林小满", "text": "你等了很久吗？"},
                    {"speaker": "林小满", "text": "这世界终将属于机器人"},
                ],
            }
        ],
    }
    r = qa_check.compare(SAMPLE, fake)
    assert r["extra_count"] >= 1, "未检出编造台词"


def test_qa_detects_missing():
    """只抽一句时，漏抽率应显著高于 0。"""
    from screenplay import qa_check

    partial = {
        "characters": ["林小满"],
        "scenes": [
            {"scene_id": "S01", "heading": "x",
             "dialogue": [{"speaker": "林小满", "text": "你等了很久吗？"}]}
        ],
    }
    r = qa_check.compare(SAMPLE, partial)
    assert r["missing_count"] > 0, "未检出漏抽"
    assert r["missing_rate"] > 50


# ---------------------------------------------------------------- runner


def main() -> int:
    tests = [(n, f) for n, f in globals().items()
             if n.startswith("test_") and callable(f)]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError as exc:
            print(f"  FAIL  {name}: {exc}")
            failed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {len(tests)} total")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
