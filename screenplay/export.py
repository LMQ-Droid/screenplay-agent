# -*- coding: utf-8 -*-
"""导出：结构化结果 → 下游可用的表格 / JSON。

导出三种形态，对应三类下游：
- scenes.xlsx     给制片/统筹看的场次表（一场景一行）
- dialogue.xlsx   给配音/字幕用的台词表（一句一行，带情绪与强度）
- screenplay.json 给程序消费的完整嵌套结构

人工复核闸门：导出前必须显式确认，避免错误结果直接流进下游生产。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .schema import Screenplay, dump_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- DataFrame


def scenes_frame(sp: Screenplay) -> pd.DataFrame:
    """场次表：一场景一行。"""
    rows: list[dict[str, Any]] = []
    for s in sp.scenes:
        rows.append(
            {
                "场次": s.scene_id,
                "场景标题": s.heading,
                "地点": s.location,
                "时间": s.time_of_day,
                "内外景": "内景" if s.is_interior else ("外景" if s.is_interior is False else "未指定"),
                "出场人物": "、".join(s.characters),
                "人物数": len(s.characters),
                "台词数": len(s.dialogue),
                "主导情绪": s.emotion.value,
                "动作描述": s.action,
                "镜头提示数": len(s.shot_hints),
                "原文长度": len(s.source_span),
            }
        )
    return pd.DataFrame(rows)


def dialogue_frame(sp: Screenplay) -> pd.DataFrame:
    """台词表：一句一行，这是配音与剪辑真正要用的粒度。"""
    rows: list[dict[str, Any]] = []
    for s in sp.scenes:
        for i, ln in enumerate(s.dialogue, start=1):
            hint = s.shot_hints[0] if s.shot_hints else None
            rows.append(
                {
                    "场次": s.scene_id,
                    "场景标题": s.heading,
                    "句序": i,
                    "说话人": ln.speaker,
                    "台词": ln.text,
                    "表演提示": ln.parenthetical or "",
                    "情绪": ln.emotion.value,
                    "强度": ln.intensity,
                    "建议景别": hint.shot_size.value if hint else "未指定",
                    "建议运镜": hint.camera_move.value if hint else "未指定",
                    "字数": len(ln.text),
                }
            )
    return pd.DataFrame(rows)


def shots_frame(sp: Screenplay) -> pd.DataFrame:
    """镜头提示表：给分镜师。"""
    rows: list[dict[str, Any]] = []
    for s in sp.scenes:
        for i, h in enumerate(s.shot_hints, start=1):
            rows.append(
                {
                    "场次": s.scene_id,
                    "序号": i,
                    "景别": h.shot_size.value,
                    "运镜": h.camera_move.value,
                    "画面描述": h.description,
                    "光线氛围": h.lighting or "",
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 导出


def export_all(
    sp: Screenplay,
    out_dir: Path,
    run_id: str | None = None,
) -> dict[str, Path]:
    """导出全部产物，返回路径字典。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    paths: dict[str, Path] = {}

    # JSON：完整嵌套结构
    json_path = out_dir / f"screenplay_{run_id}.json"
    json_path.write_text(
        json.dumps(dump_model(sp), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths["json"] = json_path

    # Excel：多 sheet 一个文件，方便制片直接看
    xlsx_path = out_dir / f"screenplay_{run_id}.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        scenes_frame(sp).to_excel(writer, sheet_name="场次表", index=False)
        dialogue_frame(sp).to_excel(writer, sheet_name="台词表", index=False)
        shots = shots_frame(sp)
        if not shots.empty:
            shots.to_excel(writer, sheet_name="镜头提示", index=False)
        # 质控信息单独一页，便于复核人快速判断质量
        pd.DataFrame(
            [{"指标": k, "值": str(v)} for k, v in _qc_rows(sp).items()]
        ).to_excel(writer, sheet_name="质控信息", index=False)
    paths["xlsx"] = xlsx_path

    return paths


def _qc_rows(sp: Screenplay) -> dict[str, Any]:
    meta = sp.meta or {}
    stats = meta.get("stats") or {}
    usage = meta.get("usage") or {}
    return {
        "剧本标题": sp.title,
        "场次数": len(sp.scenes),
        "人物数": len(sp.characters),
        "台词总数": sp.dialogue_count,
        "成功场次": stats.get("scenes_ok"),
        "失败场次": stats.get("scenes_failed"),
        "重试次数": stats.get("scenes_retried"),
        "人物冲突": "；".join(stats.get("character_conflicts") or []) or "无",
        "抽样复核场次": stats.get("qa_checked"),
        "复核漏抽台词": stats.get("qa_missing_dialogue"),
        "复核漏抽人物": stats.get("qa_missing_characters"),
        "模型调用次数": usage.get("calls"),
        "消耗token": usage.get("total_tokens"),
        "运行模式": "离线规则" if meta.get("offline") else "在线模型",
    }


# ---------------------------------------------------------------- 复核闸门


class ReviewGate:
    """人工复核闸门。

    设计意图：自动化结果在进入下游生产前，必须有一个人明确点头。
    这不是流程冗余——结构化抽取的漏抽率通常在 5%~20%，
    直接喂给配音或剪辑会造成返工。
    """

    def __init__(self, require_review: bool = True) -> None:
        self.require_review = require_review
        self.approved = not require_review

    def summary(self, sp: Screenplay) -> str:
        """生成给复核人看的摘要。"""
        qc = _qc_rows(sp)
        lines = ["=" * 56, "结构化结果复核摘要", "=" * 56]
        for k, v in qc.items():
            lines.append(f"  {k:14s}: {v}")

        # 标出需要重点看的场次
        problems: list[str] = []
        for s in sp.scenes:
            if not s.dialogue and s.action:
                problems.append(f"  {s.scene_id}: 无台词（纯动作场，确认是否正常）")
            if not s.characters:
                problems.append(f"  {s.scene_id}: 未识别到人物")
            if s.emotion.value == "未判定":
                problems.append(f"  {s.scene_id}: 情绪未判定")
        if problems:
            lines.append("\n需重点复核：")
            lines.extend(problems[:15])

        lines.append("=" * 56)
        return "\n".join(lines)

    def confirm(self, auto_approve: bool = False) -> bool:
        """请求人工确认。auto_approve 用于批处理场景。"""
        if auto_approve:
            self.approved = True
            logger.info("已跳过人工复核（--yes）")
            return True
        if not self.require_review:
            self.approved = True
            return True

        try:
            answer = input("\n确认导出？(y/N) ").strip().lower()
        except EOFError:
            answer = ""
        self.approved = answer in {"y", "yes"}
        return self.approved


# ---------------------------------------------------------------- 日志


def write_run_log(sp: Screenplay, out_dir: Path, extra: dict[str, Any] | None = None) -> Path:
    """把本次运行的质控数据追加到 JSONL 日志。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run_log.jsonl"
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "title": sp.title,
        "scenes": len(sp.scenes),
        "characters": len(sp.characters),
        "dialogue": sp.dialogue_count,
        **_qc_rows(sp),
        **(extra or {}),
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return log_path
