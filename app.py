# -*- coding: utf-8 -*-
"""Web 应用层：把 CLI 的能力包装成可交互页面。

设计取舍（和 CLI 版的关键差异）：

1. **不默认走离线模式**。CLI 用离线模式是为了测试；网页上面向真实用户，
   离线规则算不出情绪和镜头提示，界面上会是一堆"未判定"，反而像没做完。
   所以网页要求填 API key，并明确告诉用户 key 只存在当前会话。

2. **结果要能对照看**。结构化抽取最大的疑问是"你抽得对不对"，
   所以页面提供「原剧本 / 结构化结果」双栏对照，而不是只给一张表。

3. **质控数字要摆在明面上**。漏抽率、编造数、重试次数直接显示成指标卡。
   这些数字不达标时，用户应该能立刻看到。

4. **密钥不落盘**。只放 st.session_state，刷新页面即丢；
   项目里不保存、不写日志、不进 git。
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from screenplay import qa_check  # noqa: E402
from screenplay.agent import QualityGateError, ScreenplayAgent  # noqa: E402
from screenplay.export import (  # noqa: E402
    dialogue_frame,
    scenes_frame,
    shots_frame,
)
from screenplay.llm import LLMClient  # noqa: E402
from screenplay.schema import dump_model  # noqa: E402

st.set_page_config(
    page_title="剧本结构化解析",
    page_icon="🎬",
    layout="wide",
)

# 常见服务商的接口地址，省去用户查文档
PROVIDERS = {
    "DeepSeek（推荐，便宜）": ("https://api.deepseek.com/v1", "deepseek-chat"),
    "通义千问": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus"),
    "Kimi (Moonshot)": ("https://api.moonshot.cn/v1", "moonshot-v1-8k"),
    "智谱 GLM": ("https://open.bigmodel.cn/api/paas/v4", "glm-4-flash"),
    "OpenAI": ("https://api.openai.com/v1", "gpt-4o-mini"),
    "自定义": ("", ""),
}


# ---------------------------------------------------------------- 侧边栏：配置


def sidebar_config() -> dict:
    """收集模型配置。密钥只存 session_state，不写盘。"""
    with st.sidebar:
        st.markdown("### ⚙️ 模型配置")

        provider = st.selectbox("服务商", list(PROVIDERS.keys()), index=0)
        base_url, default_model = PROVIDERS[provider]

        if provider == "自定义":
            base_url = st.text_input(
                "接口地址", value="", placeholder="https://api.xxx.com/v1"
            )
            model = st.text_input("模型名", value="", placeholder="模型标识")
        else:
            model = st.text_input("模型名", value=default_model)

        api_key = st.text_input(
            "API Key",
            type="password",
            placeholder="sk-...",
            help="只保存在当前浏览器会话中，刷新即失效，不会上传到任何服务器",
        )

        with st.expander("高级选项"):
            retries = st.number_input(
                "单场重试次数", 0, 5, 2,
                help="校验不通过（缺字段/编造内容）时重试的次数",
            )
            qa_sample = st.number_input(
                "抽样复核场次", 0, 10, 3,
                help="随机抽几场做漏抽检查，0 表示关闭",
            )

        st.divider()
        st.caption(
            "🔒 **密钥安全**：仅存于本次会话内存，"
            "不写入磁盘、不进入日志、不提交代码库。"
        )
        st.caption(
            "💡 没有 key？可以先查看"
            "[使用说明](https://streamlit.io)，或点下方「用示例剧本体验」"
            "（示例数据为预生成结果）"
        )

    return {
        "provider": provider,
        "base_url": base_url,
        "model": model,
        "api_key": api_key,
        "retries": int(retries),
        "qa_sample": int(qa_sample),
    }


# ---------------------------------------------------------------- 输入区


def render_input() -> tuple[str, str]:
    """返回 (剧本正文, 标题)。"""
    st.subheader("① 输入剧本")

    tab_paste, tab_upload, tab_demo = st.tabs(
        ["📝 粘贴文本", "📁 上传文件", "🎬 示例剧本"]
    )

    text, title = "", "未命名剧本"

    with tab_paste:
        pasted = st.text_area(
            "把剧本粘贴到这里",
            height=280,
            placeholder=(
                "1. 内景 咖啡馆 日\n\n"
                "午后的阳光斜斜地穿过落地窗。\n\n"
                "林小满：你等了很久吗？\n\n"
                "陈默：还好。"
            ),
            label_visibility="collapsed",
        )
        if pasted.strip():
            text = pasted
            title = st.text_input("剧本标题", value="未命名剧本")

    with tab_upload:
        up = st.file_uploader(
            "支持 .txt / .md", type=["txt", "md"], label_visibility="collapsed"
        )
        if up is not None:
            raw = up.read()
            for enc in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
                try:
                    text = raw.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue
            if not text:
                st.error("无法识别文件编码，请转成 UTF-8 后重试")
            title = Path(up.name).stem
            st.success(f"已读取 {up.name}（{len(text)} 字）")

    with tab_demo:
        demo_path = ROOT / "samples" / "demo.txt"
        if demo_path.exists():
            demo_text = demo_path.read_text(encoding="utf-8")
            st.text_area(
                "示例剧本内容", demo_text[:800] + "\n...",
                height=200, disabled=True,
            )
            if st.button("使用这个示例", type="secondary"):
                st.session_state["use_demo"] = True
            if st.session_state.get("use_demo"):
                text, title = demo_text, "示例剧本"
        else:
            st.info("示例文件不存在")

    return text, title


# ---------------------------------------------------------------- 质控面板


def render_quality(sp, source_text: str) -> None:
    """质控指标 + 漏抽率校验。"""
    st.subheader("③ 质控结果")

    report = qa_check.compare(source_text, dump_model(sp))
    stats = (sp.meta or {}).get("stats") or {}
    usage = (sp.meta or {}).get("usage") or {}

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("场次", len(sp.scenes))
    c2.metric("人物", len(sp.characters))
    c3.metric("台词", sp.dialogue_count)
    c4.metric(
        "漏抽率",
        f"{report['missing_rate']}%",
        delta="达标" if report["missing_rate"] <= 5 else "超标",
        delta_color="normal" if report["missing_rate"] <= 5 else "inverse",
    )
    c5.metric(
        "编造内容",
        report["extra_count"],
        delta="无" if report["extra_count"] == 0 else "需核查",
        delta_color="normal" if report["extra_count"] == 0 else "inverse",
    )

    # 质控结论
    if report["missing_rate"] <= 5 and report["extra_count"] == 0:
        st.success(
            f"✅ 质控通过：漏抽率 {report['missing_rate']}%（阈值 5%），"
            f"未发现编造内容。"
        )
    else:
        st.warning(
            f"⚠️ 质控需复核：漏抽率 {report['missing_rate']}%，"
            f"多抽 {report['extra_count']} 条。"
            f"编造内容必须人工核查，不要直接交付下游。"
        )

    # 问题明细
    col_a, col_b = st.columns(2)

    with col_a:
        if report["missing_samples"]:
            st.markdown("**漏抽的台词**")
            for s in report["missing_samples"]:
                st.markdown(f"- ❌ `{s}`")
        if report["missing_characters"]:
            st.markdown("**漏抽的人物**")
            st.markdown(f"❌ {'、'.join(report['missing_characters'])}")

    with col_b:
        if report["extra_samples"]:
            st.markdown("**多出的台词（原文不存在，疑似编造）**")
            for s in report["extra_samples"]:
                st.markdown(f"- 🚨 `{s}`")
        if report["extra_characters"]:
            st.markdown("**多出的人物（疑似编造）**")
            st.markdown(f"🚨 {'、'.join(report['extra_characters'])}")

    # 运行统计
    with st.expander("📊 运行统计与重试原因"):
        s1, s2, s3 = st.columns(3)
        s1.metric("成功场次", stats.get("scenes_ok", 0))
        s2.metric("失败场次", stats.get("scenes_failed", 0))
        s3.metric("重试次数", stats.get("scenes_retried", 0))

        retry_reasons = stats.get("retry_reasons") or {}
        if retry_reasons:
            reason_cn = {
                "json_parse": "JSON 解析失败",
                "fabrication": "检测到编造内容",
                "missing_field": "缺少必填字段",
                "schema_invalid": "数据结构不符",
                "other": "其他错误",
            }
            st.markdown("**重试原因分布**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"原因": reason_cn.get(k, k), "次数": v}
                        for k, v in retry_reasons.items()
                    ]
                ),
                hide_index=True,
                use_container_width=True,
            )

        conflicts = stats.get("character_conflicts") or []
        if conflicts:
            st.markdown("**疑似人物重复（需人工判断）**")
            st.markdown("、".join(f"`{c}`" for c in conflicts))

        st.markdown("**模型调用**")
        st.markdown(
            f"调用 {usage.get('calls', 0)} 次，"
            f"消耗 {usage.get('total_tokens', 0):,} tokens，"
            f"失败 {usage.get('failures', 0)} 次"
        )


# ---------------------------------------------------------------- 结果展示


def render_results(sp, source_text: str) -> None:
    """结构化结果 + 原文对照。"""
    st.subheader("④ 结构化结果")

    tab_scenes, tab_dialogue, tab_shots, tab_compare, tab_json = st.tabs(
        ["🎬 场次", "💬 台词", "📷 镜头提示", "🔍 原文对照", "{} JSON"]
    )

    with tab_scenes:
        df = scenes_frame(sp)
        if df.empty:
            st.info("没有场次数据")
        else:
            st.dataframe(df, hide_index=True, use_container_width=True)

    with tab_dialogue:
        df = dialogue_frame(sp)
        if df.empty:
            st.info("没有台词数据")
        else:
            # 情绪筛选，方便复核人抽查
            emotions = sorted(df["情绪"].unique().tolist())
            picked = st.multiselect("按情绪筛选", emotions, default=emotions)
            view = df[df["情绪"].isin(picked)] if picked else df
            st.dataframe(view, hide_index=True, use_container_width=True)

    with tab_shots:
        df = shots_frame(sp)
        if df.empty:
            st.info(
                "本次未生成镜头提示。可能原因：模型判断本场无需提示，"
                "或抽取时该字段留空（宁缺勿编）。"
            )
        else:
            st.dataframe(df, hide_index=True, use_container_width=True)

    with tab_compare:
        # 左右对照：左边原文，右边结构化摘要
        left, right = st.columns(2)
        with left:
            st.markdown("**原剧本**")
            st.text_area(
                "source", source_text, height=520, disabled=True,
                label_visibility="collapsed",
            )
        with right:
            st.markdown("**结构化结果**")
            for s in sp.scenes:
                with st.container(border=True):
                    st.markdown(f"**{s.scene_id}** `{s.heading}`")
                    meta_bits = [s.location, s.time_of_day, s.emotion.value]
                    st.caption(" · ".join(str(b) for b in meta_bits if b))
                    if s.characters:
                        st.caption("人物：" + "、".join(s.characters))
                    if s.dialogue:
                        st.caption(f"台词 {len(s.dialogue)} 句")
                        for ln in s.dialogue[:3]:
                            st.markdown(
                                f"> **{ln.speaker}**：{ln.text} "
                                f"<span style='color:#888'>"
                                f"[{ln.emotion.value}·{ln.intensity}]</span>",
                                unsafe_allow_html=True,
                            )
                        if len(s.dialogue) > 3:
                            st.caption(f"… 另 {len(s.dialogue) - 3} 句")

    with tab_json:
        st.json(dump_model(sp))


# ---------------------------------------------------------------- 下载


def render_download(sp) -> None:
    """导出区。人工复核闸门在这里落地——必须主动勾选才能下载。"""
    st.subheader("⑤ 导出")

    st.info(
        "导出前请确认：漏抽率与编造内容已核查。"
        "结构化结果进入下游生产前，必须有人明确认可。"
    )

    confirmed = st.checkbox(
        "我已复核上述结果，确认导出", value=False
    )

    if not confirmed:
        st.caption("勾选后可下载文件")
        return

    c1, c2, c3 = st.columns(3)

    # Excel：多 sheet
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        scenes_frame(sp).to_excel(writer, sheet_name="场次表", index=False)
        dialogue_frame(sp).to_excel(writer, sheet_name="台词表", index=False)
        shots = shots_frame(sp)
        if not shots.empty:
            shots.to_excel(writer, sheet_name="镜头提示", index=False)
    buf.seek(0)
    c1.download_button(
        "⬇️ 下载 Excel",
        buf.getvalue(),
        file_name=f"{sp.title}_结构化.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    # JSON
    c2.download_button(
        "⬇️ 下载 JSON",
        json.dumps(dump_model(sp), ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"{sp.title}_结构化.json",
        mime="application/json",
        use_container_width=True,
    )

    # 台词表 CSV（给配音直接用）
    csv = dialogue_frame(sp).to_csv(index=False).encode("utf-8-sig")
    c3.download_button(
        "⬇️ 下载台词 CSV",
        csv,
        file_name=f"{sp.title}_台词.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ---------------------------------------------------------------- 主流程


def main() -> None:
    st.title("🎬 剧本结构化解析")
    st.caption(
        "自由格式剧本 → 场景 / 人物 / 台词 / 情绪 / 镜头提示 的结构化数据"
    )

    cfg = sidebar_config()
    text, title = render_input()

    st.divider()

    # 解析按钮
    st.subheader("② 解析")
    col_btn, col_hint = st.columns([1, 3])

    with col_btn:
        run = st.button("🚀 开始解析", type="primary", use_container_width=True)

    with col_hint:
        if not cfg["api_key"]:
            st.warning(
                "未填写 API Key。离线规则模式无法判定情绪与镜头提示，"
                "结果会大量出现「未判定」。建议填写 key 后使用。"
            )
        elif not text.strip():
            st.info("请先在上方输入剧本")

    if run:
        if not text.strip():
            st.error("请先输入剧本内容")
            st.stop()

        # 有 key 走真实模型，没 key 退回离线（并明确告知局限）
        offline = not bool(cfg["api_key"])
        if offline:
            st.warning(
                "⚠️ 正在使用**离线规则模式**：只能切分场次和台词，"
                "情绪与镜头提示将全部为「未判定」。"
            )
            client = LLMClient(offline=True)
        else:
            client = LLMClient(
                api_key=cfg["api_key"],
                base_url=cfg["base_url"],
                model=cfg["model"],
                offline=False,
            )

        agent = ScreenplayAgent(
            client=client,
            max_retries=cfg["retries"],
            qa_sample=cfg["qa_sample"],
        )

        with st.spinner("解析中…（逐场调用模型，可能需要 1-3 分钟）"):
            try:
                sp = agent.parse(text, title=title or "未命名剧本")
            except QualityGateError as exc:
                st.error(f"质控未通过：{exc}")
                st.stop()
            except Exception as exc:  # noqa: BLE001
                st.error(f"解析失败：{type(exc).__name__}: {exc}")
                with st.expander("查看详细错误"):
                    st.exception(exc)
                st.stop()

        st.session_state["result"] = sp
        st.session_state["source"] = text
        st.success(f"解析完成：{len(sp.scenes)} 场 / {sp.dialogue_count} 句台词")

    # 结果展示
    sp = st.session_state.get("result")
    if sp is not None:
        st.divider()
        render_quality(sp, st.session_state.get("source", ""))
        st.divider()
        render_results(sp, st.session_state.get("source", ""))
        st.divider()
        render_download(sp)
    else:
        st.divider()
        with st.expander("📖 使用说明", expanded=True):
            st.markdown("""
**三步上手**

1. 在左侧填入 API Key（DeepSeek / 通义 / Kimi 等均可，走 OpenAI 兼容接口）
2. 粘贴或上传剧本，点「开始解析」
3. 查看质控结果 → 复核 → 勾选确认 → 下载 Excel

**剧本格式要求**

不需要严格格式，满足其一即可被正确切分：

```
1. 内景 咖啡馆 日          ← 有场标最理想
林小满：你等了很久吗？       ← 台词用「人名：内容」
陈默：（低声）还好。        ← 括号内为表演提示
```

**它在做什么**

- **分场**：识别场景标题；原文没有场标时由模型判断时空切换
- **抽取**：每场单独调用模型，只喂这一场的原文（避免长剧本超上下文）
- **校验**：台词与人物必须在原文中可查，编造会触发重试
- **复核**：随机抽几场检查漏抽，算出漏抽率

**你会看到什么数字**

| 指标 | 含义 | 达标线 |
|---|---|---|
| 漏抽率 | 原文有但没抽到的台词占比 | ≤ 5% |
| 编造内容 | 抽出的原文里不存在的台词/人物 | 必须为 0 |
| 重试次数 | 校验未过触发的重试 | 越少越好 |

**关于准确性**

结构化抽取不可能 100% 准确。这个工具的设计目标是**让错误可见**，
而不是假装没有错误——所以它会把漏抽率、编造数、重试原因全部摆出来。
""")

    # 页脚
    st.divider()
    st.caption(
        "⚠️ 本工具输出为辅助生产的结构化数据，不替代人工审稿。"
        "编造内容必须人工核查后方可使用。"
    )


if __name__ == "__main__":
    main()
