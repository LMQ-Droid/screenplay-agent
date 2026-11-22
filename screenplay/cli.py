# -*- coding: utf-8 -*-
"""命令行入口。

用法：
    # 跑单个剧本
    python -m screenplay.cli run samples/demo.txt

    # 批量跑一个目录
    python -m screenplay.cli batch samples/

    # 离线模式（不调模型，用规则引擎，用于测试）
    python -m screenplay.cli run samples/demo.txt --offline

    # 指定接口（DeepSeek / 通义 / Kimi 等 OpenAI 兼容服务）
    set LLM_API_KEY=sk-xxx
    set LLM_BASE_URL=https://api.deepseek.com/v1
    python -m screenplay.cli run samples/demo.txt
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .agent import QualityGateError, ScreenplayAgent
from .export import ReviewGate, export_all, write_run_log
from .llm import LLMClient

logger = logging.getLogger("screenplay")


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s | %(message)s",
        stream=sys.stdout,
    )
    for noisy in ("urllib3", "requests"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def read_script(path: Path) -> str:
    """读剧本。兼容 UTF-8 / GBK——中文剧本这两种编码最常见。"""
    for enc in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法识别文件编码: {path}")


def process_one(
    path: Path,
    agent: ScreenplayAgent,
    out_dir: Path,
    auto_approve: bool,
    require_review: bool,
) -> dict:
    """处理单个剧本文件。"""
    logger.info("处理: %s", path.name)
    text = read_script(path)
    logger.info("文件长度: %d 字", len(text))

    sp = agent.parse(text, title=path.stem)

    gate = ReviewGate(require_review=require_review)
    print(gate.summary(sp))

    if not gate.confirm(auto_approve=auto_approve):
        logger.warning("复核未通过，已中止导出: %s", path.name)
        return {"file": path.name, "status": "rejected"}

    paths = export_all(sp, out_dir)
    log_path = write_run_log(sp, out_dir, extra={"source": str(path)})

    logger.info("导出完成:")
    for kind, p in paths.items():
        logger.info("  %-5s %s", kind, p)
    logger.info("  日志  %s", log_path)

    return {
        "file": path.name,
        "status": "ok",
        "scenes": len(sp.scenes),
        "characters": len(sp.characters),
        "dialogue": sp.dialogue_count,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="AI 短剧剧本结构化解析 Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--offline", action="store_true",
                       help="离线规则模式，不调用模型")
        p.add_argument("--out", default="out", help="输出目录")
        p.add_argument("--yes", action="store_true", help="跳过人工复核确认")
        p.add_argument("--no-review", action="store_true", help="关闭复核闸门")
        p.add_argument("--qa-sample", type=int, default=3,
                       help="抽样复核场次数，0 关闭")
        p.add_argument("--retries", type=int, default=2, help="单场重试次数")
        p.add_argument("-v", "--verbose", action="store_true")

    p_run = sub.add_parser("run", help="处理单个剧本文件")
    p_run.add_argument("file", help="剧本文件路径")
    add_common(p_run)

    p_batch = sub.add_parser("batch", help="批量处理目录下的 txt 文件")
    p_batch.add_argument("dir", help="剧本目录")
    add_common(p_batch)

    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    out_dir = Path(args.out)

    client = LLMClient(offline=args.offline)
    agent = ScreenplayAgent(
        client=client,
        max_retries=args.retries,
        qa_sample=args.qa_sample,
    )

    try:
        if args.cmd == "run":
            src = Path(args.file)
            if not src.exists():
                logger.error("文件不存在: %s", src)
                return 1
            result = process_one(
                src, agent, out_dir, args.yes, not args.no_review
            )
            return 0 if result["status"] == "ok" else 2

        # batch
        src_dir = Path(args.dir)
        if not src_dir.is_dir():
            logger.error("目录不存在: %s", src_dir)
            return 1

        files = sorted(
            p for p in src_dir.iterdir()
            if p.suffix.lower() in {".txt", ".md"} and p.is_file()
        )
        if not files:
            logger.error("目录下没有 .txt/.md 文件: %s", src_dir)
            return 1

        logger.info("批量处理 %d 个文件", len(files))
        results: list[dict] = []
        for f in files:
            try:
                results.append(
                    process_one(
                        f, agent, out_dir, args.yes, not args.no_review
                    )
                )
            except Exception as exc:  # noqa: BLE001
                # 单文件失败不中断整批——批量任务必须能跑完
                logger.error("%s 处理失败: %s", f.name, exc)
                results.append({"file": f.name, "status": "error", "error": str(exc)})

        ok = sum(1 for r in results if r["status"] == "ok")
        print("\n" + "=" * 56)
        print(f"批量完成: {ok}/{len(results)} 成功")
        for r in results:
            flag = "OK " if r["status"] == "ok" else "ERR"
            detail = (
                f"{r.get('scenes')}场 {r.get('dialogue')}句"
                if r["status"] == "ok"
                else r.get("error", r["status"])
            )
            print(f"  [{flag}] {r['file']:30s} {detail}")
        print("=" * 56)

        logger.info("token 用量: %s", client.usage.as_dict())
        return 0 if ok == len(results) else 2

    except QualityGateError as exc:
        logger.error("质控未通过: %s", exc)
        return 3
    except KeyboardInterrupt:
        logger.warning("已中断")
        return 130
    except Exception as exc:  # noqa: BLE001
        logger.exception("运行失败: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
