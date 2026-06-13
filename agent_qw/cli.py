"""CLI entry point for the 古籍识别 Agent."""

import argparse
import logging
import sys
from pathlib import Path

from .config import load_config
from .pipeline import Pipeline


def setup_logging(debug: bool = False):
    """Configure logging with appropriate level and format."""
    level = logging.DEBUG if debug else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)


def cmd_run(args):
    """Execute the recognition pipeline."""
    config_path = args.config or None
    cfg = load_config(config_path)

    # CLI overrides
    if args.output:
        cfg.output_dir = args.output
    if args.debug:
        cfg.debug = True
    if args.workers is not None:
        cfg.max_workers = args.workers

    setup_logging(cfg.debug)
    logger = logging.getLogger("agent_qw")

    logger.info("古籍识别 Agent v0.1.0")
    logger.info("Input: %s", args.input)
    logger.info("Output: %s", cfg.output_dir)
    logger.info("Workers: %d", cfg.max_workers)

    pipeline = Pipeline(cfg)
    pages = pipeline.run(
        input_path=args.input,
        output_dir=cfg.output_dir,
    )

    # Print summary
    print(f"\n{'='*50}")
    print(f"  古籍识别完成")
    print(f"{'='*50}")
    print(f"  处理页数: {len(pages)}")
    total_chars = sum(len(p.full_text) for p in pages)
    print(f"  总字符数: {total_chars}")
    print(f"  输出目录: {Path(cfg.output_dir).resolve()}")
    print(f"    {{文件名}}/txt/page_001.txt        — 纯文本")
    print(f"    {{文件名}}/json/page_001.json       — 结构化结果")
    print(f"    {{文件名}}/preprocess/page_001_binary.png — 预处理图片")
    if cfg.debug:
        print(f"    {{文件名}}/pages/                 — 原始页面图片")
    print(f"{'='*50}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="agent-qw",
        description="古籍识别 Agent: 预处理 + 多模态 LLM OCR + 导出",
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # run
    run_parser = subparsers.add_parser("run", help="运行识别流程")
    run_parser.add_argument(
        "--input", "-i",
        required=True,
        help="输入 PDF/图片 文件或目录路径",
    )
    run_parser.add_argument(
        "--output", "-o",
        default=None,
        help="输出目录（默认: out）",
    )
    run_parser.add_argument(
        "--config", "-c",
        default=None,
        help="配置文件路径（YAML）",
    )
    run_parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="启用 debug 模式（保存中间产物）",
    )
    run_parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        help="并行线程数（默认 2）",
    )

    if len(sys.argv) == 1:
        parser.print_help()
        return 0

    args = parser.parse_args()

    if args.command == "run":
        return cmd_run(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
