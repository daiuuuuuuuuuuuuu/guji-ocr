"""CLI entry point for agent_c — classical Chinese OCR text correction via LLM."""

import argparse
import logging
import shutil
import sys
from pathlib import Path

from . import __version__
from .corrector import correct_page, create_llm_client

logger = logging.getLogger("agent_c")


def setup_logging(debug: bool = False):
    level = logging.DEBUG if debug else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)


def cmd_correct(args):
    """Run LLM text correction on OCR output."""
    setup_logging(args.debug)

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    if not in_dir.is_dir():
        print(f"Error: input directory not found: {in_dir}")
        return 1

    # Find all per-book directories
    books = [d for d in in_dir.iterdir() if d.is_dir()]
    if not books:
        print(f"No book directories found in {in_dir}")
        return 1

    logger.info("agent_c v%s", __version__)
    logger.info("Input:  %s (%d book(s))", in_dir, len(books))
    logger.info("Output: %s", out_dir.resolve())

    # Create LLM client (CLI --model/--base-url override config & env)
    client, model = create_llm_client(
        section="agent_c",
        model=args.model,
        base_url=args.base_url,
    )

    total_pages = 0
    total_corrected = 0

    for book_dir in sorted(books):
        txt_dir = book_dir / "txt"
        if not txt_dir.is_dir():
            logger.warning("Skipping %s (no txt/ subdirectory)", book_dir.name)
            continue

        txt_files = sorted(txt_dir.glob("page_*.txt"))
        if not txt_files:
            logger.warning("No page_*.txt files found in %s", book_dir)
            continue

        logger.info("Book: %s (%d pages)", book_dir.name, len(txt_files))

        # Prepare output directories
        book_out = out_dir / book_dir.name
        txt_out = book_out / "txt"
        json_out = book_out / "json"
        txt_out.mkdir(parents=True, exist_ok=True)
        json_out.mkdir(parents=True, exist_ok=True)

        book_total = 0
        book_corrected = 0

        for txt_file in txt_files:
            page_name = txt_file.stem  # e.g. "page_001"
            page_index = page_name.split("_")[1]  # e.g. "001"

            # Read original text
            original = txt_file.read_text(encoding="utf-8")

            # LLM correction
            logger.debug("Correcting %s/%s ...", book_dir.name, page_name)
            corrected = correct_page(
                original,
                client,
                model=model,
            )

            # Write corrected txt
            dst_txt = txt_out / f"{page_name}.txt"
            dst_txt.write_text(corrected, encoding="utf-8")

            book_total += 1
            if corrected != original:
                book_corrected += 1

            # Copy json
            src_json = book_dir / "json" / f"{page_name}.json"
            dst_json = json_out / f"{page_name}.json"
            if src_json.exists():
                shutil.copy2(src_json, dst_json)

        logger.info(
            "  → %d/%d pages corrected",
            book_corrected, book_total,
        )
        total_pages += book_total
        total_corrected += book_corrected

    # Summary
    print(f"\n{'='*50}")
    print(f"  错别字纠正完成")
    print(f"{'='*50}")
    print(f"  处理书籍: {len(books)}")
    print(f"  处理页数: {total_pages}")
    print(f"  纠正页数: {total_corrected}")
    print(f"  输出目录: {out_dir.resolve()}")
    print(f"{'='*50}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="agent-correct",
        description="古籍错别字纠正 Agent: LLM 校勘 OCR 输出文本",
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # correct
    correct_parser = subparsers.add_parser("correct", help="LLM 纠正 OCR 错别字")
    correct_parser.add_argument(
        "--input", "-i",
        required=True,
        help="输入目录（如 out_v/ 或 out/）",
    )
    correct_parser.add_argument(
        "--output", "-o",
        default="out_c",
        help="纠正后输出目录（默认: out_c）",
    )
    correct_parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="启用 debug 日志",
    )
    correct_parser.add_argument(
        "--model",
        default=None,
        help="覆盖 LLM 模型名（优先级高于配置文件和环境变量）",
    )
    correct_parser.add_argument(
        "--base-url",
        default=None,
        help="覆盖 API 地址（优先级高于配置文件和环境变量）",
    )

    if len(sys.argv) == 1:
        parser.print_help()
        return 0

    args = parser.parse_args()

    if args.command == "correct":
        return cmd_correct(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
