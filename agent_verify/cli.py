"""CLI entry point for agent_verify — LLM-based page order verification and reordering."""

import argparse
import logging
import re
import shutil
import sys
from pathlib import Path

from . import __version__
from .verifier import (
    BATCH_SIZE,
    create_llm_client,
    extract_head_tail,
    verify_batch,
)

logger = logging.getLogger("agent_verify")


def setup_logging(debug: bool = False):
    level = logging.DEBUG if debug else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)


def cmd_verify(args):
    """Run LLM-based page order verification and reordering."""
    setup_logging(args.debug)

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    if not in_dir.is_dir():
        print(f"Error: input directory not found: {in_dir}")
        return 1

    # Find all per-book input directories
    books = [d for d in in_dir.iterdir() if d.is_dir()]
    if not books:
        print(f"No book directories found in {in_dir}")
        return 1

    logger.info("agent_verify v%s", __version__)
    logger.info("Input:  %s (%d book(s))", in_dir, len(books))
    logger.info("Output: %s", out_dir.resolve())

    # Create LLM client (CLI --model/--base-url override config & env)
    client, model = create_llm_client(
        section="agent_verify",
        model=args.model,
        base_url=args.base_url,
    )

    batch_size = args.batch_size

    total_pages = 0
    total_reordered = 0

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

        # Read all pages and extract head/tail
        pages_data = []  # list of (filename_stem, full_text, head, tail)
        for tf in txt_files:
            text = tf.read_text(encoding="utf-8")
            head, tail = extract_head_tail(text)
            pages_data.append((tf.stem, text, head, tail))

        # Split into batches
        batches = [
            pages_data[i : i + batch_size]
            for i in range(0, len(pages_data), batch_size)
        ]
        logger.info(
            "  %d batch(es) (batch_size=%d)", len(batches), batch_size,
        )

        # Process each batch through LLM
        reorder_map = []  # list of (source_stem, new_page_num)
        prev_tail = None   # tail text from the last page of the previous batch

        for batch_idx, batch in enumerate(batches):
            batch_pages = [(head, tail) for (_, _, head, tail) in batch]

            logger.info(
                "  Batch %d/%d: %d pages",
                batch_idx + 1, len(batches), len(batch_pages),
            )

            order = verify_batch(
                batch_pages,
                client,
                model,
                prev_tail=prev_tail,
            )

            # Map the LLM order back to source filenames and assign new page numbers
            for idx_in_batch in order:
                stem = batch[idx_in_batch][0]
                new_num = len(reorder_map) + 1  # 1-based global page number
                reorder_map.append((stem, new_num))

            # Cross-batch continuity: tail of the LAST page in the correct order
            last_idx = order[-1]
            prev_tail = batch[last_idx][3]  # tail is at index 3

        # Write reordered output
        book_out = out_dir / book_dir.name
        txt_out = book_out / "txt"
        json_out = book_out / "json"
        txt_out.mkdir(parents=True, exist_ok=True)
        json_out.mkdir(parents=True, exist_ok=True)

        for src_stem, new_num in reorder_map:
            _copy_page(book_dir, src_stem, new_num, txt_out, json_out)

        # Report changes
        original_order = {tf.stem: i for i, tf in enumerate(txt_files)}
        changes = sum(
            1 for i, (stem, _) in enumerate(reorder_map)
            if original_order.get(stem) != i
        )
        logger.info(
            "  → %d pages written (%d changed position)",
            len(reorder_map), changes,
        )
        total_pages += len(reorder_map)
        total_reordered += changes

    # Summary
    print(f"\n{'='*50}")
    print(f"  页序校验完成 (LLM)")
    print(f"{'='*50}")
    print(f"  处理书籍: {len(books)}")
    print(f"  处理页数: {total_pages}")
    print(f"  调整页数: {total_reordered}")
    print(f"  输出目录: {out_dir.resolve()}")
    print(f"{'='*50}")

    return 0


def _copy_page(
    book_dir: Path,
    src_stem: str,
    new_num: int,
    txt_out: Path,
    json_out: Path,
):
    """Copy txt and json for a single page with new numbering.

    Args:
        book_dir: Source book directory (e.g. out/SomeBook/).
        src_stem: Source filename stem (e.g. "page_001").
        new_num: New 1-based page number.
        txt_out: Destination txt directory.
        json_out: Destination json directory.
    """
    m = re.match(r"page_(\d+)", src_stem)
    page_index = m.group(1) if m else src_stem.split("_", 1)[-1]

    # Copy txt
    src_txt = book_dir / "txt" / f"{src_stem}.txt"
    dst_txt = txt_out / f"page_{new_num:03d}.txt"
    if src_txt.exists():
        shutil.copy2(src_txt, dst_txt)

    # Copy json
    src_json = book_dir / "json" / f"{src_stem}.json"
    dst_json = json_out / f"page_{new_num:03d}.json"
    if src_json.exists():
        shutil.copy2(src_json, dst_json)


def main():
    parser = argparse.ArgumentParser(
        prog="agent-verify",
        description="古籍页序校验 Agent: LLM 文字连贯性判断 + 重排序",
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # verify
    verify_parser = subparsers.add_parser("verify", help="LLM 校验并重排页序")
    verify_parser.add_argument(
        "--input", "-i",
        required=True,
        help="输入目录（如 out/ 或 out_c/）",
    )
    verify_parser.add_argument(
        "--output", "-o",
        default="out_v",
        help="重排后输出目录（默认: out_v）",
    )
    verify_parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="启用 debug 日志",
    )
    verify_parser.add_argument(
        "--model",
        default=None,
        help="覆盖 LLM 模型名（优先级高于配置文件和环境变量）",
    )
    verify_parser.add_argument(
        "--base-url",
        default=None,
        help="覆盖 API 地址（优先级高于配置文件和环境变量）",
    )
    verify_parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"每批最大页数（默认: {BATCH_SIZE}）",
    )

    if len(sys.argv) == 1:
        parser.print_help()
        return 0

    args = parser.parse_args()

    if args.command == "verify":
        return cmd_verify(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
