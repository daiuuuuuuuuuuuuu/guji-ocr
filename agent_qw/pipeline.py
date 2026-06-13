"""Main pipeline orchestrator: preprocessing → OCR → export."""

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from PIL import Image

from .config import PipelineConfig
from .schemas import Page, Block, BBox, TextDirection, ReadingOrder
from .io_utils import convert_path_to_images, collect_input_files
from .export import export_txt, export_json
from .vision.preprocess import preprocess
from .postprocess.normalize import normalize_page_text

from collections import Counter

logger = logging.getLogger(__name__)

# Regex to extract LLM-detected page number marker: [[PAGE:5]]
_PAGE_MARKER = re.compile(r'\[\[PAGE:(\d+)\]\]\s*')


def _parse_page_marker(text: str):
    """Extract [[PAGE:N]] marker from LLM OCR output.

    Returns (clean_text, page_number_or_none).
    """
    m = _PAGE_MARKER.match(text)
    if m:
        return text[m.end():], int(m.group(1))
    return text, None


def _strip_headers_footers(pages: List[Page], threshold: float = 0.4) -> List[Page]:
    """Detect and strip repeated header/footer lines across all pages.

    A line is identified as a header/footer if it appears as the first or
    last non-empty line on at least ``threshold`` fraction of pages (e.g.
    0.4 = 40%).  The original text is saved in ``Page.raw_text`` before
    stripping; ``full_text`` receives the cleaned version.

    Returns the same list (mutated in place).
    """
    if len(pages) < 3:
        # Not enough pages for meaningful detection; save raw_text anyway
        for p in pages:
            p.raw_text = p.full_text
        return pages

    # Collect first and last non-empty lines from every page
    first_lines: List[str] = []
    last_lines: List[str] = []
    for p in pages:
        lines = [ln.strip() for ln in p.full_text.split("\n") if ln.strip()]
        if lines:
            first_lines.append(lines[0])
            last_lines.append(lines[-1])

    if not first_lines:
        for p in pages:
            p.raw_text = p.full_text
        return pages

    # Lines that recur on enough pages are headers/footers
    min_pages = max(2, int(len(pages) * threshold))
    first_counts = Counter(first_lines)
    last_counts = Counter(last_lines)
    headers = {ln for ln, c in first_counts.items() if c >= min_pages}
    footers = {ln for ln, c in last_counts.items() if c >= min_pages}

    if not headers and not footers:
        for p in pages:
            p.raw_text = p.full_text
        return pages

    stripped_count = 0
    for p in pages:
        p.raw_text = p.full_text  # preserve original
        lines = [ln.strip() for ln in p.full_text.split("\n") if ln.strip()]
        if not lines:
            continue
        changed = False
        # Strip from top
        while lines and lines[0] in headers:
            lines.pop(0)
            changed = True
        # Strip from bottom
        while lines and lines[-1] in footers:
            lines.pop()
            changed = True
        if changed:
            p.full_text = "\n".join(lines).strip()
            stripped_count += 1

    logger.info(
        "Header/footer stripping: %d headers, %d footers → cleaned %d/%d pages",
        len(headers), len(footers), stripped_count, len(pages),
    )
    return pages


class Pipeline:
    """古籍识别 pipeline: preprocessing → OCR → export.

    Usage:
        cfg = load_config("config.yaml")
        pipeline = Pipeline(cfg)
        pipeline.run(input_path="samples/", output_dir="out/")
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._ocr_engine = None
        self._llm = None

    @property
    def ocr_engine(self):
        if self._ocr_engine is None:
            llm = self.llm
            if llm is None or not llm.is_available:
                raise RuntimeError(
                    "OCR requires a configured LLM "
                    "(set GUJI_LLM_API_KEY or configure llm in config.yaml)"
                )
            from .ocr.llm_vision import LLMVisionOCR
            self._ocr_engine = LLMVisionOCR(
                llm_client=llm,
                model=self.config.ocr.model,
                temperature=self.config.ocr.temperature,
                max_tokens=self.config.ocr.max_tokens,
            )
            logger.info(
                "OCR engine: llm_vision (model=%s)",
                self.config.ocr.model,
            )
        return self._ocr_engine

    @property
    def llm(self):
        if self._llm is None:
            from .llm.openai_compat import OpenAICompatLLM
            self._llm = OpenAICompatLLM(
                base_url=self.config.llm.base_url,
                api_key=self.config.llm.api_key,
                model=self.config.llm.model,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
            )
            if not self._llm.is_available:
                logger.warning("LLM not configured (no API key)")
        return self._llm

    def run(
        self,
        input_path: str,
        output_dir: Optional[str] = None,
    ) -> List[Page]:
        """Run the full pipeline on an input file or directory.

        For each input file, creates a subdirectory under output_dir
        named after the file stem, with categorized per-page output:
            out/文件名/txt/page_001.txt
            out/文件名/json/page_001.json
            out/文件名/preprocess/page_001_binary.png
        """
        start_time = time.time()
        out_root = Path(output_dir or self.config.output_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        input_files = collect_input_files(input_path)
        if not input_files:
            raise FileNotFoundError(f"No supported files found in: {input_path}")

        logger.info("Found %d input file(s)", len(input_files))

        all_pages = []
        total_chars = 0

        for file_path in input_files:
            stem = Path(file_path).stem
            file_dir = out_root / stem
            file_dir.mkdir(parents=True, exist_ok=True)

            # Create categorized output subdirectories
            txt_dir = file_dir / "txt"
            json_dir = file_dir / "json"
            preprocess_dir = file_dir / "preprocess"
            txt_dir.mkdir(parents=True, exist_ok=True)
            json_dir.mkdir(parents=True, exist_ok=True)
            preprocess_dir.mkdir(parents=True, exist_ok=True)
            dirs = {
                "txt": txt_dir,
                "json": json_dir,
                "preprocess": preprocess_dir,
            }
            if self.config.debug:
                (file_dir / "pages").mkdir(exist_ok=True)
                dirs["pages"] = file_dir / "pages"

            logger.info("Processing: %s → %s/ (%d workers)", file_path, file_dir.name, self.config.max_workers)
            images = convert_path_to_images(file_path)
            file_pages = []

            # Force OCR engine init before threading (avoids race on lazy init)
            _ = self.ocr_engine

            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_idx = {
                    executor.submit(self._process_page, img, i, file_path, dirs): i
                    for i, img in enumerate(images)
                }
                for future in as_completed(future_to_idx):
                    page_idx = future_to_idx[future]
                    try:
                        page = future.result()
                    except Exception:
                        logger.exception("Page %d failed, skipping", page_idx + 1)
                        continue
                    file_pages.append(page)

            # Restore original page order
            file_pages.sort(key=lambda p: p.page_index)

            # Cross-page: detect and strip repeated headers/footers
            file_pages = _strip_headers_footers(file_pages)

            # Export per-page results (after stripping, so .txt is clean)
            for page in file_pages:
                num = page.detected_page_number or (page.page_index + 1)
                export_txt([page], str(txt_dir / f"page_{num:03d}.txt"))
                export_json([page], str(json_dir / f"page_{num:03d}.json"))

                if self.config.debug:
                    logger.info(
                        "  page_%03d: %d chars → txt/, json/",
                        num, len(page.full_text),
                    )

            all_pages.extend(file_pages)
            total_chars += sum(len(p.full_text) for p in file_pages)

        elapsed = time.time() - start_time
        logger.info(
            "Pipeline complete: %d pages, %d chars in %.1fs (%.1f s/page)",
            len(all_pages), total_chars, elapsed,
            elapsed / max(len(all_pages), 1),
        )

        return all_pages

    def _process_page(
        self,
        pil_image: Image.Image,
        page_index: int,
        source_path: str,
        dirs: dict,
    ) -> Page:
        """Process a single page: preprocess → OCR → normalize."""
        w, h = pil_image.size

        # Step 1: Preprocess — binarize + sharpen
        binary, _ = preprocess(
            pil_image,
            binarize=self.config.layout.binarize,
            do_sharpen=self.config.layout.sharpen,
            sharpen_amount=self.config.layout.sharpen_amount,
            do_deskew=self.config.layout.deskew,
        )

        # Always save preprocessed image
        pre_path = dirs["preprocess"] / f"page_{page_index + 1:03d}_binary.png"
        Image.fromarray(binary).save(str(pre_path))
        logger.debug("Saved preprocessed image: %s", pre_path)

        # Step 2: OCR — send preprocessed image to multimodal LLM
        preprocessed_pil = Image.fromarray(binary)
        ocr_lines = self.ocr_engine.recognize(preprocessed_pil)
        raw_text = ocr_lines[0].text if ocr_lines else ""

        # Extract LLM-detected page number from marker (e.g. [[PAGE:5]])
        clean_text, detected_page = _parse_page_marker(raw_text)
        full_text = clean_text
        if ocr_lines:
            ocr_lines[0].text = clean_text

        reading_order = ReadingOrder.RIGHT_TO_LEFT
        direction = TextDirection.VERTICAL

        blocks = [
            Block(
                id="full",
                bbox=BBox(x=0, y=0, w=w, h=h),
                text=full_text,
                lines=ocr_lines,
                direction=direction,
                order_index=0,
            )
        ]

        # Step 3: Normalization
        full_text = normalize_page_text(full_text)

        # Save debug page image
        if self.config.debug and "pages" in dirs:
            page_img_path = dirs["pages"] / f"page_{page_index:03d}.png"
            pil_image.save(str(page_img_path))
            logger.debug("Saved page image: %s", page_img_path)

        return Page(
            page_index=page_index,
            image_path=source_path,
            width=w, height=h,
            blocks=blocks,
            full_text=full_text,
            direction=direction,
            reading_order=reading_order,
            detected_page_number=detected_page,
        )
