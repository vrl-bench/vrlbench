#!/usr/bin/env python3
"""Render PNG previews from the SVGs maintained in docs/assets.

With no figure names, render only the light and dark system overviews. Other
figures must be selected explicitly. SVG inputs are not rewritten unless
--refresh-dark requests a colour-only export of the light overview. No paper
sources, experiment results, cropping, or plotting code are needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
DEFAULT_FIGURES = ("simulator-overview", "simulator-overview-dark")
FIGURES = (
    *DEFAULT_FIGURES,
    "memory-effects",
    "reflection-exposure",
    "vex2-scheduler",
    "memory-research",
    "memory-systems",
    "trial-and-error",
)

# The dark export changes colours only; the reversible mapping guards the
# light overview's geometry and wording. Regeneration requires --refresh-dark.
OVERVIEW_DARK_PALETTE = {
    b"#ffffff": b"#0d1117",
    b"#f4f5f8": b"#161e2a",
    b"#eaf0ff": b"#1d2a3b",
    b"#d8dee8": b"#344357",
    b"#a6b2c4": b"#50647a",
    b"#18212f": b"#e6edf3",
    b"#5c697b": b"#9fb3c8",
    b"#758398": b"#8da2b7",
    b"#3157c8": b"#58a6ff",
    b"#0b766e": b"#48c5ba",
    b"#e8f3ef": b"#122d30",
    b"#9fc3b8": b"#286665",
}
HEX_COLOUR = re.compile(rb"#[0-9a-fA-F]{6}\b")


def dark_overview(source: bytes) -> bytes:
    used = {colour.lower() for colour in HEX_COLOUR.findall(source)}
    unknown = used - OVERVIEW_DARK_PALETTE.keys()
    if unknown:
        raise ValueError(f"Overview uses unmapped colours: {sorted(unknown)}")
    dark = HEX_COLOUR.sub(lambda match: OVERVIEW_DARK_PALETTE[match[0].lower()], source)
    inverse = {value: key for key, value in OVERVIEW_DARK_PALETTE.items()}
    if len(inverse) != len(OVERVIEW_DARK_PALETTE):
        raise ValueError("Overview palette must have a reversible colour mapping")
    restored = HEX_COLOUR.sub(lambda match: inverse[match[0].lower()], dark)
    if restored != source:
        raise ValueError("Dark export changed content beyond the palette")
    ET.fromstring(dark)
    return dark


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available figures: " + ", ".join(FIGURES),
    )
    parser.add_argument(
        "figures",
        metavar="FIGURE",
        nargs="*",
        help="SVG basenames to render; defaults to the light and dark overviews",
    )
    parser.add_argument(
        "--refresh-dark",
        action="store_true",
        help="export the light overview with the dark palette and render its PNG",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ASSETS,
        help="preview destination (default: docs/assets); SVG inputs stay in docs/assets",
    )
    args = parser.parse_args(argv)
    unknown = set(args.figures) - set(FIGURES)
    if unknown:
        parser.error("unknown figure name(s): " + ", ".join(sorted(unknown)))
    args.figures = list(dict.fromkeys(args.figures or DEFAULT_FIGURES))
    if args.refresh_dark and "simulator-overview-dark" not in args.figures:
        args.figures.append("simulator-overview-dark")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    renderer = shutil.which("rsvg-convert")
    if renderer is None:
        raise SystemExit("rsvg-convert is required to render the PNG previews")

    # Validate all selected inputs before creating or overwriting any output.
    sources = {name: ASSETS / f"{name}.svg" for name in args.figures}
    dark = None
    if args.refresh_dark:
        dark = dark_overview((ASSETS / "simulator-overview.svg").read_bytes())
    for name, source in sources.items():
        content = (
            dark
            if name == "simulator-overview-dark" and dark is not None
            else source.read_bytes()
        )
        ET.fromstring(content)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if dark is not None:
        destination = output_dir / "simulator-overview-dark.svg"
        if not destination.exists() or destination.read_bytes() != dark:
            destination.write_bytes(dark)
        sources["simulator-overview-dark"] = destination
        print(f"Dark palette SVG: {destination}")

    for name, source in sources.items():
        destination = output_dir / f"{name}.png"
        subprocess.run(
            [renderer, "--output", str(destination), str(source)],
            check=True,
        )
        print(f"PNG preview: {destination}")
    print(f"Rendered {len(sources)} selected SVG(s) with rsvg-convert.")


if __name__ == "__main__":
    main()
