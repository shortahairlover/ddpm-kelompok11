"""Konversi laporan Markdown -> HTML -> PDF (T7.6) memakai Microsoft Edge headless.

Jalankan dari root repo:
    python -m scripts.build_report_pdf
    python -m scripts.build_report_pdf --md reports/laporan_akhir.md --pdf reports/laporan.pdf

Butuh: paket `markdown` (sudah terpasang sebagai dependensi tensorboard) dan Microsoft Edge.
Gambar dirujuk relatif terhadap folder Markdown sehingga ikut tercetak.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
EDGE_PATHS = [
    Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
    Path("C:/Program Files/Microsoft/Edge/Application/msedge.exe"),
]
CSS = """
@page { size: A4; margin: 16mm 14mm; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 10.5pt; line-height: 1.45;
       color: #1a1a1a; }
h1 { font-size: 18pt; border-bottom: 2px solid #333; padding-bottom: 4px; }
h2 { font-size: 14pt; margin-top: 18px; border-bottom: 1px solid #ccc; padding-bottom: 2px; }
h3 { font-size: 12pt; }
table { border-collapse: collapse; margin: 8px 0; font-size: 9.5pt; width: 100%; }
th, td { border: 1px solid #bbb; padding: 3px 6px; vertical-align: top; }
th { background: #f0f0f0; }
code { font-family: Consolas, monospace; font-size: 9pt; background: #f4f4f4; padding: 0 2px; }
pre { background: #f6f6f6; border: 1px solid #ddd; padding: 6px; font-size: 8pt;
      white-space: pre; overflow: hidden; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
img { max-width: 100%; max-height: 120mm; display: block; margin: 6px auto;
      page-break-inside: avoid; }
"""


def find_edge() -> Path:
    for path in EDGE_PATHS:
        if path.exists():
            return path
    found = shutil.which("msedge")
    if found:
        return Path(found)
    raise FileNotFoundError("Microsoft Edge tidak ditemukan; buka HTML lalu cetak manual ke PDF")


def markdown_to_html(md_path: Path) -> str:
    """Markdown -> dokumen HTML lengkap; <base> agar path gambar relatif tetap benar."""
    body = markdown.markdown(md_path.read_text(encoding="utf8"),
                             extensions=["tables", "fenced_code", "sane_lists"])
    base = md_path.parent.resolve().as_uri() + "/"
    return (f"<!doctype html><html lang='id'><head><meta charset='utf-8'><base href='{base}'>"
            f"<title>{md_path.stem}</title><style>{CSS}</style></head><body>{body}</body></html>")


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Laporan Markdown ke PDF via Edge headless")
    parser.add_argument("--md", type=Path, default=ROOT / "reports" / "laporan_akhir.md")
    parser.add_argument("--pdf", type=Path, default=ROOT / "reports" / "laporan_akhir.pdf")
    args = parser.parse_args(argv)
    html = markdown_to_html(args.md)
    with tempfile.TemporaryDirectory() as tmp:
        html_path = Path(tmp) / "laporan.html"
        html_path.write_text(html, encoding="utf8")
        subprocess.run(
            [str(find_edge()), "--headless", "--disable-gpu", "--no-pdf-header-footer",
             f"--user-data-dir={Path(tmp) / 'edge'}",
             f"--print-to-pdf={args.pdf.resolve()}", html_path.as_uri()],
            check=True, timeout=180, capture_output=True,
        )
    if not args.pdf.exists() or args.pdf.stat().st_size == 0:
        raise RuntimeError("PDF gagal dibuat")
    print(f"PDF: {args.pdf} ({args.pdf.stat().st_size / 1024:.0f} KB)")
    return args.pdf


if __name__ == "__main__":
    main()
