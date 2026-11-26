"""Render README.md into a typeset PDF, so the deliverable matches the report.

Markdown -> HTML (with a print stylesheet) -> Word -> PDF.  Word is used as the
PDF engine because the report is already produced by Word, which keeps the two
submitted documents visually consistent.

Usage
    python src/export_readme_pdf.py          # writes outputs/_readme.html
then run the Word export step (see README / build command below) which converts
that HTML into report/Code_Documentation.pdf.

The HTML alone is a complete, self-contained document: opening it in any browser
and printing to PDF gives the same result.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import markdown

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OUT_DIR, PROJECT_ROOT

README = PROJECT_ROOT / "README.md"
HTML_OUT = OUT_DIR / "_readme.html"

CSS = """
@page { size: A4; margin: 2.0cm 2.0cm 2.0cm 2.0cm; }
body { font-family: "Times New Roman", serif; font-size: 10.5pt;
       line-height: 1.45; color: #1a1a1a; margin: 0; }
h1 { font-family: "Times New Roman", serif; font-size: 17pt; color: #14324F;
     margin: 0 0 4pt 0; padding-bottom: 5pt;
     border-bottom: 1.2pt solid #28527A; }
h2 { font-size: 12.5pt; color: #28527A; margin: 15pt 0 4pt 0;
     padding-bottom: 2pt; border-bottom: 0.5pt solid #C9D6E4; }
h3 { font-size: 11pt; color: #28527A; margin: 11pt 0 3pt 0; }
p  { margin: 0 0 6pt 0; text-align: justify; }
ul, ol { margin: 0 0 6pt 0; padding-left: 16pt; }
li { margin: 0 0 2.5pt 0; }
code { font-family: "Consolas", monospace; font-size: 9pt;
       background: #F2F5F8; padding: 0 1.5pt; }
pre { font-family: "Consolas", monospace; font-size: 8.8pt; line-height: 1.3;
      background: #F6F8FA; border: 0.5pt solid #DCE3EA; border-left: 2.5pt solid #28527A;
      padding: 6pt 7pt; margin: 5pt 0 8pt 0; white-space: pre-wrap; }
pre code { background: transparent; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 5pt 0 9pt 0;
        font-family: "Times New Roman", serif; font-size: 9.5pt; }
th { background: #EAF1F8; border: 0.5pt solid #9FB3C8; padding: 3.5pt 4pt;
     text-align: left; font-weight: bold; color: #14324F; }
td { border: 0.5pt solid #B9C7D6; padding: 3.5pt 4pt; vertical-align: top; }
tr { page-break-inside: avoid; }
hr { border: 0; border-top: 0.5pt solid #C9D6E4; margin: 10pt 0; }
blockquote { margin: 5pt 0 7pt 0; padding: 2pt 0 2pt 8pt;
             border-left: 2.5pt solid #C9D6E4; color: #3a3a3a; }
strong { color: #10263c; }
.meta { text-align: left; color: #333333; }
"""

TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>{css}</style></head>
<body>
{body}
</body></html>
"""


def main() -> None:
    src = README.read_text(encoding="utf-8")

    # python-markdown needs four spaces before a nested list marker; GitHub also
    # accepts three, and the README uses three, so normalise before parsing.
    src = re.sub(r"(?m)^ {3}([*+-] |\d+\. )", r"    \1", src)

    # the pointer to the Chinese mirror is useful in the repository only; the
    # submitted code documentation is English throughout, so drop that line here
    src = "\n".join(l for l in src.splitlines() if "README_zh.md" not in l)

    html_body = markdown.markdown(
        src,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
        output_format="html5",
    )
    # the author / repository lines are a short paragraph, so justifying it only
    # produces wide gaps between words
    html_body = html_body.replace("<p><strong>Author:</strong>",
                                  '<p class="meta"><strong>Author:</strong>')
    html = TEMPLATE.format(title="Code Documentation", css=CSS, body=html_body)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HTML_OUT.write_text(html, encoding="utf-8")
    print(f"saved -> {HTML_OUT}")
    print(f"size  -> {HTML_OUT.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
