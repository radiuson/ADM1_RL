#!/usr/bin/env python3
"""Rebuild the web copy's payload from paper.tex.

The manuscript artifact is a reader: index.html ships the interface and
payload.js carries the text.  This regenerates the payload so the published
copy matches the file on disk, which has to happen after every edit to
paper.tex.

    python3 scripts/build_artifact_payload.py <outdir>

Writes paper.json and payload.js into <outdir>.
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
import time

PAPER = '/home/ihpc/code/biogas/ADM1/papers/mypaper/paper.tex'
BIB = '/home/ihpc/code/biogas/ADM1/papers/mypaper/references.bib'

SEC_RE = re.compile(r'^\\(section|subsection)\*?\{(.+?)\}', re.M)


def strip_label(title: str) -> str:
    """Drop a trailing \\label{...} and tidy the heading text."""
    title = re.sub(r'\\label\{[^}]*\}', '', title)
    return title.replace('\\', '').strip()


def main() -> None:
    outdir = sys.argv[1]
    os.makedirs(outdir, exist_ok=True)

    src = io.open(PAPER, encoding='utf-8').read()

    # Title: the \title{...} argument, which may run over several lines.
    m = re.search(r'\\title\{(.+?)\n*\}', src, re.S)
    title = strip_label(m.group(1)) if m else 'Manuscript'
    title = re.sub(r'\s+', ' ', title)

    # Abstract.
    m = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', src, re.S)
    abstract = m.group(1).strip() if m else ''

    # Body sections: everything between one heading and the next.  The
    # appendix and the back matter are headings too, so the reader shows the
    # whole document rather than stopping at the conclusion.
    heads = list(SEC_RE.finditer(src))
    sections = []
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(src)
        body = src[h.end():end].strip()
        sections.append({
            'level': 1 if h.group(1) == 'section' else 2,
            'title': strip_label(h.group(2)),
            'tex': body,
        })

    bib = ''
    if os.path.exists(BIB):
        bib = io.open(BIB, encoding='utf-8').read()

    payload = {
        'title': title,
        'abstract': abstract,
        'sections': sections,
        'source': src,
        'lines': src.count('\n') + 1,
        'nsec': sum(1 for s in sections if s['level'] == 1),
        'builtAt': int(time.time() * 1000),
        'bib': bib,
        'nbib': bib.count('\n@') + (1 if bib.startswith('@') else 0),
    }

    jpath = os.path.join(outdir, 'paper.json')
    with io.open(jpath, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, ensure_ascii=False)

    blob = json.dumps(payload, ensure_ascii=False)
    with io.open(os.path.join(outdir, 'payload.js'), 'w', encoding='utf-8') as fh:
        fh.write('window.PAPER=' + blob + ';\n')

    print(f'{payload["nsec"]} sections ({len(sections)} headings), '
          f'{payload["lines"]} lines, {payload["nbib"]} bib entries')
    print(f'wrote {outdir}/paper.json and payload.js')


if __name__ == '__main__':
    main()
