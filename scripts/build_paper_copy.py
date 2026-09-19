#!/usr/bin/env python3
"""Rebuild the web copy's payload from the manuscript on disk.

The published copy cannot read paper.tex itself -- it is a page on claude.ai
with no route to this filesystem -- so syncing is this script plus a republish
of the artifact. It flattens the \\input files so the copy carries a single
compilable source, splits the body into sections for the reading view, and
stamps the build time the page counts "updated Ns ago" from.

    python3 scripts/build_paper_copy.py
"""
from __future__ import annotations

import json
import os
import re
import time

PAPER = os.path.expanduser('~/code/biogas/ADM1/papers/mypaper')
OUT = ('/tmp/claude-1000/-home-ihpc-code/'
       'd8e23994-23f1-4be7-bbbc-30cb1d90de5a/scratchpad/paper_copy')


def flatten(src: str) -> str:
    def expand(m):
        name = m.group(1)
        path = name if name.endswith('.tex') else name + '.tex'
        if os.path.exists(path):
            return open(path).read().rstrip('\n')
        return m.group(0)
    return re.sub(r'\\input\{([^}]+)\}(?:\s*%[^\n]*)?', expand, src)


def main() -> None:
    os.chdir(PAPER)
    flat = flatten(open('paper.tex').read())

    title = ' '.join(re.search(r'\\title\{(.+?)\}', flat, re.S).group(1).split())
    abstract = ' '.join(
        re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', flat, re.S)
        .group(1).split())

    body = flat.split('\\end{frontmatter}', 1)[-1].split('\\end{document}')[0]
    parts = re.split(r'\\(section|subsection)\*?\{([^}]*)\}', body)
    sections, i = [], 1
    while i < len(parts):
        sections.append({
            'level': 1 if parts[i] == 'section' else 2,
            'title': re.sub(r'\\label\{[^}]*\}', '', parts[i + 1]).strip(),
            'tex': parts[i + 2].strip(),
        })
        i += 3

    doc = {
        'title': title,
        'abstract': abstract,
        'sections': sections,
        'source': flat,
        'lines': len(flat.split('\n')),
        'nsec': sum(1 for s in sections if s['level'] == 1),
        'builtAt': int(time.time() * 1000),
    }

    os.makedirs(OUT, exist_ok=True)
    json.dump(doc, open(os.path.join(OUT, 'paper.json'), 'w'),
              ensure_ascii=False)
    payload = json.dumps(doc, ensure_ascii=False).replace(
        '</script>', '<\\/script>')
    open(os.path.join(OUT, 'payload.js'), 'w').write(
        'window.PAPER=' + payload + ';')

    print(f"{len(sections)} sections, {doc['lines']} lines, "
          f"{len(payload) // 1024} KB")
    print('Now republish the artifact, then clear sync/request in its store '
          'so the page\'s button returns to "Request sync".')


if __name__ == '__main__':
    main()
