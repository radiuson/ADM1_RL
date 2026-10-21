#!/usr/bin/env python3
"""Build the payload for the bilingual (English/Chinese) reading copy.

The English reader keyed on sections is too coarse for translating: one
section of the introduction is nearly six thousand characters.  This splits
each section into paragraphs instead, which is the unit a translation is
revised in, and marks the blocks that carry no prose -- equations, tables,
figures -- so the page can skip them rather than ask for a translation of
LaTeX markup.

    python3 scripts/build_bilingual_payload.py <outdir>

Writes bilingual.json and bilingual.js into <outdir>.
"""
from __future__ import annotations

import io
import hashlib
import json
import os
import re
import sys
import time

PAPER = '/home/ihpc/code/biogas/ADM1/papers/mypaper/paper.tex'
# The Chinese working translation, kept in the repository so it survives an
# artifact being deleted or the author signing in from another account.  The
# page treats it as the baseline and the artifact's shared store as the
# working copy on top of it.
ZH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  'paper_zh', 'translations.json')

SEC_RE = re.compile(r'^\\(section|subsection)\*?\{(.+?)\}', re.M)

# A paragraph that is only one of these is structure, not prose.
ENV_RE = re.compile(
    r'^\\(begin\{(table\*?|figure\*?|longtable|align|equation|enumerate|'
    r'itemize|tabular)\}|end\{|includegraphics|label|caption|toprule|'
    r'midrule|bottomrule|addlinespace|multicolumn|cmidrule|onecolumn|'
    r'twocolumn|appendix|bibliography)')


def strip_label(title: str) -> str:
    title = re.sub(r'\\label\{[^}]*\}', '', title)
    return re.sub(r'\s+', ' ', title).strip()


def strip_leading_noise(block: str) -> str:
    """Drop labels and comment rules that sit above a paragraph's first word.

    A heading's \\label and the rule of % signs below it land at the top of
    the following paragraph, which would otherwise classify the paragraph as
    markup and hide the prose underneath it.
    """
    lines = block.split('\n')
    while lines and (lines[0].strip().startswith('%')
                     or re.fullmatch(r'\\label\{[^}]*\}', lines[0].strip())):
        lines.pop(0)
    return '\n'.join(lines).strip()


def classify(block: str) -> str:
    """'prose' when the block is text to translate, else what it is."""
    t = strip_leading_noise(block)
    if not t:
        return 'blank'
    if ENV_RE.match(t):
        return 'markup'
    # A block that is mostly a table body: many & separators, little prose.
    amp = t.count('&')
    if amp >= 4 and len(re.findall(r'[A-Za-z]{4,}', t)) < amp:
        return 'markup'
    # Needs at least a couple of real words to be worth translating.
    if len(re.findall(r'[A-Za-z]{3,}', t)) < 4:
        return 'markup'
    return 'prose'


def paragraphs(tex: str) -> list[dict]:
    out = []
    for raw in re.split(r'\n\s*\n', tex):
        if not raw.strip():
            continue
        body = strip_leading_noise(raw)
        kind = classify(raw)
        # Show the paragraph without the label/comment rule that preceded it,
        # so the English column is the text the translation pairs with.
        out.append({'tex': body if kind == 'prose' else raw.strip(),
                    'kind': kind})
    return out


def text_hash(t: str) -> str:
    """Short digest of a block, insensitive to line breaks and indentation."""
    return hashlib.sha1(
        re.sub(r'\s+', ' ', t).strip().encode('utf-8')).hexdigest()[:10]


def main() -> None:
    outdir = sys.argv[1]
    os.makedirs(outdir, exist_ok=True)
    src = io.open(PAPER, encoding='utf-8').read()

    m = re.search(r'\\title\{(.+?)\n*\}', src, re.S)
    title = re.sub(r'\s+', ' ', strip_label(m.group(1))) if m else 'Manuscript'

    m = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', src, re.S)
    abstract = m.group(1).strip() if m else ''

    heads = list(SEC_RE.finditer(src))
    units, sections = [], []
    if abstract:
        units.append({'sec': 'Abstract', 'level': 0, 'tex': abstract,
                      'kind': 'prose', 'h': text_hash(abstract)})
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(src)
        name = strip_label(h.group(2))
        level = 1 if h.group(1) == 'section' else 2
        sections.append({'title': name, 'level': level})
        for p in paragraphs(src[h.end():end]):
            units.append({'sec': name, 'level': level,
                          'tex': p['tex'], 'kind': p['kind'],
                          'h': text_hash(p['tex'])})

    n_prose = sum(1 for u in units if u['kind'] == 'prose')
    zh, zh_src = {}, {}
    if os.path.exists(ZH):
        doc = json.load(io.open(ZH, encoding='utf-8'))
        zh = doc.get('zh') or {}
        zh_src = doc.get('src') or {}
    payload = {
        'title': title,
        'sections': sections,
        'units': units,
        'nprose': n_prose,
        'nunits': len(units),
        'zh': zh,
        'zhSrc': zh_src,
        'builtAt': int(time.time() * 1000),
    }
    with io.open(os.path.join(outdir, 'bilingual.json'), 'w',
                 encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False)
    with io.open(os.path.join(outdir, 'bilingual.js'), 'w',
                 encoding='utf-8') as f:
        f.write('window.PAPER=')
        json.dump(payload, f, ensure_ascii=False)
        f.write(';')
    stale = sum(1 for k, v in zh.items()
                if v.strip() and int(k) < len(units)
                and zh_src.get(k) not in (None, units[int(k)]['h']))
    print(f'{len(sections)} headings, {len(units)} blocks, '
          f'{n_prose} translatable, {len(zh)} translated'
          + (f' ({stale} against changed English)' if stale else '')
          + f' -> {outdir}')


if __name__ == '__main__':
    main()
