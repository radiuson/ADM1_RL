#!/usr/bin/env python3
"""Write the page's shared translation back into the repository.

The page keeps the working copy in the artifact's shared store, which any
account that can open the artifact reads and writes.  The repository file is
the baseline the page starts from, so it has to be updated from that store or
the two drift apart -- and the store disappears if the artifact is deleted.

Pass the store's document as JSON on stdin or as a file.  ArtifactData's get
action saves it with the document wrapped in a "data" key; both shapes are
accepted.

    # read the shared doc with the ArtifactData tool, then:
    python3 scripts/export_translations.py shared.json
    python3 scripts/export_translations.py < shared.json
"""
from __future__ import annotations

import io
import json
import os
import sys

ZH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  'paper_zh', 'translations.json')
NOTE = ('Chinese working translation of paper.tex, keyed by the block index '
        'build_bilingual_payload.py assigns. src holds the hash of the '
        'English the translation was written against, so the page can flag a '
        'block whose English has since changed.')


def main() -> None:
    raw = (io.open(sys.argv[1], encoding='utf-8').read() if len(sys.argv) > 1
           else sys.stdin.read())
    doc = json.loads(raw)
    # An ArtifactData get wraps the document; a hand-written file may not.
    body = doc.get('data', doc)
    zh = {k: v for k, v in (body.get('zh') or {}).items() if v and v.strip()}
    src = body.get('src') or {}
    if not zh:
        print('no translations in the input, nothing written')
        return

    old = {}
    if os.path.exists(ZH):
        old = (json.load(io.open(ZH, encoding='utf-8')).get('zh') or {})

    added = sorted(set(zh) - set(old), key=int)
    changed = sorted((k for k in zh if k in old and zh[k] != old[k]), key=int)
    dropped = sorted(set(old) - set(zh), key=int)

    os.makedirs(os.path.dirname(ZH), exist_ok=True)
    with io.open(ZH, 'w', encoding='utf-8') as f:
        json.dump({'zh': zh, 'src': {k: v for k, v in src.items() if k in zh},
                   'note': NOTE}, f, ensure_ascii=False, indent=1)
        f.write('\n')

    chars = sum(len(v) for v in zh.values())
    print(f'wrote {len(zh)} blocks, {chars} characters to '
          f'{os.path.relpath(ZH, os.path.dirname(os.path.dirname(ZH)))}')
    for label, ks in (('new', added), ('changed', changed),
                      ('no longer in the store', dropped)):
        if ks:
            print(f'  {label}: ' + ', '.join('#' + str(int(k) + 1) for k in ks))


if __name__ == '__main__':
    main()
