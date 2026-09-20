#!/bin/bash
# Save a version of the manuscript to the history repository.
#
# The manuscript directory is deliberately not a git repository -- the paper is
# not published with the code -- so history is kept separately at
# ~/paper_history/mypaper.  Each run copies the sources and the compiled PDF
# there and commits whatever changed, so any earlier version can be recovered
# with an ordinary git checkout.
#
#     scripts/snapshot_paper.sh ["what changed"]
set -u

SRC="$HOME/code/biogas/ADM1/papers/mypaper"
DST="$HOME/paper_history/mypaper"

[ -d "$SRC" ] || { echo "manuscript not found: $SRC" >&2; exit 1; }
[ -d "$DST/.git" ] || { echo "history repo not found: $DST" >&2; exit 1; }

# Sources, bibliography, generated tables and the figures the paper includes,
# plus the compiled PDF so a version can be read without recompiling.
rsync -a --delete \
      --include='*.tex' --include='*.bib' --include='*.pdf' --include='*.md' \
      --exclude='*' \
      "$SRC/" "$DST/"

cd "$DST" || exit 1
if git diff --quiet && git diff --cached --quiet && [ -z "$(git status --porcelain)" ]; then
    echo "no change since the last snapshot"
    exit 0
fi

MSG="${1:-Snapshot $(date '+%Y-%m-%d %H:%M')}"
PAGES=$(pdfinfo paper.pdf 2>/dev/null | awk '/^Pages/{print $2}')
LINES=$(wc -l < paper.tex 2>/dev/null)

git add -A
git commit -q -m "$MSG" -m "paper.tex ${LINES:-?} lines, ${PAGES:-?} pages" || exit 1
echo "saved: $(git log --oneline -1)"
echo "history: $(git rev-list --count HEAD) versions in $DST"
