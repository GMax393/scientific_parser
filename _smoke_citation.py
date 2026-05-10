import json
import sys

OUT_PATH = "_smoke_out.txt"
buf = []

def out(*args):
    buf.append(" ".join(str(a) for a in args))

from inference_pipeline import (
    PaperMetadata,
    format_citation,
    journal_indices_for,
    parse_citation_string,
    validate_draft_text,
)

cases = [
    "Иванов И. И., Петров А. Б. Распознавание образов в медицине // Журнал общей химии. — 2003. — Т. 73, № 5. — С. 12–18.",
    "Smith J., Doe A. (2015) Some article title here. Journal of X, 12(3), 45-67. DOI: 10.1234/xyz",
    "10.1038/s41586-020-2649-2",
    "Колмогоров А. Н. Основные понятия теории вероятностей. — М.: Наука, 1974. — 120 с.",
]

for idx, line in enumerate(cases, 1):
    m, info = parse_citation_string(line)
    out(f"=== CASE {idx}")
    out("INPUT  :", line)
    out("AUTHORS:", m.authors)
    out("TITLE  :", m.title)
    out("YEAR   :", m.year)
    out("JOURNAL:", m.journal)
    out("VOL/ISS:", m.volume, "/", m.issue)
    out("PAGES  :", m.pages)
    out("DOI    :", m.doi)
    out("VIA    :", info.get("matched_via"))
    out("GOST   :", format_citation(m, "gost"))
    out("INDICES:", journal_indices_for(m))
    out("")

draft = """1. Иванов И. И., Петров А. Б. Распознавание образов в медицине // Журнал общей химии. — 2003. — Т. 73, № 5. — С. 12–18.
2. Колмогоров А. Н. Основные понятия теории вероятностей. — М.: Наука, 1974. — 120 с.
3. ничего полезного"""

report = validate_draft_text(draft)
out("=== DRAFT REPORT")
out(json.dumps(report, ensure_ascii=False, indent=2)[:3000])

with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(buf))
print("OK")
