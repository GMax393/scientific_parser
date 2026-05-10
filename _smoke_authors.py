import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from inference_pipeline import _AUTHOR_TOKEN_RE, _AUTHOR_SEP_RE, _extract_leading_authors

cases = [
    "Smith J., Doe A.",
    "Smith J. and Doe A.",
    "Иванов И. И., Петров А. Б.",
    "J. Smith, A. Doe",
]
for c in cases:
    print("---", repr(c))
    print("authors:", _extract_leading_authors(c))
    pos = 0
    while pos < len(c):
        m = _AUTHOR_TOKEN_RE.match(c, pos)
        if not m:
            print("  no match at", pos, "rest=", repr(c[pos:]))
            break
        print("  match span", m.span(), "groups", m.groupdict())
        pos = m.end()
        s = _AUTHOR_SEP_RE.match(c[pos:])
        if not s:
            print("  no sep")
            break
        pos += s.end()
