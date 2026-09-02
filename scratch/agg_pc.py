"""Agregacja licznikow wydajnosci benchmark_app (-pc) po layerType/execType."""
import re
import sys
from collections import defaultdict

RE = re.compile(
    r"layerType:\s*(?P<lt>\S+).*?execType:\s*(?P<et>\S+)\s+realTime \(ms\):\s*(?P<rt>[\d.]+)",
    re.S,
)


def load(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    by_type = defaultdict(lambda: [0, 0.0])
    by_exec = defaultdict(lambda: [0, 0.0])
    for m in RE.finditer(text):
        lt, et, rt = m.group("lt"), m.group("et"), float(m.group("rt"))
        by_type[lt][0] += 1
        by_type[lt][1] += rt
        by_exec[et][0] += 1
        by_exec[et][1] += rt
    return by_type, by_exec


def dump(title, d, limit=14):
    total = sum(v[1] for v in d.values())
    print(f"--- {title}  (suma {total:.3f} ms) ---")
    for k, (n, t) in sorted(d.items(), key=lambda kv: -kv[1][1])[:limit]:
        print(f"  {t:8.3f} ms  n={n:4d}  {k}")
    print()


for path in sys.argv[1:]:
    by_type, by_exec = load(path)
    print(f"===== {path} =====")
    dump("layerType", by_type)
    dump("execType", by_exec)
