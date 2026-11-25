"""Which vault items have stored model outputs on themselves or on a copy; writes configs/vault-exposure.yaml.

Every text file under the given roots (default: runs/ and results/, without the parts cache, which holds item texts
but no outputs, and without strata tables, which are uid lists) is searched for item uids: Round 2 uids `<source>/<id>` and Round 1's external uids (`gsm8k_*/<row>`,
`mmlu_pro/<id>`, `math500/<unique_id>`, which maps to its MATH test item by text). A vault item is exposed when a
mentioned item is the item itself or one of its copies under the de-duplication criterion (`external.copies`, one
step). The output lists SHA-256 hashes of the exposed vault uids, and counts by source and by origin (the first two
components of the path holding the mention); never uids or texts. `external.exposed(uids)` reads it at the opening.
The record only grows: the hashes of the previous record are kept, and so are its origins that no longer exist on
disk, so deleting old outputs never hides an exposure.

Usage (compute host, repository root, after `source scripts/env.sh`):
    python scripts/vault_exposure.py [root ...]
"""

import collections
import gzip
import hashlib
import re
import sys
from pathlib import Path

import yaml

from evotaste import external, sources

OUT = Path("configs/vault-exposure.yaml")
SKIP = {Path("results/cache")}
BINARY = {".safetensors", ".pt", ".bin", ".pkl", ".npy", ".png", ".pdf", ".lock"}


def texts(root: Path):
    """Text files that may hold model outputs; strata tables are inputs (uid lists, possibly synthetic), and a
    census's own table is derived from its records, which are read."""
    for path in sorted(root.rglob("*")):
        if (not path.is_file() or path.suffix in BINARY or any(s in path.parents for s in SKIP)
                or path.name.startswith("strata")):
            continue
        data = path.read_bytes()
        if path.suffix == ".gz":
            data = gzip.decompress(data)
        yield path, data.decode("utf-8", errors="replace")


def main(roots: list[Path]) -> None:
    f = external.SuitesFile(external.SUITES)
    items, parts, reasons, _, (groups, near) = external.build_with_copies(f.units, f.tokenizer)
    index = {it.uid: k for k, it in enumerate(items)}
    by_text = {external.normalized(it.problem): k for k, it in enumerate(items) if it.source == "math_test"}
    math500 = {r["unique_id"]: by_text.get(external.normalized(r["problem"]))
               for r in sources._load("HuggingFaceH4/MATH-500", None, "test")}
    prefixes = sorted([*sources.SOURCES, "math500"], key=len, reverse=True)
    pattern = re.compile(rf"(?<![\w/])({'|'.join(map(re.escape, prefixes))})/([^\s\"',\]\)\}}|]+)")
    mentions = collections.defaultdict(set)  # item index -> origins
    for root in roots:
        for path, text in texts(root):
            origin = "/".join([str(root), *path.relative_to(root).parts[:2]])  # e.g. runs/pilot0/base
            for m in pattern.finditer(text):
                prefix, rest = m.groups()
                k = math500.get(rest) if prefix == "math500" else index.get(f"{prefix}/{rest}")
                if k is not None:
                    mentions[k].add(origin)
    neighbours = collections.defaultdict(set)
    for g in groups:
        for i in g:
            neighbours[i].update(j for j in g if j != i)
    for i, j in near:
        neighbours[i].add(j)
        neighbours[j].add(i)
    vault = {k for k, (p, why) in enumerate(zip(parts, reasons)) if p == external.VAULT and why is None}
    exposed = collections.defaultdict(set)  # vault item -> (origin, how)
    for k, origins in mentions.items():
        for v in ({k} | neighbours[k]) & vault:
            exposed[v].update((o, "itself" if v == k else "copy") for o in origins)
    previous = yaml.safe_load(OUT.read_text()) if OUT.exists() else {"sha256": [], "by_origin": []}
    source_of = {hashlib.sha256(it.uid.encode()).hexdigest(): it.source for it in items}
    hashes = {hashlib.sha256(items[v].uid.encode()).hexdigest() for v in exposed} | set(previous["sha256"])
    unknown = hashes - set(source_of)
    if unknown:
        raise SystemExit(f"{len(unknown)} recorded exposures name no current item; the uid scheme changed?")
    by_source = collections.Counter(source_of[h] for h in hashes)
    by_origin = [{"origin": o, "how": how, "source": s, "items": n} for (o, how, s), n in sorted(collections.Counter(
        (o, how, items[v].source) for v, e in exposed.items() for o, how in e).items())]
    by_origin += [e for e in previous["by_origin"] if not Path(e["origin"]).exists()]  # deleted outputs
    doc = {"generated": {"script": "scripts/vault_exposure.py", "roots": [str(r) for r in roots],
                         "suites_sha256": hashlib.sha256(external.SUITES.read_bytes()).hexdigest(),
                         "criterion": "the item itself or a copy under external.copies (one step)"},
           "exposed": len(hashes), "by_source": dict(sorted(by_source.items())),
           "by_origin": sorted(by_origin, key=lambda e: (e["origin"], e["how"], e["source"])),
           "sha256": sorted(hashes)}
    header = ("# Generated by scripts/vault_exposure.py; do not edit by hand. Vault items with a stored model output on\n"
              "# the item or on a copy of it, as SHA-256 hashes of their uids only. external.exposed(uids) reads it.\n")
    OUT.write_text(header + yaml.safe_dump(doc, sort_keys=False, width=120))
    print(f"{len(hashes)} exposed vault items; by source {dict(by_source)}")
    for e in doc["by_origin"]:
        print(f"  {e['origin']:45s} {e['how']:7s} {e['source']:20s} {e['items']}")


if __name__ == "__main__":
    main([Path(a) for a in sys.argv[1:]] or [Path("runs"), Path("results")])
