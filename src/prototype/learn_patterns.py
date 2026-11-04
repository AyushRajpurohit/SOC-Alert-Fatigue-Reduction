"""
learn_patterns.py -- build the KNOWN-ATTACK PATTERN LIBRARY the engine matches against.

A pattern is a canonical tactic sequence. One baseline run gives its own sequence; several runs give their
longest common backbone (your sessionize_and_mine.longest_common_backbone).

IMPORTANT FOR HONEST EVALUATION: learn patterns only from the runs you want the engine to KNOW. Keep the variant
and the second chain OUT of the library when you evaluate, otherwise they are in-sample.

Usage:
    python learn_patterns.py --baseline data/lab/joined/joined_run{1,2,3,4,5}.csv --name original_chain \
        --out src/prototype/patterns.json
    # optional: grow the library with another known chain (single run) -- shows how new knowledge is added
    python learn_patterns.py --baseline ... --name original_chain --add exfil_chain=data/lab/joined/joined_exfil2.csv --out ...
"""
import argparse
import _common_path  # noqa: F401
from sessionize_and_mine import longest_common_backbone
from soc_common import wazuh_joined_adapter, tokens
from socflow import save_patterns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", nargs="+", required=True)
    ap.add_argument("--name", default="original_chain")
    ap.add_argument("--add", nargs="*", default=[], help="name=path single-run patterns")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    seqs = [tokens(wazuh_joined_adapter(p), "canonical") for p in args.baseline]
    pats = [{"name": args.name, "tokens": longest_common_backbone(seqs), "mode": "canonical",
             "built_from": args.baseline}]
    for kv in args.add:
        name, path = kv.split("=", 1)
        pats.append({"name": name, "tokens": tokens(wazuh_joined_adapter(path), "canonical"),
                     "mode": "canonical", "built_from": [path]})
    save_patterns(args.out, pats)
    for p in pats:
        print(f"{p['name']:<16} ({len(p['tokens'])} tokens): {' -> '.join(p['tokens'])}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
