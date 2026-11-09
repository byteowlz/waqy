"""Cut an immutable public subset locally; no retraining or network access.

    uv run --no-project python curate_release.py SOURCE OUT_ROOT NAME --heads hey_hex,hex

Use --dry-run to inspect without writing. Unknown provenance needs a hash-bound
--metadata review; owner-derived heads need exact-scope --consent. Neither record
is itself uploaded. See training/README.md for the record shapes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from publication import prepare, selected_heads, write_release


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("out_root", type=Path)
    parser.add_argument("name")
    parser.add_argument("--heads", required=True, help="comma-separated classifier IDs; never inferred")
    parser.add_argument("--repo", default="byteowlz/waqy", help="destination bound by owner publication consent")
    parser.add_argument("--consent", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        publication = prepare(args.source, args.name, selected_heads(args.heads), args.repo, args.consent, args.metadata)
        result = {"release": args.name, "heads": publication.manifest["classifiers"],
                  "manifest": publication.manifest, "files": sorted(["manifest.json", *publication.files]),
                  "dry_run": args.dry_run}
        if not args.dry_run:
            result["out"] = str(write_release(publication, args.out_root))
        print(json.dumps(result, indent=1, ensure_ascii=False))
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
