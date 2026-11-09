"""Preview or publish only explicitly selected wake heads to Hugging Face.

    uv run --no-project python publish_hf.py SOURCE --heads hey_hex,hex --name public-1

Default: offline JSON dry run, no credentials/client/network or local release writes.
Use --catalog for an offline merge preview; --apply performs a fresh remote check
and one parent-fenced atomic commit. This tool never trains or qualifies a head.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

from head_export_config import resolve_options
from publication import (
    Publication,
    digest,
    json_bytes,
    prepare,
    read_json,
)

ROOT = Path(__file__).resolve().parent.parent
CATALOG_SCHEMA = "waqy-catalog/1"


def catalog_entry(publication: Publication) -> dict:
    manifest = publication.manifest
    heads = []
    for name in manifest["classifiers"]:
        entry, meta = manifest["files"][name], manifest["heads"][name]
        word = name.split("_", 1)[1] if name.startswith(("hey_", "he_")) else name
        heads.append({"name": name, "file": entry["file"], "sha256": entry["sha256"],
                      "bytes": len(publication.files[entry["file"]]),
                      "phrase": meta.get("phrase", word.replace("_", " ").title()),
                      "languages": meta.get("languages", []), "kind": meta.get("kind", "wake"),
                      "addressed": meta.get("addressed", name.startswith(("hey_", "he_"))),
                      "owner_recordings": meta["owner_recordings"],
                      "measured": {k: meta[k] for k in ("pos_test_median", "owner_holdout_recall", "false_wakes_per_hour") if k in meta}})
    return {"name": manifest["release"], "threshold": manifest["threshold"],
            "manifest_sha256": digest(publication.manifest_bytes),
            "frontend": {k: {**manifest["files"][k], "bytes": len(publication.files[manifest["files"][k]["file"]])} for k in ("mel", "embedding")},
            "curated_from": manifest["curated_from"], "heads": heads}


def merged_catalog(existing: dict | None, entry: dict) -> dict:
    catalog = copy.deepcopy(existing) if existing is not None else {"schema": CATALOG_SCHEMA, "repo": None, "releases": []}
    if catalog.get("schema") != CATALOG_SCHEMA or not isinstance(catalog.get("releases"), list):
        raise ValueError("unknown or malformed catalog schema")
    names = [old["name"] for old in catalog["releases"]]
    if len(names) != len(set(names)):
        raise ValueError("catalog has duplicate release IDs")
    for old in catalog["releases"]:
        if old["name"] == entry["name"]:
            if old["manifest_sha256"] != entry["manifest_sha256"]:
                raise ValueError("release ID already names a different manifest; pick a new name")
            return catalog
    catalog["releases"].append(entry)
    return catalog


def card(manifest: dict, catalog: dict) -> bytes:
    rows = []
    for release in catalog["releases"]:
        heads = ", ".join(f"`{head['name']}`" for head in release["heads"])
        rows.append(f"| `{release['name']}` | {heads} |")
    measurements = []
    for head in manifest["classifiers"]:
        meta = manifest["heads"][head]
        measurements.append(f"| `{head}` | {meta.get('owner_holdout_recall', 'not supplied')} | {meta.get('false_wakes_per_hour', 'not supplied')} | {meta['owner_recordings']} |")
    return ("""---
license: apache-2.0
tags: [wake-word, keyword-spotting, speech, onnx, rust, wasm]
library_name: waqy
---

# waqy wake-phrase releases

Each immutable release contains a shared mel/speech-embedding front end and only
its selected classifier heads. `manifest.json` and `catalog.json` carry SHA-256
hashes. Audio, private head names, raw reports and consent documents are not distributed.
A detection is evidence, never authentication, speaker identity or permission to act.

## Published releases

| release | heads |
|---|---|
""" + "\n".join(rows) + f"""

## Selected release `{manifest['release']}`

| head | owner holdout recall | general false wakes/hour | owner-derived |
|---|---|---|---|
""" + "\n".join(measurements) + """

Numbers are copied from reviewed per-head metadata, not recomputed by publication.
Missing evidence is reported as not supplied; publication is not an accuracy or
security qualification. Observation conditions and frozen operating points must
be reviewed separately. Each release records its source manifest digest and, for
owner-derived heads, the digest and exact scope of publication consent.

Use `ears wake catalog` and `ears wake install <head>` to select heads. Native waqy
and browser/WASM hosts load the same manifest and verify its hashes.
See LICENSE and NOTICE for the front-end and training-tool attribution. Per-head
threshold support depends on the consumer; publication does not enable it.
""").encode()



def upload_files(publication: Publication, catalog: dict) -> dict[str, bytes]:
    prefix = publication.manifest["release"] + "/"
    files = {prefix + name: data for name, data in publication.files.items()}
    files[prefix + "manifest.json"] = publication.manifest_bytes
    files.update({"catalog.json": json_bytes(catalog), "README.md": card(publication.manifest, catalog)})
    for name in ("LICENSE", "NOTICE"):
        files[name] = (ROOT / name).read_bytes()
    return files


def plan(publication: Publication, repo: str, existing: dict | None = None) -> tuple[dict, dict[str, bytes]]:
    if existing and existing.get("repo") not in (None, repo):
        raise ValueError("catalog belongs to a different repository")
    catalog = merged_catalog(existing, catalog_entry(publication))
    catalog["repo"] = repo
    files = upload_files(publication, catalog)
    result = {"repo": repo, "release": publication.manifest["release"],
              "heads": publication.manifest["classifiers"], "manifest": publication.manifest,
              "manifest_sha256": digest(publication.manifest_bytes), "catalog": catalog,
              "files": [{"path": path, "sha256": digest(data), "bytes": len(data)} for path, data in sorted(files.items())]}
    return result, files


def remote_state(api, repo: str, download) -> tuple[str, dict | None, list[str]]:
    info = api.repo_info(repo_id=repo, repo_type="model", revision="main")
    if info.private:
        raise ValueError("target is private; visibility changes require separate approval")
    paths = api.list_repo_files(repo_id=repo, repo_type="model", revision=info.sha)
    if "catalog.json" in paths:
        catalog = json.loads(download(repo, "catalog.json", info.sha))
    else:
        unexpected = set(paths) - {".gitattributes", "README.md", "LICENSE", "NOTICE"}
        if unexpected:
            raise ValueError("nonempty repository lacks a catalog; audit/backfill it before publishing")
        catalog = None
    return info.sha, catalog, paths


def check_remote_release(publication: Publication, paths: list[str], repo: str, revision: str, download) -> None:
    prefix = publication.manifest["release"] + "/"
    remote = {path for path in paths if path.startswith(prefix)}
    if not remote:
        return
    expected = {prefix + name: data for name, data in publication.files.items()}
    expected[prefix + "manifest.json"] = publication.manifest_bytes
    if remote != set(expected):
        raise ValueError("immutable release folder has a different file set; pick a new name")
    for path, data in expected.items():
        if digest(download(repo, path, revision)) != digest(data):
            raise ValueError("immutable release content differs; pick a new name")


def apply(publication: Publication, repo: str, api, download, operation) -> dict:
    revision, existing, paths = remote_state(api, repo, download)
    check_remote_release(publication, paths, repo, revision, download)
    result, files = plan(publication, repo, existing)
    operations = [operation(path_in_repo=path, path_or_fileobj=data) for path, data in files.items()]
    commit = api.create_commit(repo_id=repo, repo_type="model", revision="main", parent_commit=revision,
                               operations=operations, commit_message=f"Add selected heads: {publication.manifest['release']}")
    return {**result, "dry_run": False, "remote_checked": True, "commit": commit.oid}


def publish(publication: Publication, repo: str) -> dict:
    # Lazy imports: a dry run does not discover credentials or instantiate an HF client.
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    api = HfApi()
    def download(repo_id, path, revision):
        return Path(hf_hub_download(repo_id, path, repo_type="model", revision=revision)).read_bytes()
    # Target creation/visibility changes are deliberately not implicit in head publication.
    return apply(publication, repo, api, download, CommitOperationAdd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?")
    parser.add_argument("--config", type=Path, help="optional private TOML profile; CLI overrides it")
    parser.add_argument("--heads")
    parser.add_argument("--name", help="new immutable public release ID")
    parser.add_argument("--repo")
    parser.add_argument("--consent", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--catalog", type=Path, help="optional existing catalog for offline preview only")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="default; offline and read-only")
    mode.add_argument("--apply", action="store_true", help="publish the exact selected subset to an existing public repo")
    args = parser.parse_args()
    try:
        options = resolve_options(args)
        if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", options["repo"]):
            raise ValueError("--repo must be namespace/repository")
        if args.apply and options["catalog"]:
            raise ValueError("--apply fetches a fresh remote catalog; --catalog is offline-only")
        publication = prepare(options["source"], options["name"], options["heads"], options["repo"], options["consent"], options["metadata"])
        if args.apply:
            result = publish(publication, options["repo"])
        else:
            existing = read_json(options["catalog"].expanduser())[0] if options["catalog"] else None
            result, _ = plan(publication, options["repo"], existing)
            result.update(dry_run=True, remote_checked=False)
        print(json.dumps(result, indent=1, ensure_ascii=False, allow_nan=False))
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
