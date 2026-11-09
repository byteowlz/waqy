"""Pure, local preparation of an explicit public subset. No HF client or uploads."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

FRONTEND = {"mel": "melspectrogram_raw.onnx", "embedding": "embedding_model.onnx"}
IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9_-]{0,95}\Z")
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024


def private_roster() -> set[str]:
    root = Path(os.environ.get("WAQY_PRIVATE_WORDS", Path.home() / ".local/share/foxline/waqy-roster-private/words"))
    return {json.loads(path.read_text()).get("name", path.stem) for path in root.glob("*.json")} if root.exists() else set()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=1, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode()


def read_json(path: Path) -> tuple[dict, bytes]:
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("JSON record exceeds 1 MiB")
    with path.open("rb") as file:
        data = file.read(1024 * 1024 + 1)
    if len(data) > 1024 * 1024:
        raise ValueError("JSON record exceeds 1 MiB")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise TypeError("expected a JSON object")
    return value, data


def identifier(value: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError("release/head IDs must be lowercase letters, digits, underscores or hyphens")
    return value


def selected_heads(value: str) -> list[str]:
    heads = [identifier(part.strip()) for part in value.split(",")]
    if not heads or len(heads) > 128 or len(set(heads)) != len(heads):
        raise ValueError("select 1-128 distinct head IDs explicitly")
    if set(heads) & set(FRONTEND):
        raise ValueError("front-end IDs are not classifier heads")
    return heads


def scoped_record(path: Path | None, source_hash: str, heads: list[str]) -> dict:
    if path is None:
        return {}
    record, raw = read_json(path.expanduser())
    if record.get("source_manifest_sha256") != source_hash:
        raise ValueError("review/consent record names a different source manifest")
    scope = record.get("heads")
    if not isinstance(scope, (list, dict)) or sorted(scope) != sorted(heads):
        raise ValueError("review/consent record must cover exactly the selected heads")
    return {**record, "_record_sha256": digest(raw)}


def public_text(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 160 or any(c in value for c in "\r\n/\\<>|"):
        raise ValueError("invalid public phrase/kind")
    return value


def public_boolean(value):
    if type(value) is not bool:
        raise ValueError("expected an explicit boolean")
    return value


def nonnegative_number(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("expected a finite nonnegative measurement")
    return value


def unit_interval(value):
    if nonnegative_number(value) > 1:
        raise ValueError("threshold/recall/median must be between zero and one")
    return value


def public_languages(value):
    if not isinstance(value, list) or not all(isinstance(x, str) and re.fullmatch(r"[a-z]{2,3}(?:-[A-Za-z0-9]+)?", x) for x in value):
        raise ValueError("invalid public languages")
    return value


def head_list(value):
    if not isinstance(value, list) or not all(isinstance(x, str) and IDENTIFIER.fullmatch(x) for x in value):
        raise ValueError("expected a list of head IDs")
    return value


PUBLIC_FIELDS = {
    "phrase": public_text, "kind": public_text, "languages": public_languages,
    "addressed": public_boolean, "onset_gate": public_boolean,
    "threshold": unit_interval, "pos_test_median": unit_interval,
    "owner_holdout_recall": unit_interval, "false_wakes_per_hour": nonnegative_number,
    "implies": head_list, "suppress": head_list,
}


def public_metadata(head: str, original: dict, reviewed: dict, private: set[str]) -> dict:
    metadata = {**original, **reviewed}
    for flag in ("private", "owner_recordings"):
        if flag in original and (type(original[flag]) is not bool or reviewed.get(flag, original[flag]) != original[flag]):
            raise ValueError(f"{head}: review cannot override recorded {flag}")
        if type(metadata.get(flag)) is not bool:
            raise ValueError(f"{head}: explicit boolean {flag} is required (use a hash-bound --metadata review)")
    base = head.split("_", 1)[-1] if head.startswith(("hey_", "he_")) else head
    if metadata["private"] or base in private:
        raise ValueError(f"{head}: personal/private heads are never published")
    # No training paths, raw reports, source names, private notes or arbitrary nested metadata.
    public = {key: validate(metadata[key]) for key, validate in PUBLIC_FIELDS.items() if key in metadata}
    return {"private": False, "owner_recordings": metadata["owner_recordings"], **public}


def read_artifact(source: Path, entry: dict) -> bytes:
    name = entry["file"]
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+\.onnx", name):
        raise ValueError("artifact paths must be flat ONNX filenames")
    path = source / name
    if path.is_symlink() or not path.is_file() or path.resolve().parent != source.resolve() or path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("artifact must be a bounded regular file inside the release")
    with path.open("rb") as file:
        data = file.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES or digest(data) != entry["sha256"]:
        raise ValueError(f"release hash mismatch: {name}")
    return data


@dataclass
class Publication:
    manifest: dict
    files: dict[str, bytes]

    @property
    def manifest_bytes(self) -> bytes:
        return json_bytes(self.manifest)


def prepare(source: Path, name: str, heads: list[str], repo: str,
            consent: Path | None = None, metadata: Path | None = None) -> Publication:
    identifier(name)
    selected_heads(",".join(heads))
    source = source.expanduser()
    manifest, original_bytes = read_json(source / "manifest.json")
    source_hash = digest(original_bytes)
    if len(manifest["classifiers"]) != len(set(manifest["classifiers"])):
        raise ValueError("source has duplicate classifier IDs")
    if any(head not in manifest["classifiers"] for head in heads):
        raise ValueError("selected head is not in the source release")
    review_record = scoped_record(metadata, source_hash, heads)
    review = review_record.get("heads", {})
    if not isinstance(review, dict):
        raise TypeError("--metadata heads must map IDs to reviewed metadata")
    private = private_roster()
    selected = {head: public_metadata(head, manifest.get("heads", {}).get(head, {}), review.get(head, {}), private) for head in heads}
    for head, info in selected.items():
        for key in ("implies", "suppress"):
            if set(info.get(key, [])) - set(heads):
                raise ValueError(f"{head}: {key} references an excluded head")
    consent_info = None
    if any(info["owner_recordings"] for info in selected.values()):
        record = scoped_record(consent, source_hash, heads)
        if not isinstance(record.get("heads"), list) or record.get("publish_derived_weights") is not True or record.get("repo") != repo:
            raise ValueError("owner-derived heads need explicit publication consent for this repo, source hash and exact subset")
        # Publish the scope and record hash, never the private consent document or personal ID.
        consent_info = {"sha256": record["_record_sha256"], "repo": repo, "heads": heads, "source_manifest_sha256": source_hash}
    files, entries, total = {}, {}, 0
    for key in [*FRONTEND, *heads]:
        entry = manifest["files"][key]
        data = read_artifact(source, entry)
        total += len(data)
        if total > MAX_TOTAL_BYTES:
            raise ValueError("selected artifacts exceed 256 MiB")
        target = FRONTEND.get(key, f"{key}.onnx")
        files[target] = data
        entries[key] = {"file": target, "sha256": digest(data)}
    threshold = unit_interval(manifest["threshold"])
    public = {"release": name, "threshold": threshold, "classifiers": heads,
              "heads": selected, "files": entries,
              "training_data": "See per-head owner_recordings and immutable source manifest digest; raw audio is not distributed.",
              "curated_from": {"manifest_sha256": source_hash, "metadata_review_sha256": review_record.get("_record_sha256"), "owner_consent": consent_info}}
    return Publication(public, files)


def write_release(publication: Publication, out_root: Path) -> Path:
    out = out_root.expanduser() / publication.manifest["release"]
    out.mkdir(parents=True, exist_ok=False)
    try:
        for name, data in publication.files.items():
            (out / name).write_bytes(data)
        (out / "manifest.json").write_bytes(publication.manifest_bytes)
    except OSError:
        shutil.rmtree(out)
        raise
    return out
