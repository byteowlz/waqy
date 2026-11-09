"""Generated-byte fixtures and a fake HF client only: no models, identity data or network."""
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import curate_release
import publish_hf
from publication import digest, json_bytes, prepare, selected_heads, write_release

REPO = "example/wake-heads"


class FakeHub:
    def __init__(self, files=None, private=False):
        self.files = dict(files or {})
        self.sha = "remote-base"
        self.private = private
        self.commits = []
        self.race = False

    def repo_info(self, **kwargs):
        return SimpleNamespace(sha=self.sha, private=self.private)

    def list_repo_files(self, **kwargs):
        assert kwargs["revision"] == self.sha
        return list(self.files)

    def download(self, repo, path, revision):
        assert repo == REPO and revision == "remote-base"
        return self.files[path]

    def create_commit(self, **kwargs):
        if self.race or kwargs["parent_commit"] != self.sha:
            raise ValueError("remote moved; parent_commit conflict")
        update = {op.path_in_repo: op.path_or_fileobj for op in kwargs["operations"]}
        self.files = {**self.files, **update}
        self.commits.append(kwargs)
        return SimpleNamespace(oid="published-commit")


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "internal pack"
        self.source.mkdir()
        self.env = patch.dict("os.environ", {"WAQY_PRIVATE_WORDS": str(self.root / "private-roster")})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.manifest = {"release": "internal", "threshold": 0.5,
                         "training_data": "LOCAL_ONLY_PRIVATE_MARKER",
                         "classifiers": ["hey_hex", "hex", "hey_pi", "private_name"],
                         "files": {}, "heads": {}}
        for key in ["mel", "embedding", *self.manifest["classifiers"]]:
            name, data = key + ".onnx", ("fake ONNX bytes for " + key).encode()
            (self.source / name).write_bytes(data)
            self.manifest["files"][key] = {"file": name, "sha256": digest(data), "trained_as": "LOCAL_ONLY_PRIVATE_MARKER"}
        for head in self.manifest["classifiers"]:
            self.manifest["heads"][head] = {"private": head == "private_name", "owner_recordings": False,
                                            "phrase": "Test phrase", "languages": ["de", "en"],
                                            "private_notes": "LOCAL_ONLY_PRIVATE_MARKER"}
        (self.source / "stray.onnx").write_bytes(b"never upload")
        (self.source / "recording.wav").write_bytes(b"never upload audio")
        self.save()

    def save(self):
        (self.source / "manifest.json").write_bytes(json_bytes(self.manifest))

    def prepared(self, heads=None, **kwargs):
        return prepare(self.source, "public-1", heads or ["hey_hex"], REPO, **kwargs)

    def record(self, name, heads, **fields):
        path = self.root / name
        path.write_bytes(json_bytes({"source_manifest_sha256": digest((self.source / "manifest.json").read_bytes()),
                                     "heads": heads, **fields}))
        return path

    def test_selected_subset_preserves_hashes_not_internal_metadata(self):
        publication = self.prepared(["hey_hex", "hey_pi"])
        self.assertEqual(publication.manifest["classifiers"], ["hey_hex", "hey_pi"])
        self.assertEqual(set(publication.manifest["files"]), {"mel", "embedding", "hey_hex", "hey_pi"})
        self.assertEqual(publication.manifest["files"]["mel"]["sha256"], self.manifest["files"]["mel"]["sha256"])
        self.assertNotIn("LOCAL_ONLY_PRIVATE_MARKER", publication.manifest_bytes.decode())
        self.assertNotIn("private_name", publication.manifest_bytes.decode())
        self.assertEqual(publication.manifest["curated_from"]["manifest_sha256"], digest((self.source / "manifest.json").read_bytes()))

    def test_head_selection_never_defaults_to_all_or_accepts_paths(self):
        for value in ["", ",", "hey_hex,", "hey_hex,hey_hex", "../head", "a/b", "mel", "embedding"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                selected_heads(value)
        with self.assertRaisesRegex(ValueError, "not in the source"):
            self.prepared(["absent"])
        for name in ["../other", "/tmp/release", "", "a/b"]:
            with self.assertRaises(ValueError):
                prepare(self.source, name, ["hey_hex"], REPO)

    def test_private_flag_and_roster_cannot_be_overridden(self):
        with self.assertRaisesRegex(ValueError, "never published"):
            self.prepared(["private_name"])
        review = self.record("review.json", {"private_name": {"private": False}})
        with self.assertRaisesRegex(ValueError, "cannot override"):
            self.prepared(["private_name"], metadata=review)
        roster = self.root / "private-roster"
        roster.mkdir()
        (roster / "hex.json").write_text('{"name":"hex"}')
        with self.assertRaisesRegex(ValueError, "never published"):
            self.prepared()

    def test_missing_provenance_requires_hash_bound_review(self):
        self.manifest["heads"]["hey_hex"] = {}
        self.save()
        with self.assertRaisesRegex(ValueError, "explicit boolean"):
            self.prepared()
        review = self.record("review.json", {"hey_hex": {"private": False, "owner_recordings": False}})
        self.prepared(metadata=review)
        self.manifest["threshold"] = 0.6
        self.save()
        with self.assertRaisesRegex(ValueError, "different source"):
            self.prepared(metadata=review)

    def test_owner_publication_consent_is_exact_and_not_copied(self):
        self.manifest["heads"]["hex"]["owner_recordings"] = True
        self.save()
        with self.assertRaisesRegex(ValueError, "explicit publication consent"):
            self.prepared(["hex"])
        for fields in [{"publish_derived_weights": False, "repo": REPO},
                       {"publish_derived_weights": "true", "repo": REPO},
                       {"publish_derived_weights": True, "repo": "another/repo"}]:
            consent = self.record("consent.json", ["hex"], **fields)
            with self.assertRaisesRegex(ValueError, "explicit publication consent"):
                self.prepared(["hex"], consent=consent)
        consent = self.record("consent.json", ["hex"], publish_derived_weights=True, repo=REPO, private_identity="LOCAL_ONLY_PRIVATE_MARKER")
        publication = self.prepared(["hex"], consent=consent)
        self.assertEqual(publication.manifest["curated_from"]["owner_consent"]["sha256"], digest(consent.read_bytes()))
        self.assertNotIn("LOCAL_ONLY_PRIVATE_MARKER", publication.manifest_bytes.decode())
        with self.assertRaisesRegex(ValueError, "exactly the selected"):
            self.prepared(["hex", "hey_hex"], consent=consent)
        override = self.record("review.json", {"hex": {"owner_recordings": False}})
        with self.assertRaisesRegex(ValueError, "cannot override"):
            self.prepared(["hex"], metadata=override)

    def test_artifact_hash_path_and_symlink_checks_precede_publication(self):
        (self.source / "hey_hex.onnx").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.prepared()
        self.manifest["files"]["hey_hex"]["file"] = "../outside.onnx"
        self.save()
        with self.assertRaisesRegex(ValueError, "flat ONNX"):
            self.prepared()
        (self.root / "outside.onnx").write_bytes(b"secret")
        (self.source / "link.onnx").symlink_to(self.root / "outside.onnx")
        self.manifest["files"]["hey_hex"] = {"file": "link.onnx", "sha256": digest(b"secret")}
        self.save()
        with self.assertRaisesRegex(ValueError, "inside the release"):
            self.prepared()

    def test_runtime_relationships_cannot_point_at_excluded_heads(self):
        self.manifest["heads"]["hey_hex"]["implies"] = ["hex"]
        self.save()
        with self.assertRaisesRegex(ValueError, "excluded head"):
            self.prepared()
        self.assertEqual(self.prepared(["hey_hex", "hex"]).manifest["heads"]["hey_hex"]["implies"], ["hex"])

    def test_default_dry_run_is_offline_and_does_not_modify_source(self):
        before = {p.name: p.read_bytes() for p in self.source.iterdir()}
        args = ["publish_hf.py", str(self.source), "--heads", "hey_hex", "--name", "public-1", "--repo", REPO]
        stdout = io.StringIO()
        with patch.object(sys, "argv", args), patch.object(publish_hf, "publish", side_effect=AssertionError("network")), patch.dict(sys.modules, {"huggingface_hub": None}), contextlib.redirect_stdout(stdout):
            self.assertEqual(publish_hf.main(), 0)
        plan = json.loads(stdout.getvalue())
        self.assertTrue(plan["dry_run"])
        self.assertFalse(plan["remote_checked"])
        self.assertEqual({p.name: p.read_bytes() for p in self.source.iterdir()}, before)
        paths = {entry["path"] for entry in plan["files"]}
        self.assertEqual(paths, {"public-1/manifest.json", "public-1/melspectrogram_raw.onnx", "public-1/embedding_model.onnx", "public-1/hey_hex.onnx", "LICENSE", "NOTICE", "README.md", "catalog.json"})
        self.assertNotIn("LOCAL_ONLY_PRIVATE_MARKER", stdout.getvalue())

    def test_cli_subprocess_and_curator_are_local_only(self):
        result = subprocess.run([sys.executable, str(Path(publish_hf.__file__)), str(self.source), "--name", "public-1", "--heads", "hey_hex"], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)["heads"], ["hey_hex"])
        repo = Path(publish_hf.__file__).resolve().parent.parent
        recipe = subprocess.run(["just", "publish-heads", str(self.source), "public-1", "hey_hex"], cwd=repo, capture_output=True, text=True, check=True)
        self.assertTrue(json.loads(recipe.stdout)["dry_run"])
        out = self.root / "curated"
        args = ["curate_release.py", str(self.source), str(out), "public-1", "--heads", "hey_hex", "--dry-run"]
        with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(curate_release.main(), 0)
        self.assertFalse(out.exists())
        publication = self.prepared()
        release = write_release(publication, out)
        self.assertEqual({p.name for p in release.iterdir()}, {"manifest.json", *publication.files})
        with self.assertRaises(FileExistsError):
            write_release(publication, out)

    def test_atomic_fake_hf_commit_uses_exact_bytes_and_parent_revision(self):
        publication = self.prepared()
        hub = FakeHub()
        result = publish_hf.apply(publication, REPO, hub, hub.download, SimpleNamespace)
        self.assertEqual(result["commit"], "published-commit")
        self.assertEqual(len(hub.commits), 1)
        self.assertEqual(hub.commits[0]["parent_commit"], "remote-base")
        self.assertEqual(set(hub.files), {row["path"] for row in result["files"]})
        self.assertEqual(json.loads(hub.files["catalog.json"])["releases"][0]["heads"][0]["name"], "hey_hex")
        self.assertNotIn("LOCAL_ONLY_PRIVATE_MARKER", b"".join(hub.files.values()).decode())
        # Mutation after preparation cannot smuggle different bytes into the upload.
        (self.source / "hey_hex.onnx").write_bytes(b"replaced")
        self.assertNotEqual(hub.files["public-1/hey_hex.onnx"], b"replaced")

    def test_catalog_is_append_only_and_release_ids_immutable(self):
        old = {"schema": "waqy-catalog/1", "repo": REPO, "releases": [{"name": "old", "manifest_sha256": "old", "heads": []}]}
        original = json.loads(json.dumps(old))
        new = publish_hf.merged_catalog(old, publish_hf.catalog_entry(self.prepared()))
        self.assertEqual(old, original)
        self.assertEqual([r["name"] for r in new["releases"]], ["old", "public-1"])
        self.assertEqual(publish_hf.merged_catalog(new, publish_hf.catalog_entry(self.prepared())), new)
        self.manifest["threshold"] = 0.6
        self.save()
        with self.assertRaisesRegex(ValueError, "different manifest"):
            publish_hf.merged_catalog(new, publish_hf.catalog_entry(self.prepared()))

    def test_public_fields_are_validated_without_exporting_unknown_data(self):
        invalid = {"phrase": "internal/path", "kind": {"private": "data"}, "languages": ["not a language"],
                   "addressed": "false", "onset_gate": 1, "threshold": 1.1, "pos_test_median": float("nan"),
                   "owner_holdout_recall": 2, "false_wakes_per_hour": -1, "implies": "hex"}
        for key, value in invalid.items():
            with self.subTest(key=key):
                self.manifest["heads"]["hey_hex"][key] = value
                # Raw JSON intentionally allows NaN to test schema rejection, not the writer.
                (self.source / "manifest.json").write_text(json.dumps(self.manifest))
                with self.assertRaises(ValueError):
                    self.prepared()
                self.manifest["heads"]["hey_hex"].pop(key)
        self.save()
        with patch("publication.MAX_TOTAL_BYTES", 1), self.assertRaisesRegex(ValueError, "256 MiB"):
            self.prepared()
        with patch("publication.MAX_FILE_BYTES", 1), self.assertRaisesRegex(ValueError, "bounded regular"):
            self.prepared()

    def test_failed_local_export_is_removed_but_existing_release_is_never_deleted(self):
        publication = self.prepared()
        out = self.root / "out"
        with patch.object(Path, "write_bytes", side_effect=OSError("disk full")), self.assertRaises(OSError):
            write_release(publication, out)
        self.assertFalse((out / "public-1").exists())
        release = write_release(publication, out)
        with self.assertRaises(FileExistsError):
            write_release(publication, out)
        self.assertTrue((release / "manifest.json").exists())

    def test_offline_catalog_mismatches_and_apply_catalog_override_are_rejected(self):
        wrong = {"schema": "waqy-catalog/1", "repo": "another/repo", "releases": []}
        with self.assertRaisesRegex(ValueError, "different repository"):
            publish_hf.plan(self.prepared(), REPO, wrong)
        with self.assertRaisesRegex(ValueError, "catalog schema"):
            publish_hf.plan(self.prepared(), REPO, {"schema": "unknown", "releases": []})
        args = ["publish_hf.py", str(self.source), "--heads", "hey_hex", "--name", "public-1", "--apply", "--catalog", "unused.json"]
        with patch.object(sys, "argv", args), patch.object(publish_hf, "publish", side_effect=AssertionError("network")), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            publish_hf.main()
        self.assertEqual(error.exception.code, 2)

    def test_embedded_old_consent_cannot_authorize_a_new_owner_subset(self):
        self.manifest["heads"]["hex"]["owner_recordings"] = True
        self.manifest["curated_from"] = {"owner_consent": {"record": "broad-training-permission"}}
        self.save()
        with self.assertRaisesRegex(ValueError, "explicit publication consent"):
            self.prepared(["hex"])

    def test_conflicts_unknown_remote_history_and_private_targets_never_commit(self):
        publication = self.prepared()
        for hub in [FakeHub({"old/manifest.json": b"{}"}), FakeHub(private=True), FakeHub({"public-1/extra.onnx": b"x"})]:
            with self.subTest(files=list(hub.files), private=hub.private), self.assertRaises(ValueError):
                publish_hf.apply(publication, REPO, hub, hub.download, SimpleNamespace)
            self.assertEqual(hub.commits, [])
        valid = FakeHub()
        publish_hf.apply(publication, REPO, valid, valid.download, SimpleNamespace)
        valid.commits.clear()
        valid.files["public-1/hey_hex.onnx"] = b"different remote weights"
        with self.assertRaisesRegex(ValueError, "content differs"):
            publish_hf.apply(publication, REPO, valid, valid.download, SimpleNamespace)
        self.assertEqual(valid.commits, [])
        raced = FakeHub()
        raced.race = True
        with self.assertRaisesRegex(ValueError, "parent_commit"):
            publish_hf.apply(publication, REPO, raced, raced.download, SimpleNamespace)
        self.assertEqual(raced.files, {})


if __name__ == "__main__":
    unittest.main()
