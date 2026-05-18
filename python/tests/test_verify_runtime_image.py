"""Tests for the GHCR runtime image verifier."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import verify_runtime_image  # noqa: E402

GOOD_REVISION = "a" * 40


class FakeClient:
    """In-memory OCI registry client used by verifier unit tests."""

    index_digest = "sha256:index"
    manifest_digest = "sha256:manifest"
    config_digest = "sha256:config"

    def __init__(
        self,
        *,
        labels: dict[str, Any] | None = None,
        include_platform: bool = True,
    ) -> None:
        self.labels = labels or {
            "org.opencontainers.image.source": verify_runtime_image.EXPECTED_SOURCE,
            "org.opencontainers.image.revision": GOOD_REVISION,
            "org.opencontainers.image.version": "f1-runtime-test",
            "org.opencontainers.image.created": "2026-05-18T00:00:00Z",
        }
        self.include_platform = include_platform

    def fetch_manifest(
        self,
        _image: verify_runtime_image.ImageRef,
        reference: str,
    ) -> tuple[dict[str, Any], str | None]:
        if reference == "f1-runtime-test":
            manifests: list[dict[str, Any]] = []
            if self.include_platform:
                manifests.append(
                    {
                        "digest": self.manifest_digest,
                        "platform": {"os": "linux", "architecture": "amd64"},
                    }
                )
            manifests.append(
                {
                    "digest": "sha256:attestation",
                    "platform": {"os": "unknown", "architecture": "unknown"},
                }
            )
            return (
                {
                    "mediaType": "application/vnd.oci.image.index.v1+json",
                    "manifests": manifests,
                },
                self.index_digest,
            )
        if reference == self.manifest_digest:
            return (
                {
                    "mediaType": "application/vnd.oci.image.manifest.v1+json",
                    "config": {"digest": self.config_digest},
                },
                self.manifest_digest,
            )
        raise AssertionError(f"unexpected manifest reference {reference!r}")

    def fetch_blob(
        self,
        _image: verify_runtime_image.ImageRef,
        digest: str,
    ) -> tuple[dict[str, Any], str | None]:
        assert digest == self.config_digest
        return {"config": {"Labels": self.labels}}, None


def test_image_ref_from_tag_uses_default_repository() -> None:
    image = verify_runtime_image.image_ref_from_args(None, "f1-runtime-test")

    assert image.raw == "ghcr.io/abdelstark/lewm-rs:f1-runtime-test"


def test_rejects_placeholder_tag() -> None:
    with pytest.raises(verify_runtime_image.RuntimeImageError, match="real published tag"):
        verify_runtime_image.image_ref_from_args(None, "REPLACE_WITH_RUNTIME_IMAGE_TAG")


def test_rejects_latest_by_default() -> None:
    with pytest.raises(verify_runtime_image.RuntimeImageError, match="latest"):
        verify_runtime_image.image_ref_from_args(None, "latest")


def test_rejects_wrong_repository() -> None:
    with pytest.raises(verify_runtime_image.RuntimeImageError, match="must use"):
        verify_runtime_image.parse_image_ref("ghcr.io/other/lewm-rs:f1-runtime-test")


def test_verify_runtime_image_accepts_matching_revision() -> None:
    image = verify_runtime_image.image_ref_from_args(None, "f1-runtime-test")

    report = verify_runtime_image.verify_runtime_image(
        image,
        GOOD_REVISION,
        client=FakeClient(),
    )

    assert report.image == image.raw
    assert report.index_digest == FakeClient.index_digest
    assert report.manifest_digest == FakeClient.manifest_digest
    assert report.config_digest == FakeClient.config_digest
    assert report.revision == GOOD_REVISION
    assert report.version == "f1-runtime-test"


def test_verify_runtime_image_rejects_revision_mismatch() -> None:
    image = verify_runtime_image.image_ref_from_args(None, "f1-runtime-test")

    with pytest.raises(verify_runtime_image.RuntimeImageError, match="does not match"):
        verify_runtime_image.verify_runtime_image(
            image,
            "b" * 40,
            client=FakeClient(),
        )


def test_verify_runtime_image_rejects_missing_revision_label() -> None:
    image = verify_runtime_image.image_ref_from_args(None, "f1-runtime-test")
    labels = {"org.opencontainers.image.source": verify_runtime_image.EXPECTED_SOURCE}

    with pytest.raises(verify_runtime_image.RuntimeImageError, match="revision"):
        verify_runtime_image.verify_runtime_image(
            image,
            GOOD_REVISION,
            client=FakeClient(labels=labels),
        )


def test_verify_runtime_image_rejects_wrong_source_label() -> None:
    image = verify_runtime_image.image_ref_from_args(None, "f1-runtime-test")
    labels = {
        "org.opencontainers.image.source": "https://github.com/example/other",
        "org.opencontainers.image.revision": GOOD_REVISION,
    }

    with pytest.raises(verify_runtime_image.RuntimeImageError, match="source label"):
        verify_runtime_image.verify_runtime_image(
            image,
            GOOD_REVISION,
            client=FakeClient(labels=labels),
        )


def test_verify_runtime_image_rejects_missing_platform_manifest() -> None:
    image = verify_runtime_image.image_ref_from_args(None, "f1-runtime-test")

    with pytest.raises(verify_runtime_image.RuntimeImageError, match="no manifest"):
        verify_runtime_image.verify_runtime_image(
            image,
            GOOD_REVISION,
            client=FakeClient(include_platform=False),
        )
