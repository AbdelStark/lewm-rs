#!/usr/bin/env python3
"""Verify that a published GHCR runtime image is safe for paid HF Jobs."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_IMAGE_REPOSITORY = "ghcr.io/abdelstark/lewm-rs"
EXPECTED_SOURCE = "https://github.com/AbdelStark/lewm-rs"
IMAGE_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
PLACEHOLDER_RE = re.compile(r"(REPLACE_WITH_|<[^>]+>|\{[^}]+\})")
MANIFEST_ACCEPT = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)


class RuntimeImageError(RuntimeError):
    """Raised when an image tag cannot be trusted for a paid runtime."""


@dataclass(frozen=True)
class ImageRef:
    """A parsed tag-based OCI image reference."""

    registry: str
    repository: str
    tag: str

    @property
    def raw(self) -> str:
        """Return the original image reference form."""
        return f"{self.registry}/{self.repository}:{self.tag}"


@dataclass(frozen=True)
class RuntimeImageReport:
    """Verification evidence for a published runtime image."""

    image: str
    platform: str
    index_digest: str | None
    manifest_digest: str
    config_digest: str
    source: str
    revision: str
    version: str | None
    created: str | None
    expected_revision: str


class OciRegistryClient:
    """Small read-only OCI registry client for public GHCR artifacts."""

    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout
        self._tokens: dict[tuple[str, str], str] = {}

    def fetch_manifest(self, image: ImageRef, reference: str) -> tuple[dict[str, Any], str | None]:
        """Fetch an OCI manifest or index by tag/digest."""
        url = f"https://{image.registry}/v2/{image.repository}/manifests/{reference}"
        return self._fetch_json(image, url, {"Accept": MANIFEST_ACCEPT})

    def fetch_blob(self, image: ImageRef, digest: str) -> tuple[dict[str, Any], str | None]:
        """Fetch an OCI config blob by digest."""
        url = f"https://{image.registry}/v2/{image.repository}/blobs/{digest}"
        return self._fetch_json(image, url, {"Accept": "application/vnd.oci.image.config.v1+json"})

    def _fetch_json(
        self,
        image: ImageRef,
        url: str,
        headers: dict[str, str],
    ) -> tuple[dict[str, Any], str | None]:
        auth_headers = dict(headers)
        auth_headers["Authorization"] = f"Bearer {self._token(image)}"
        request = urllib.request.Request(url, headers=auth_headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
                digest = response.headers.get("Docker-Content-Digest")
        except urllib.error.HTTPError as error:
            raise RuntimeImageError(
                f"failed to fetch {url}: HTTP {error.code} {error.reason}"
            ) from error
        except urllib.error.URLError as error:
            raise RuntimeImageError(f"failed to fetch {url}: {error.reason}") from error
        if not isinstance(payload, dict):
            raise RuntimeImageError(f"{url}: expected JSON object")
        return payload, digest

    def _token(self, image: ImageRef) -> str:
        key = (image.registry, image.repository)
        if key in self._tokens:
            return self._tokens[key]

        scope = urllib.parse.quote(f"repository:{image.repository}:pull", safe=":")
        url = f"https://{image.registry}/token?service={image.registry}&scope={scope}"
        request = urllib.request.Request(url)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeImageError(
                f"failed to obtain registry token: HTTP {error.code} {error.reason}"
            ) from error
        except urllib.error.URLError as error:
            raise RuntimeImageError(f"failed to obtain registry token: {error.reason}") from error

        token = payload.get("token") or payload.get("access_token")
        if not isinstance(token, str) or not token:
            raise RuntimeImageError("registry token response did not include a bearer token")
        self._tokens[key] = token
        return token


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        default=None,
        help="full image reference, e.g. ghcr.io/abdelstark/lewm-rs:f1-runtime",
    )
    parser.add_argument(
        "--image-tag",
        default=None,
        help="tag under ghcr.io/abdelstark/lewm-rs to verify",
    )
    parser.add_argument(
        "--expected-revision",
        default=None,
        help="git commit SHA expected in org.opencontainers.image.revision; defaults to HEAD",
    )
    parser.add_argument(
        "--platform",
        default="linux/amd64",
        help="OCI platform to verify from a multi-arch index; default: linux/amd64",
    )
    parser.add_argument(
        "--allow-latest",
        action="store_true",
        help="allow verifying the mutable latest tag; production F1 should not use this",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable evidence")
    args = parser.parse_args()

    try:
        image = image_ref_from_args(args.image, args.image_tag, allow_latest=args.allow_latest)
        expected_revision = args.expected_revision or current_git_revision()
        report = verify_runtime_image(image, expected_revision, platform=args.platform)
    except RuntimeImageError as error:
        print(f"verify_runtime_image.py: {error}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(asdict(report), indent=2, sort_keys=True))
    else:
        print(
            "runtime image ok: "
            f"{report.image} {report.platform} {report.manifest_digest} "
            f"revision={report.revision}"
        )
    return 0


def image_ref_from_args(
    image: str | None,
    tag: str | None,
    *,
    allow_latest: bool = False,
) -> ImageRef:
    """Resolve CLI image arguments into one validated image reference."""
    if image and tag:
        raise RuntimeImageError("pass either --image or --image-tag, not both")
    if tag:
        validate_image_tag(tag, allow_latest=allow_latest)
        return parse_image_ref(f"{DEFAULT_IMAGE_REPOSITORY}:{tag}", allow_latest=allow_latest)
    if image:
        return parse_image_ref(image, allow_latest=allow_latest)
    raise RuntimeImageError("pass --image or --image-tag")


def validate_image_tag(tag: str, *, allow_latest: bool = False) -> None:
    """Reject mutable, placeholder, or syntactically invalid image tags."""
    if PLACEHOLDER_RE.search(tag):
        raise RuntimeImageError("image tag must be replaced with a real published tag")
    if tag == "latest" and not allow_latest:
        raise RuntimeImageError("refusing mutable image tag 'latest'")
    if not IMAGE_TAG_RE.fullmatch(tag):
        raise RuntimeImageError(f"invalid image tag: {tag!r}")


def parse_image_ref(raw: str, *, allow_latest: bool = False) -> ImageRef:
    """Parse a tag-based OCI image reference."""
    if "://" in raw:
        raise RuntimeImageError(f"invalid image reference: {raw!r}")
    if "@" in raw:
        raise RuntimeImageError("runtime image must be passed by tag, not digest")

    slash = raw.find("/")
    if slash <= 0:
        raise RuntimeImageError(f"image reference must include a registry: {raw!r}")
    last_colon = raw.rfind(":")
    last_slash = raw.rfind("/")
    if last_colon <= last_slash:
        raise RuntimeImageError(f"image reference must include an explicit tag: {raw!r}")

    registry = raw[:slash]
    repository = raw[slash + 1 : last_colon]
    tag = raw[last_colon + 1 :]
    validate_image_tag(tag, allow_latest=allow_latest)
    if raw[:last_colon] != DEFAULT_IMAGE_REPOSITORY:
        raise RuntimeImageError(
            f"runtime image must use {DEFAULT_IMAGE_REPOSITORY}, got {raw[:last_colon]}"
        )
    return ImageRef(registry=registry, repository=repository, tag=tag)


def current_git_revision() -> str:
    """Return the current git commit SHA for the workspace."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeImageError(result.stderr.strip() or "failed to resolve git HEAD")
    revision = result.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeImageError(f"git HEAD is not a full commit SHA: {revision!r}")
    return revision


def verify_runtime_image(
    image: ImageRef,
    expected_revision: str,
    *,
    platform: str = "linux/amd64",
    client: OciRegistryClient | None = None,
) -> RuntimeImageReport:
    """Verify source and revision labels for a published runtime image."""
    if not re.fullmatch(r"[0-9a-f]{40}", expected_revision):
        raise RuntimeImageError(
            f"expected revision must be a full 40-character git SHA, got {expected_revision!r}"
        )

    client = client or OciRegistryClient()
    top_manifest, top_digest = client.fetch_manifest(image, image.tag)
    manifest, manifest_digest, index_digest = select_platform_manifest(
        image,
        top_manifest,
        top_digest,
        platform=platform,
        client=client,
    )
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise RuntimeImageError(f"{image.raw}: manifest missing config descriptor")
    config_digest = config.get("digest")
    if not isinstance(config_digest, str) or not config_digest.startswith("sha256:"):
        raise RuntimeImageError(f"{image.raw}: manifest has invalid config digest")

    config_json, _digest = client.fetch_blob(image, config_digest)
    labels = config_json.get("config", {}).get("Labels", {})
    if not isinstance(labels, dict):
        raise RuntimeImageError(f"{image.raw}: image config labels are missing")

    source = require_label(labels, "org.opencontainers.image.source", image)
    if source != EXPECTED_SOURCE:
        raise RuntimeImageError(f"{image.raw}: source label is {source!r}, expected {EXPECTED_SOURCE!r}")
    revision = require_label(labels, "org.opencontainers.image.revision", image)
    if revision != expected_revision:
        raise RuntimeImageError(
            f"{image.raw}: revision label {revision!r} does not match expected "
            f"{expected_revision!r}"
        )

    version = labels.get("org.opencontainers.image.version")
    created = labels.get("org.opencontainers.image.created")
    return RuntimeImageReport(
        image=image.raw,
        platform=platform,
        index_digest=index_digest,
        manifest_digest=manifest_digest,
        config_digest=config_digest,
        source=source,
        revision=revision,
        version=version if isinstance(version, str) else None,
        created=created if isinstance(created, str) else None,
        expected_revision=expected_revision,
    )


def select_platform_manifest(
    image: ImageRef,
    manifest: dict[str, Any],
    digest: str | None,
    *,
    platform: str,
    client: OciRegistryClient,
) -> tuple[dict[str, Any], str, str | None]:
    """Select the requested platform from an OCI index, or return a single manifest."""
    media_type = manifest.get("mediaType")
    if media_type in {
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    }:
        if not digest:
            raise RuntimeImageError(f"{image.raw}: registry did not return a manifest digest")
        return manifest, digest, None

    manifests = manifest.get("manifests")
    if not isinstance(manifests, list):
        raise RuntimeImageError(f"{image.raw}: expected an OCI image index")

    os_name, arch = parse_platform(platform)
    for descriptor in manifests:
        if not isinstance(descriptor, dict):
            continue
        descriptor_platform = descriptor.get("platform")
        if not isinstance(descriptor_platform, dict):
            continue
        if descriptor_platform.get("os") != os_name or descriptor_platform.get("architecture") != arch:
            continue
        descriptor_digest = descriptor.get("digest")
        if not isinstance(descriptor_digest, str):
            raise RuntimeImageError(f"{image.raw}: platform descriptor missing digest")
        selected, selected_digest = client.fetch_manifest(image, descriptor_digest)
        if not selected_digest:
            selected_digest = descriptor_digest
        return selected, selected_digest, digest

    raise RuntimeImageError(f"{image.raw}: no manifest for platform {platform!r}")


def parse_platform(platform: str) -> tuple[str, str]:
    """Parse `os/arch` platform strings."""
    parts = platform.split("/")
    if len(parts) != 2 or not all(parts):
        raise RuntimeImageError(f"platform must use os/arch form, got {platform!r}")
    return parts[0], parts[1]


def require_label(labels: dict[str, Any], key: str, image: ImageRef) -> str:
    """Return a required string OCI label."""
    value = labels.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeImageError(f"{image.raw}: missing required label {key!r}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
