#!/usr/bin/env python3
"""Inspect CUDA Docker image manifest digests for the linux/amd64 variant."""
from __future__ import annotations

import argparse
import sys
from typing import Iterable

import requests

REGISTRY_AUTH = "https://auth.docker.io/token"
REGISTRY_BASE = "https://registry-1.docker.io/v2"
DEFAULT_REPOSITORY = "nvidia/cuda"
DEFAULT_TAGS = (
    "12.8.1-cudnn-devel-ubuntu24.04",
    "12.8.1-cudnn-runtime-ubuntu24.04",
)


def fetch_token(repository: str) -> str:
    response = requests.get(
        REGISTRY_AUTH,
        params={"service": "registry.docker.io", "scope": f"repository:{repository}:pull"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["token"]


def fetch_manifest_digest(repository: str, tag: str, *, architecture: str = "amd64", os: str = "linux") -> str:
    token = fetch_token(repository)
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.docker.distribution.manifest.list.v2+json",
    }
    response = requests.get(f"{REGISTRY_BASE}/{repository}/manifests/{tag}", headers=headers, timeout=30)
    response.raise_for_status()
    manifest = response.json()
    manifests = manifest.get("manifests")

    if not manifests:
        config = manifest.get("config")
        if config and "digest" in config:
            return config["digest"]
        raise RuntimeError(f"No manifest list entries found for {tag!r}")

    for entry in manifests:
        platform = entry.get("platform", {})
        if platform.get("architecture") == architecture and platform.get("os") == os:
            digest = entry.get("digest")
            if digest:
                return digest
            raise RuntimeError(f"Manifest entry for {architecture}/{os} lacks a digest: {entry}")

    raise RuntimeError(
        f"Unable to locate {architecture}/{os} manifest entry for {repository}:{tag}."
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tags", nargs="*", default=DEFAULT_TAGS, help="Image tags to inspect")
    parser.add_argument(
        "--repository",
        default=DEFAULT_REPOSITORY,
        help="Docker repository (default: %(default)s)",
    )
    parser.add_argument("--architecture", default="amd64", help="Target architecture (default: %(default)s)")
    parser.add_argument("--os", default="linux", help="Target OS (default: %(default)s)")
    args = parser.parse_args(argv)

    try:
        for tag in args.tags:
            digest = fetch_manifest_digest(
                args.repository,
                tag,
                architecture=args.architecture,
                os=args.os,
            )
            print(f"{tag} {digest}")
    except Exception as exc:  # pragma: no cover - CLI utility
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no branch
    raise SystemExit(main())
