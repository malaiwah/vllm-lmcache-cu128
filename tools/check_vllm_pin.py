#!/usr/bin/env python3
"""Check the pinned vLLM commit in the Containerfile against upstream."""
from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys
from typing import Iterable

DEFAULT_CONTAINERFILE = pathlib.Path(__file__).resolve().parents[1] / "Containerfile"
DEFAULT_REMOTE = "https://github.com/vllm-project/vllm.git"
DEFAULT_BRANCH = "main"

_PIN_PATTERN = re.compile(r"^(?:ENV|ARG)\s+VLLM_COMMIT=([0-9a-fA-F]{7,40})\s*$")


class PinCheckError(RuntimeError):
    """Raised when the pin cannot be inspected."""


def read_pinned_commit(containerfile: pathlib.Path) -> str:
    try:
        text = containerfile.read_text(encoding="utf-8")
    except FileNotFoundError as exc:  # pragma: no cover - CLI guard
        raise PinCheckError(f"Missing Containerfile: {containerfile}") from exc

    for line in text.splitlines():
        match = _PIN_PATTERN.match(line.strip())
        if match:
            return match.group(1)

    raise PinCheckError(
        f"Unable to locate VLLM_COMMIT line in {containerfile}."
    )


def get_remote_commit(remote: str, branch: str) -> str:
    try:
        result = subprocess.run(
            ["git", "ls-remote", remote, branch],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:  # pragma: no cover - CLI guard
        raise PinCheckError(
            f"Failed to query {remote} {branch}: {exc.stderr.strip()}"
        ) from exc

    stdout = result.stdout.strip()
    if not stdout:
        raise PinCheckError(
            f"git ls-remote returned no data for {remote} {branch}."
        )

    first_line = stdout.splitlines()[0]
    commit, *_ = first_line.split()
    if not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
        raise PinCheckError(
            f"Unexpected git ls-remote output: {first_line!r}"
        )
    return commit.lower()


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--containerfile",
        type=pathlib.Path,
        default=DEFAULT_CONTAINERFILE,
        help="Path to the Containerfile (default: %(default)s)",
    )
    parser.add_argument(
        "--remote",
        default=DEFAULT_REMOTE,
        help="Git remote to query (default: %(default)s)",
    )
    parser.add_argument(
        "--branch",
        default=DEFAULT_BRANCH,
        help="Remote branch or ref to compare against (default: %(default)s)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress descriptive output; only exit status indicates drift.",
    )
    args = parser.parse_args(argv)

    try:
        pinned = read_pinned_commit(args.containerfile)
        upstream = get_remote_commit(args.remote, args.branch)
    except PinCheckError as exc:  # pragma: no cover - CLI guard
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    drift = pinned.lower() != upstream

    if not args.quiet:
        print(f"Pinned commit   : {pinned}")
        print(f"Upstream {args.branch:>8}: {upstream}")
        if drift:
            print("Status          : drift detected (update required)")
        else:
            print("Status          : up to date")

    return int(drift)


if __name__ == "__main__":  # pragma: no branch
    raise SystemExit(main())
