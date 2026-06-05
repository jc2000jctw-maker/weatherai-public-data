#!/usr/bin/env python3
"""Preflight WeatherAI release remote-data configuration."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = "Config/WeatherAI-Release.xcconfig"
DEFAULT_PROJECT = "WeatherAI.xcodeproj"
DEFAULT_SCHEME = "WeatherAI"
DEFAULT_CONFIGURATION = "Release"
MODEL_FILE = "model-calibration.json"
CLIMATE_FILE = "climate-signal.json"
MANIFEST_FILE = "manifest.json"
HEALTH_FILE = "health.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify Release xcconfig and hosted WeatherAI data manifest before App Store builds."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--scheme", default=DEFAULT_SCHEME)
    parser.add_argument("--configuration", default=DEFAULT_CONFIGURATION)
    parser.add_argument("--manifest-url", help="Override WEATHER_DATA_MANIFEST_URL for this preflight.")
    parser.add_argument("--allow-empty", action="store_true", help="Allow an empty manifest URL for local checks.")
    parser.add_argument(
        "--allow-direct-overrides",
        action="store_true",
        help="Allow direct model/climate URLs in Release config. App Store builds should normally use only the manifest URL.",
    )
    parser.add_argument("--skip-fetch", action="store_true", help="Only validate config URL syntax; do not fetch JSON.")
    parser.add_argument("--allow-http", action="store_true", help="Allow http:// manifest URLs for non-App-Store checks.")
    parser.add_argument(
        "--skip-xcode-build-settings",
        action="store_true",
        help="Skip verifying that the Release target build settings include WEATHER_DATA_MANIFEST_URL.",
    )
    parser.add_argument("--min-entry-samples", type=int, default=48)
    parser.add_argument("--min-calibration-days", type=int, default=14)
    parser.add_argument("--min-sample-confidence", type=float, default=0.5)
    parser.add_argument("--min-climate-source-successes", type=int, default=2)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--retry-count", type=int, default=3)
    parser.add_argument("--retry-backoff-seconds", type=float, default=3.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = parse_xcconfig(Path(args.config))
    if not args.allow_direct_overrides:
        direct_overrides = configured_direct_overrides(config)
        if direct_overrides:
            print(
                "Release config must leave direct data URL overrides empty so the hosted manifest, "
                f"hashes, health, and quality gates remain the single source of truth: {', '.join(direct_overrides)}",
                file=sys.stderr,
            )
            return 1

    raw_manifest_url = args.manifest_url or config.get("WEATHER_DATA_MANIFEST_URL", "")
    manifest_url = normalize_xcconfig_url(raw_manifest_url)

    if not manifest_url:
        if args.allow_empty:
            print(f"{args.config}: WEATHER_DATA_MANIFEST_URL is empty; local fallback mode is allowed.")
            return 0
        print(
            f"{args.config}: WEATHER_DATA_MANIFEST_URL is empty. "
            "Set it before Release/App Store builds.",
            file=sys.stderr,
        )
        return 1

    parsed = urllib.parse.urlparse(manifest_url)
    if parsed.scheme not in {"https", "http", "file"}:
        print("WEATHER_DATA_MANIFEST_URL must be https://, http://, or file:// for local preflight.", file=sys.stderr)
        return 1
    if parsed.scheme == "http" and not args.allow_http:
        print("WEATHER_DATA_MANIFEST_URL must use HTTPS for Release/App Store builds. Pass --allow-http only for non-App-Store checks.", file=sys.stderr)
        return 1

    print(f"Manifest URL: {manifest_url}")
    if not args.manifest_url and not args.skip_xcode_build_settings:
        verify_xcode_build_settings(args, manifest_url)

    if args.skip_fetch:
        return 0

    with tempfile.TemporaryDirectory(prefix="weatherai-release-preflight-") as directory:
        bundle_dir = Path(directory)
        try:
            manifest = fetch_json(manifest_url, args)
            write_json(bundle_dir / MANIFEST_FILE, manifest)
            fetch_manifest_file(manifest_url, manifest, "modelCalibration", MODEL_FILE, bundle_dir, args)
            fetch_manifest_file(manifest_url, manifest, "climateSignal", CLIMATE_FILE, bundle_dir, args)
            fetch_manifest_file(manifest_url, manifest, "health", HEALTH_FILE, bundle_dir, args)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to fetch hosted WeatherAI data bundle: {exc}", file=sys.stderr)
            return 1

        command = [
            sys.executable,
            "tools/validate_weather_data_bundle.py",
            "--bundle-dir",
            str(bundle_dir),
            "--min-entry-samples",
            str(args.min_entry_samples),
            "--min-calibration-days",
            str(args.min_calibration_days),
            "--min-sample-confidence",
            str(args.min_sample_confidence),
            "--min-climate-source-successes",
            str(args.min_climate_source_successes),
        ]
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            return result.returncode

    print("Release remote-data preflight passed.")
    return 0


def parse_xcconfig(path: Path) -> dict[str, str]:
    settings: dict[str, str] = {}
    if not path.exists():
        raise SystemExit(f"Missing xcconfig: {path}")

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        settings[key.strip()] = value.strip()
    return settings


def normalize_xcconfig_url(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        return ""
    return cleaned.replace(":/$()/", "://")


def configured_direct_overrides(config: dict[str, str]) -> list[str]:
    direct_keys = ["WEATHER_MODEL_CALIBRATION_URL", "WEATHER_CLIMATE_SIGNAL_URL"]
    configured = []
    for key in direct_keys:
        value = normalize_xcconfig_url(config.get(key, ""))
        if value and not value.startswith("$("):
            configured.append(key)
    return configured


def verify_xcode_build_settings(args: argparse.Namespace, expected_manifest_url: str) -> None:
    command = [
        "xcodebuild",
        "-project",
        args.project,
        "-scheme",
        args.scheme,
        "-configuration",
        args.configuration,
        "-showBuildSettings",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(result.stderr.strip() or "Failed to read Xcode build settings.")

    settings = parse_xcode_build_settings(result.stdout)
    raw_value = settings.get("WEATHER_DATA_MANIFEST_URL", "")
    actual_manifest_url = normalize_xcconfig_url(raw_value)
    if actual_manifest_url != expected_manifest_url:
        raise SystemExit(
            "Xcode Release build setting WEATHER_DATA_MANIFEST_URL does not match "
            f"{args.config}. Expected {expected_manifest_url!r}, got {actual_manifest_url!r}."
        )


def parse_xcode_build_settings(output: str) -> dict[str, str]:
    settings: dict[str, str] = {}
    for line in output.splitlines():
        stripped = line.strip()
        if " = " not in stripped:
            continue
        key, value = stripped.split(" = ", 1)
        settings[key.strip()] = value.strip()
    return settings


def fetch_manifest_file(
    manifest_url: str,
    manifest: dict[str, Any],
    file_key: str,
    output_name: str,
    bundle_dir: Path,
    args: argparse.Namespace,
) -> None:
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("manifest.files must be an object")
    file_info = files.get(file_key)
    if not isinstance(file_info, dict):
        raise ValueError(f"manifest.files.{file_key} must be an object")
    relative_path = file_info.get("path")
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(f"manifest.files.{file_key}.path must be non-empty")
    if relative_path.startswith("/") or ".." in Path(relative_path).parts:
        raise ValueError(f"manifest.files.{file_key}.path must be a safe relative path")

    source_url = urllib.parse.urljoin(manifest_url, relative_path)
    payload = fetch_bytes(source_url, args)
    (bundle_dir / output_name).write_bytes(payload)


def fetch_json(url: str, args: argparse.Namespace) -> dict[str, Any]:
    payload = fetch_bytes(url, args)
    parsed = json.loads(payload.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"{url} must contain a JSON object")
    return parsed


def fetch_bytes(url: str, args: argparse.Namespace) -> bytes:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "file":
        return Path(urllib.request.url2pathname(parsed.path)).read_bytes()

    request = urllib.request.Request(url, headers={"User-Agent": "WeatherAI-release-preflight/1.0"})
    last_error: Exception | None = None
    for attempt in range(1, max(1, args.retry_count) + 1):
        try:
            with urllib.request.urlopen(request, timeout=args.request_timeout) as response:
                if response.status < 200 or response.status >= 300:
                    raise RuntimeError(f"HTTP {response.status} for {url}")
                return response.read()
        except (TimeoutError, urllib.error.URLError, RuntimeError) as exc:
            last_error = exc
            if attempt >= max(1, args.retry_count):
                break
            delay = max(0, args.retry_backoff_seconds) * (2 ** (attempt - 1))
            print(
                f"Retrying {url} after {type(exc).__name__}: attempt {attempt + 1}/{args.retry_count}",
                file=sys.stderr,
            )
            time.sleep(delay)

    raise RuntimeError(f"Failed to fetch {url}: {last_error}") from last_error


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
