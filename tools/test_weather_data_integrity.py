#!/usr/bin/env python3
"""Negative test for WeatherAI data-bundle checksum validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


MODEL_FILE = "model-calibration.json"
HEALTH_FILE = "health.json"
MANIFEST_FILE = "manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify checksum validation rejects a tampered WeatherAI bundle.")
    parser.add_argument("--bundle-dir", default="public/weatherai-data")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle_dir = Path(args.bundle_dir)
    base_result = run_validator(bundle_dir)
    if base_result.returncode != 0:
        print(base_result.stderr or base_result.stdout, file=sys.stderr)
        print("Base bundle must pass before running tamper test.", file=sys.stderr)
        return base_result.returncode

    with tempfile.TemporaryDirectory(prefix="weatherai-integrity-test-") as directory:
        checksum_dir = Path(directory) / "checksum-tamper"
        shutil.copytree(bundle_dir, checksum_dir)
        with (checksum_dir / MODEL_FILE).open("ab") as handle:
            handle.write(b"\n")

        checksum_result = run_validator(checksum_dir)
        checksum_output = checksum_result.stderr + checksum_result.stdout
        if checksum_result.returncode == 0:
            print("Checksum-tampered bundle unexpectedly passed validation.", file=sys.stderr)
            return 1
        if "sha256" not in checksum_output:
            print(checksum_output, file=sys.stderr)
            print("Tampered bundle failed, but not because of checksum validation.", file=sys.stderr)
            return 1

        freshness_dir = Path(directory) / "freshness-tamper"
        shutil.copytree(bundle_dir, freshness_dir)
        tamper_health_freshness(freshness_dir)
        freshness_result = run_validator(freshness_dir)
        freshness_output = freshness_result.stderr + freshness_result.stdout
        if freshness_result.returncode == 0:
            print("Freshness-tampered bundle unexpectedly passed validation.", file=sys.stderr)
            return 1
        if "freshness" not in freshness_output:
            print(freshness_output, file=sys.stderr)
            print("Tampered bundle failed, but not because of freshness validation.", file=sys.stderr)
            return 1

        performance_dir = Path(directory) / "performance-tamper"
        shutil.copytree(bundle_dir, performance_dir)
        tamper_model_performance(performance_dir)
        performance_result = run_validator(performance_dir)
        performance_output = performance_result.stderr + performance_result.stdout
        if performance_result.returncode == 0:
            print("Model-performance-tampered bundle unexpectedly passed validation.", file=sys.stderr)
            return 1
        if "modelPerformanceSummary" not in performance_output:
            print(performance_output, file=sys.stderr)
            print("Tampered bundle failed, but not because of model performance validation.", file=sys.stderr)
            return 1

        rain_bias_dir = Path(directory) / "rain-bias-tamper"
        shutil.copytree(bundle_dir, rain_bias_dir)
        tamper_rain_bias_performance(rain_bias_dir)
        rain_bias_result = run_validator(rain_bias_dir)
        rain_bias_output = rain_bias_result.stderr + rain_bias_result.stdout
        if rain_bias_result.returncode == 0:
            print("Rain-bias-tampered bundle unexpectedly passed validation.", file=sys.stderr)
            return 1
        if "averageRainBiasCorrection" not in rain_bias_output:
            print(rain_bias_output, file=sys.stderr)
            print("Tampered bundle failed, but not because of rain-bias validation.", file=sys.stderr)
            return 1

        quality_rain_bias_dir = Path(directory) / "quality-rain-bias-tamper"
        shutil.copytree(bundle_dir, quality_rain_bias_dir)
        tamper_manifest_rain_bias_quality(quality_rain_bias_dir)
        quality_rain_bias_result = run_validator(quality_rain_bias_dir)
        quality_rain_bias_output = quality_rain_bias_result.stderr + quality_rain_bias_result.stdout
        if quality_rain_bias_result.returncode == 0:
            print("Manifest-rain-bias-tampered bundle unexpectedly passed validation.", file=sys.stderr)
            return 1
        if "maxAbsoluteRainBiasCorrection" not in quality_rain_bias_output:
            print(quality_rain_bias_output, file=sys.stderr)
            print("Tampered bundle failed, but not because of manifest rain-bias quality validation.", file=sys.stderr)
            return 1

        climate_status_dir = Path(directory) / "climate-status-tamper"
        shutil.copytree(bundle_dir, climate_status_dir)
        tamper_climate_source_status(climate_status_dir)
        climate_status_result = run_validator(climate_status_dir)
        climate_status_output = climate_status_result.stderr + climate_status_result.stdout
        if climate_status_result.returncode == 0:
            print("Climate-source-status-tampered bundle unexpectedly passed validation.", file=sys.stderr)
            return 1
        if "sourceStatus" not in climate_status_output:
            print(climate_status_output, file=sys.stderr)
            print("Tampered bundle failed, but not because of climate source status validation.", file=sys.stderr)
            return 1

    print("WeatherAI integrity tamper test passed: checksum, freshness, model-performance, rain-bias, manifest-rain-bias, and climate-source-status tampering were rejected.")
    return 0


def run_validator(bundle_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "tools/validate_weather_data_bundle.py",
            "--bundle-dir",
            str(bundle_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def tamper_health_freshness(bundle_dir: Path) -> None:
    health_path = bundle_dir / HEALTH_FILE
    manifest_path = bundle_dir / MANIFEST_FILE
    health = json.loads(health_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    health["freshness"]["staleAfter"] = health["generatedAt"]
    health_bytes = (json.dumps(health, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    health_path.write_bytes(health_bytes)
    manifest["files"]["health"]["sha256"] = hashlib.sha256(health_bytes).hexdigest()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def tamper_model_performance(bundle_dir: Path) -> None:
    health_path = bundle_dir / HEALTH_FILE
    manifest_path = bundle_dir / MANIFEST_FILE
    health = json.loads(health_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    health["modelPerformanceSummary"]["models"][0]["averageCombinedError"] = 0.0
    health_bytes = (json.dumps(health, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    health_path.write_bytes(health_bytes)
    manifest["files"]["health"]["sha256"] = hashlib.sha256(health_bytes).hexdigest()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def tamper_rain_bias_performance(bundle_dir: Path) -> None:
    health_path = bundle_dir / HEALTH_FILE
    manifest_path = bundle_dir / MANIFEST_FILE
    health = json.loads(health_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    health["modelPerformanceSummary"]["models"][0]["averageRainBiasCorrection"] = 0.0
    health_bytes = (json.dumps(health, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    health_path.write_bytes(health_bytes)
    manifest["files"]["health"]["sha256"] = hashlib.sha256(health_bytes).hexdigest()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def tamper_manifest_rain_bias_quality(bundle_dir: Path) -> None:
    manifest_path = bundle_dir / MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    manifest["qualitySummary"]["maxAbsoluteRainBiasCorrection"] = 0.0
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def tamper_climate_source_status(bundle_dir: Path) -> None:
    health_path = bundle_dir / HEALTH_FILE
    manifest_path = bundle_dir / MANIFEST_FILE
    health = json.loads(health_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    health["climateSignal"]["sourceStatus"][0]["status"] = "failed"
    health["climateSignal"]["sourceStatus"][0]["error"] = "tampered status"
    health_bytes = (json.dumps(health, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    health_path.write_bytes(health_bytes)
    manifest["files"]["health"]["sha256"] = hashlib.sha256(health_bytes).hexdigest()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
