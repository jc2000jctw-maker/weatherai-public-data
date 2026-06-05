#!/usr/bin/env python3
"""Audit WeatherAI release readiness for live public weather data."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import urllib.parse
from pathlib import Path
from typing import Any


DEFAULT_BUNDLE_DIR = "public/weatherai-data"
DEFAULT_CONFIG = "Config/WeatherAI-Release.xcconfig"
DEFAULT_WORKFLOW = ".github/workflows/weatherai-data-refresh.yml"
MIN_TRAINING_DAYS = 14
MIN_ENTRY_SAMPLES = 48
MIN_CLIMATE_SOURCE_SUCCESSES = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report whether WeatherAI is ready to ship with live hosted forecast data."
    )
    parser.add_argument("--bundle-dir", default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--workflow", default=DEFAULT_WORKFLOW)
    parser.add_argument("--skip-release-preflight", action="store_true")
    parser.add_argument(
        "--allow-empty-release-url",
        action="store_true",
        help="Treat an empty Release manifest URL as a warning for data-refresh reports.",
    )
    parser.add_argument("--report-json", help="Write a machine-readable readiness report.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results: list[tuple[str, str, str]] = []

    bundle_dir = Path(args.bundle_dir)
    config_path = Path(args.config)
    workflow_path = Path(args.workflow)

    validate_local_bundle(bundle_dir, results)
    inspect_quality_summary(bundle_dir, results)
    inspect_climate_source_coverage(bundle_dir, results)
    inspect_expected_app_data_mode(bundle_dir, results)
    inspect_workflow(workflow_path, results)
    manifest_url = inspect_release_config(config_path, results, allow_empty=args.allow_empty_release_url)
    if not args.skip_release_preflight:
        run_release_preflight(manifest_url, results, allow_empty=args.allow_empty_release_url)

    for status, title, detail in results:
        print(f"[{status}] {title}: {detail}")

    report_status = status_for_results(results)
    if args.report_json:
        write_report_json(
            Path(args.report_json),
            status=report_status,
            manifest_url=manifest_url,
            bundle_dir=bundle_dir,
            results=results,
        )

    if report_status == "blocked":
        print("WeatherAI release readiness: blocked.")
        return 1

    if report_status == "warning":
        print("WeatherAI release readiness: warnings.")
    else:
        print("WeatherAI release readiness: passed.")
    return 0


def status_for_results(results: list[tuple[str, str, str]]) -> str:
    if any(status == "FAIL" for status, _, _ in results):
        return "blocked"
    if any(status == "WARN" for status, _, _ in results):
        return "warning"
    return "passed"


def write_report_json(
    path: Path,
    status: str,
    manifest_url: str,
    bundle_dir: Path,
    results: list[tuple[str, str, str]],
) -> None:
    quality_summary: dict[str, Any] = {}
    health_summary: dict[str, Any] = {}
    try:
        manifest = load_json(bundle_dir / "manifest.json")
        summary = manifest.get("qualitySummary")
        if isinstance(summary, dict):
            quality_summary = summary
    except Exception:  # noqa: BLE001
        quality_summary = {}

    try:
        health = load_json(bundle_dir / "health.json")
        health_summary = {
            "generatedAt": health.get("generatedAt"),
            "modelCalibration": health.get("modelCalibration"),
            "climateSignal": health.get("climateSignal"),
            "freshness": health.get("freshness"),
            "modelPerformanceSummary": health.get("modelPerformanceSummary"),
        }
    except Exception:  # noqa: BLE001
        health_summary = {}

    report = {
        "status": status,
        "checkedAt": utc_now(),
        "releaseManifestURL": manifest_url,
        "checks": [
            {
                "status": result_status,
                "title": title,
                "detail": detail,
            }
            for result_status, title, detail in results
        ],
        "qualitySummary": quality_summary,
        "healthSummary": health_summary,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def validate_local_bundle(bundle_dir: Path, results: list[tuple[str, str, str]]) -> None:
    command = [
        sys.executable,
        "tools/validate_weather_data_bundle.py",
        "--bundle-dir",
        str(bundle_dir),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode == 0:
        results.append(("OK", "local production bundle", result.stdout.strip()))
    else:
        detail = (result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}")
        results.append(("FAIL", "local production bundle", detail))


def inspect_quality_summary(bundle_dir: Path, results: list[tuple[str, str, str]]) -> None:
    try:
        manifest = load_json(bundle_dir / "manifest.json")
    except Exception as exc:  # noqa: BLE001
        results.append(("FAIL", "manifest quality summary", str(exc)))
        return

    summary = manifest.get("qualitySummary")
    if not isinstance(summary, dict):
        results.append(("FAIL", "manifest quality summary", "missing qualitySummary"))
        return

    failures: list[str] = []
    if summary.get("validationProfile") != "deployment-default":
        failures.append("validationProfile is not deployment-default")
    if as_int(summary.get("trainingWindowDays")) < MIN_TRAINING_DAYS:
        failures.append(f"trainingWindowDays below {MIN_TRAINING_DAYS}")
    if as_int(summary.get("minSampleCount")) < MIN_ENTRY_SAMPLES:
        failures.append(f"minSampleCount below {MIN_ENTRY_SAMPLES}")
    if as_int(summary.get("climateSourceSuccessCount")) < MIN_CLIMATE_SOURCE_SUCCESSES:
        failures.append(f"climateSourceSuccessCount below {MIN_CLIMATE_SOURCE_SUCCESSES}")

    detail = (
        f"trainingWindowDays={summary.get('trainingWindowDays')}, "
        f"minSampleCount={summary.get('minSampleCount')}, "
        f"minSampleConfidence={summary.get('minSampleConfidence')}, "
        f"maxAbsoluteRainBiasCorrection={summary.get('maxAbsoluteRainBiasCorrection')}, "
        f"climateSourceSuccessCount={summary.get('climateSourceSuccessCount')}"
    )
    if failures:
        results.append(("FAIL", "manifest quality summary", "; ".join(failures) + f" ({detail})"))
    else:
        results.append(("OK", "manifest quality summary", detail))


def inspect_climate_source_coverage(bundle_dir: Path, results: list[tuple[str, str, str]]) -> None:
    try:
        climate = load_json(bundle_dir / "climate-signal.json")
    except Exception as exc:  # noqa: BLE001
        results.append(("FAIL", "climate source coverage", str(exc)))
        return

    source_status = climate.get("sourceStatus")
    if not isinstance(source_status, list) or not source_status:
        results.append(("FAIL", "climate source coverage", "missing sourceStatus"))
        return

    attempted = len([source for source in source_status if isinstance(source, dict)])
    ok_sources = source_names_with_status(source_status, "ok")
    failed_sources = source_names_with_status(source_status, "failed")
    no_signal_sources = source_names_with_status(source_status, "no_signal")
    degraded_sources = failed_sources + no_signal_sources
    detail = f"{len(ok_sources)}/{attempted} official sources usable"
    if degraded_sources:
        detail += f"; degraded: {', '.join(degraded_sources)}"

    if len(ok_sources) < MIN_CLIMATE_SOURCE_SUCCESSES:
        results.append(("FAIL", "climate source coverage", detail))
    elif degraded_sources:
        results.append(("WARN", "climate source coverage", detail))
    else:
        results.append(("OK", "climate source coverage", detail))


def inspect_expected_app_data_mode(bundle_dir: Path, results: list[tuple[str, str, str]]) -> None:
    try:
        model = load_json(bundle_dir / "model-calibration.json")
        climate = load_json(bundle_dir / "climate-signal.json")
    except Exception as exc:  # noqa: BLE001
        results.append(("FAIL", "expected app data mode", str(exc)))
        return

    has_remote_calibration = model_uses_generated_quality_gate(model)
    has_remote_climate = climate_is_usable_remote_normalized(climate)
    if has_remote_calibration and has_remote_climate:
        results.append(("OK", "expected app data mode", "遠端校準"))
    elif has_remote_calibration or has_remote_climate:
        parts = []
        if has_remote_calibration:
            parts.append("model calibration")
        if has_remote_climate:
            parts.append("climate signal")
        results.append(("FAIL", "expected app data mode", f"混合資料; only {', '.join(parts)} passed remote gates"))
    else:
        results.append(("FAIL", "expected app data mode", "本機保守; no generated remote data passes app gates"))


def inspect_workflow(workflow_path: Path, results: list[tuple[str, str, str]]) -> None:
    try:
        text = workflow_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        results.append(("FAIL", "daily refresh workflow", f"missing {workflow_path}"))
        return

    required_snippets = [
        "schedule:",
        "tools/build_weather_data_bundle.py",
        "tools/validate_weather_data_bundle.py",
        "--report-json outputs/weatherai_release_readiness.json",
        "actions/deploy-pages",
        "Verify deployed WeatherAI manifest",
        "tools/verify_release_weather_config.py",
    ]
    missing = [snippet for snippet in required_snippets if snippet not in text]
    if missing:
        results.append(("FAIL", "daily refresh workflow", f"missing {', '.join(missing)}"))
    else:
        results.append(("OK", "daily refresh workflow", "builds, validates, deploys, and verifies hosted manifest"))


def inspect_release_config(config_path: Path, results: list[tuple[str, str, str]], allow_empty: bool) -> str:
    try:
        settings = parse_xcconfig(config_path)
    except Exception as exc:  # noqa: BLE001
        results.append(("FAIL", "release manifest URL", str(exc)))
        return ""

    direct_overrides = configured_direct_overrides(settings)
    if direct_overrides:
        results.append((
            "FAIL",
            "release direct URL overrides",
            f"must be empty so manifest hashes and quality gates are authoritative: {', '.join(direct_overrides)}",
        ))
    else:
        results.append(("OK", "release direct URL overrides", "empty; manifest is the single data source"))

    manifest_url = normalize_xcconfig_url(settings.get("WEATHER_DATA_MANIFEST_URL", ""))
    if not manifest_url:
        if allow_empty:
            results.append((
                "WARN",
                "release manifest URL",
                f"{config_path} WEATHER_DATA_MANIFEST_URL is empty; allowed for data-refresh report only",
            ))
        else:
            results.append(("FAIL", "release manifest URL", f"{config_path} WEATHER_DATA_MANIFEST_URL is empty"))
        return ""

    parsed = urllib.parse.urlparse(manifest_url)
    if parsed.scheme != "https" or not parsed.netloc or not parsed.path.endswith("/manifest.json"):
        results.append(("FAIL", "release manifest URL", f"must be an HTTPS /manifest.json URL; got {manifest_url!r}"))
        return manifest_url

    results.append(("OK", "release manifest URL", manifest_url))
    return manifest_url


def run_release_preflight(manifest_url: str, results: list[tuple[str, str, str]], allow_empty: bool) -> None:
    if not manifest_url:
        if allow_empty:
            results.append(("WARN", "release preflight", "skipped because release manifest URL is empty"))
        else:
            results.append(("FAIL", "release preflight", "skipped because release manifest URL is empty"))
        return

    command = [sys.executable, "tools/verify_release_weather_config.py"]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode == 0:
        results.append(("OK", "release preflight", result.stdout.strip().splitlines()[-1]))
    else:
        detail = (result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}")
        results.append(("FAIL", "release preflight", detail))


def model_uses_generated_quality_gate(model: dict[str, Any]) -> bool:
    version = str(model.get("version", ""))
    if version.startswith("static-") or version.startswith("mvp-local-"):
        return False

    training_window = model.get("trainingWindow")
    regional = model.get("regional")
    global_entries = model.get("global")
    if not isinstance(training_window, dict):
        return False
    if as_int(training_window.get("days")) < MIN_TRAINING_DAYS:
        return False
    if not isinstance(global_entries, list) or len(global_entries) < 3:
        return False
    if not isinstance(regional, list) or len(regional) < 5:
        return False

    regional_groups = [
        region for region in regional
        if isinstance(region, dict) and isinstance(region.get("entries"), list)
    ]
    if len(regional_groups) < 5 or any(len(region["entries"]) < 3 for region in regional_groups):
        return False

    entries = global_entries + [entry for region in regional_groups for entry in region["entries"]]
    return bool(entries) and all(model_entry_passes_app_gate(entry) for entry in entries)


def model_entry_passes_app_gate(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False

    required = [
        "sampleCount",
        "temperatureMAE",
        "rainBrierScore",
        "combinedError",
        "referenceError",
        "sampleConfidence",
        "rawWeightMultiplier",
        "calibrationMethod",
    ]
    if any(field not in entry for field in required):
        return False
    if not isinstance(entry.get("calibrationMethod"), str) or not entry["calibrationMethod"].strip():
        return False

    sample_count = as_float(entry.get("sampleCount"))
    temperature_mae = as_float(entry.get("temperatureMAE"))
    rain_brier = as_float(entry.get("rainBrierScore"))
    combined_error = as_float(entry.get("combinedError"))
    reference_error = as_float(entry.get("referenceError"))
    sample_confidence = as_float(entry.get("sampleConfidence"))
    raw_multiplier = as_float(entry.get("rawWeightMultiplier"))
    multiplier = as_float(entry.get("weightMultiplier"))
    rain_bias = as_float(entry.get("rainBiasCorrection"))
    if None in (
        sample_count,
        temperature_mae,
        rain_brier,
        combined_error,
        reference_error,
        sample_confidence,
        raw_multiplier,
        multiplier,
        rain_bias,
    ):
        return False

    return (
        sample_count >= MIN_ENTRY_SAMPLES
        and temperature_mae <= 8
        and 0 <= rain_brier <= 0.45
        and combined_error > 0
        and reference_error > 0
        and 0.5 <= sample_confidence <= 1
        and 0.65 <= raw_multiplier <= 1.35
        and abs(multiplier - 1) <= 0.35
        and -0.12 <= rain_bias <= 0.12
    )


def climate_is_usable_remote_normalized(climate: dict[str, Any]) -> bool:
    if not climate.get("generatedAt"):
        return False
    source_status = climate.get("sourceStatus")
    if not isinstance(source_status, list):
        return False
    successful_sources = [
        source for source in source_status
        if isinstance(source, dict) and source.get("status") == "ok"
    ]
    return len(successful_sources) >= MIN_CLIMATE_SOURCE_SUCCESSES


def source_names_with_status(source_status: list[Any], status: str) -> list[str]:
    return [
        str(source.get("name") or source.get("url") or "unknown")
        for source in source_status
        if isinstance(source, dict) and source.get("status") == status
    ]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def parse_xcconfig(path: Path) -> dict[str, str]:
    settings: dict[str, str] = {}
    if not path.exists():
        raise FileNotFoundError(path)
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        settings[key.strip()] = value.strip()
    return settings


def normalize_xcconfig_url(value: str) -> str:
    return value.strip().replace(":/$()/", "://")


def configured_direct_overrides(config: dict[str, str]) -> list[str]:
    configured = []
    for key in ("WEATHER_MODEL_CALIBRATION_URL", "WEATHER_CLIMATE_SIGNAL_URL"):
        value = normalize_xcconfig_url(config.get(key, ""))
        if value and not value.startswith("$("):
            configured.append(key)
    return configured


def as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
