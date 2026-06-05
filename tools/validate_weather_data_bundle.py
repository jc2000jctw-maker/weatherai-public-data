#!/usr/bin/env python3
"""Validate a WeatherAI deployable public-data bundle before publishing."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


MODEL_FILE = "model-calibration.json"
CLIMATE_FILE = "climate-signal.json"
MANIFEST_FILE = "manifest.json"
HEALTH_FILE = "health.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate WeatherAI generated data bundle quality gates.")
    parser.add_argument("--bundle-dir", default="public/weatherai-data")
    parser.add_argument("--min-entry-samples", type=int, default=48)
    parser.add_argument("--min-global-entries", type=int, default=3)
    parser.add_argument("--min-regional-groups", type=int, default=5)
    parser.add_argument("--min-regional-entries-per-group", type=int, default=3)
    parser.add_argument("--min-calibration-days", type=int, default=14)
    parser.add_argument("--max-temperature-mae", type=float, default=8.0)
    parser.add_argument("--max-rain-brier", type=float, default=0.45)
    parser.add_argument("--min-sample-confidence", type=float, default=0.5)
    parser.add_argument("--min-raw-weight-multiplier", type=float, default=0.65)
    parser.add_argument("--max-raw-weight-multiplier", type=float, default=1.35)
    parser.add_argument("--max-final-weight-delta", type=float, default=0.35)
    parser.add_argument("--max-ttl-hours", type=int, default=336)
    parser.add_argument("--max-bundle-age-hours", type=int, default=192)
    parser.add_argument("--min-climate-source-details", type=int, default=2)
    parser.add_argument("--min-climate-source-successes", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle_dir = Path(args.bundle_dir)
    failures: list[str] = []

    model = load_json(bundle_dir / MODEL_FILE, failures)
    climate = load_json(bundle_dir / CLIMATE_FILE, failures)
    manifest = load_json(bundle_dir / MANIFEST_FILE, failures)
    health = load_json(bundle_dir / HEALTH_FILE, failures)
    if failures:
        report(failures)
        return 1

    validate_model(model, args, failures)
    validate_climate(climate, args, failures)
    validate_manifest(manifest, model, climate, args, bundle_dir, failures)
    validate_health(health, manifest, model, climate, failures)

    if failures:
        report(failures)
        return 1

    print(
        "WeatherAI data bundle passed quality gates: "
        f"{count_entries(model)} model entries, climate phase {climate.get('phase', 'unknown')!r}."
    )
    return 0


def load_json(path: Path, failures: list[str]) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        failures.append(f"Missing file: {path}")
        return {}
    except json.JSONDecodeError as exc:
        failures.append(f"Invalid JSON in {path}: {exc}")
        return {}

    if not isinstance(payload, dict):
        failures.append(f"{path} must contain a JSON object")
        return {}
    return payload


def validate_model(payload: dict[str, Any], args: argparse.Namespace, failures: list[str]) -> None:
    required = {"version", "updatedAt", "ttlHours", "sourceSummary", "global", "regional"}
    missing = required - payload.keys()
    if missing:
        failures.append(f"Model calibration missing fields: {sorted(missing)}")
        return

    updated_at = parse_iso8601(str(payload["updatedAt"]))
    if not updated_at:
        failures.append("Model calibration updatedAt must be ISO-8601")

    ttl_hours = as_number(payload["ttlHours"])
    if ttl_hours is None or ttl_hours <= 0 or ttl_hours > args.max_ttl_hours:
        failures.append(f"Model calibration ttlHours must be 1..{args.max_ttl_hours}")
    elif updated_at and is_older_than(updated_at, float(ttl_hours)):
        failures.append("Model calibration updatedAt is older than ttlHours")

    validate_training_window(payload.get("trainingWindow"), args, failures)

    global_entries = payload.get("global")
    if not isinstance(global_entries, list):
        failures.append("Model calibration global must be a list")
        global_entries = []
    if len(global_entries) < args.min_global_entries:
        failures.append(
            f"Model calibration needs at least {args.min_global_entries} global entries; found {len(global_entries)}"
        )

    regional = payload.get("regional")
    if not isinstance(regional, list):
        failures.append("Model calibration regional must be a list")
        regional = []
    else:
        regional_groups = [
            region for region in regional
            if isinstance(region, dict) and isinstance(region.get("entries"), list)
        ]
        if len(regional_groups) < args.min_regional_groups:
            failures.append(
                f"Model calibration needs at least {args.min_regional_groups} regional groups; found {len(regional_groups)}"
            )
        for index, region in enumerate(regional_groups):
            entries = region.get("entries", [])
            if len(entries) < args.min_regional_entries_per_group:
                region_id = region.get("id", index)
                failures.append(
                    f"Model calibration regional group {region_id!r} needs at least "
                    f"{args.min_regional_entries_per_group} entries; found {len(entries)}"
                )

    for path, entry in iter_entries(global_entries, regional):
        validate_model_entry(path, entry, args, failures)


def validate_model_entry(path: str, entry: Any, args: argparse.Namespace, failures: list[str]) -> None:
    if not isinstance(entry, dict):
        failures.append(f"{path} must be an object")
        return

    for field in ("modelName", "leadTimeMaxHours", "weightMultiplier", "rainBiasCorrection"):
        if field not in entry:
            failures.append(f"{path} missing {field}")

    model_name = entry.get("modelName")
    if not isinstance(model_name, str) or not model_name.strip():
        failures.append(f"{path} modelName must be non-empty")

    lead_time = as_number(entry.get("leadTimeMaxHours"))
    if lead_time is None or lead_time <= 0 or lead_time > 240:
        failures.append(f"{path} leadTimeMaxHours must be 1..240")

    multiplier = as_number(entry.get("weightMultiplier"))
    if multiplier is None or multiplier < 0.55 or multiplier > 1.45:
        failures.append(f"{path} weightMultiplier must be 0.55..1.45")
    elif abs(multiplier - 1.0) > args.max_final_weight_delta:
        failures.append(
            f"{path} weightMultiplier delta {abs(multiplier - 1.0):g} exceeds {args.max_final_weight_delta:g}"
        )

    rain_bias = as_number(entry.get("rainBiasCorrection"))
    if rain_bias is None or rain_bias < -0.12 or rain_bias > 0.12:
        failures.append(f"{path} rainBiasCorrection must be -0.12..0.12")

    sample_count = as_number(entry.get("sampleCount"))
    if sample_count is None:
        failures.append(f"{path} missing audit field sampleCount")
    elif sample_count < args.min_entry_samples:
        failures.append(f"{path} sampleCount {sample_count:g} is below {args.min_entry_samples}")

    temperature_mae = as_number(entry.get("temperatureMAE"))
    if temperature_mae is None:
        failures.append(f"{path} missing audit field temperatureMAE")
    elif temperature_mae > args.max_temperature_mae:
        failures.append(f"{path} temperatureMAE {temperature_mae:g} exceeds {args.max_temperature_mae:g}")

    rain_brier = as_number(entry.get("rainBrierScore"))
    if rain_brier is None:
        failures.append(f"{path} missing audit field rainBrierScore")
    elif rain_brier < 0 or rain_brier > args.max_rain_brier:
        failures.append(f"{path} rainBrierScore {rain_brier:g} exceeds {args.max_rain_brier:g}")

    combined_error = as_number(entry.get("combinedError"))
    if combined_error is None:
        failures.append(f"{path} missing audit field combinedError")
    elif combined_error <= 0:
        failures.append(f"{path} combinedError must be positive")

    reference_error = as_number(entry.get("referenceError"))
    if reference_error is None:
        failures.append(f"{path} missing audit field referenceError")
    elif reference_error <= 0:
        failures.append(f"{path} referenceError must be positive")

    sample_confidence = as_number(entry.get("sampleConfidence"))
    if sample_confidence is None:
        failures.append(f"{path} missing audit field sampleConfidence")
    elif sample_confidence < args.min_sample_confidence or sample_confidence > 1:
        failures.append(f"{path} sampleConfidence must be {args.min_sample_confidence:g}..1")

    raw_multiplier = as_number(entry.get("rawWeightMultiplier"))
    if raw_multiplier is None:
        failures.append(f"{path} missing audit field rawWeightMultiplier")
    elif raw_multiplier < args.min_raw_weight_multiplier or raw_multiplier > args.max_raw_weight_multiplier:
        failures.append(
            f"{path} rawWeightMultiplier must be "
            f"{args.min_raw_weight_multiplier:g}..{args.max_raw_weight_multiplier:g}"
        )

    calibration_method = entry.get("calibrationMethod")
    if not isinstance(calibration_method, str) or not calibration_method.strip():
        failures.append(f"{path} missing audit field calibrationMethod")


def validate_training_window(payload: Any, args: argparse.Namespace, failures: list[str]) -> None:
    if not isinstance(payload, dict):
        failures.append("Model calibration trainingWindow must be an object")
        return

    start = payload.get("startDate")
    end = payload.get("endDate")
    days = as_number(payload.get("days"))
    if not isinstance(start, str) or not parse_iso_date(start):
        failures.append("Model calibration trainingWindow.startDate must be YYYY-MM-DD")
    if not isinstance(end, str) or not parse_iso_date(end):
        failures.append("Model calibration trainingWindow.endDate must be YYYY-MM-DD")
    if days is None or days < args.min_calibration_days:
        failures.append(
            f"Model calibration trainingWindow.days must be at least {args.min_calibration_days}"
        )
        return

    start_date = parse_iso_date(start) if isinstance(start, str) else None
    end_date = parse_iso_date(end) if isinstance(end, str) else None
    if start_date and end_date:
        expected_days = (end_date - start_date).days + 1
        if expected_days != int(days):
            failures.append("Model calibration trainingWindow.days must match startDate/endDate")


def validate_climate(payload: dict[str, Any], args: argparse.Namespace, failures: list[str]) -> None:
    required = {
        "phase",
        "probability",
        "updatedAt",
        "generatedAt",
        "summary",
        "uncertaintyShrinkBonus",
        "confidencePenalty",
        "sourceDetails",
        "sourceStatus",
    }
    missing = required - payload.keys()
    if missing:
        failures.append(f"Climate signal missing fields: {sorted(missing)}")
        return

    for field in ("phase", "updatedAt", "summary"):
        if not isinstance(payload.get(field), str) or not payload[field].strip():
            failures.append(f"Climate signal {field} must be non-empty")

    generated_at = parse_iso8601(str(payload.get("generatedAt", "")))
    if not generated_at:
        failures.append("Climate signal generatedAt must be ISO-8601")
    elif is_older_than(generated_at, args.max_bundle_age_hours):
        failures.append(f"Climate signal generatedAt is older than {args.max_bundle_age_hours} hours")

    probability = as_number(payload.get("probability"))
    if probability is None or probability < 0 or probability > 1:
        failures.append("Climate signal probability must be 0..1")

    uncertainty_bonus = as_number(payload.get("uncertaintyShrinkBonus"))
    if uncertainty_bonus is None or uncertainty_bonus < 0 or uncertainty_bonus > 0.07:
        failures.append("Climate signal uncertaintyShrinkBonus must be 0..0.07")

    confidence_penalty = as_number(payload.get("confidencePenalty"))
    if confidence_penalty is None or confidence_penalty < 0 or confidence_penalty > 7:
        failures.append("Climate signal confidencePenalty must be 0..7")

    source_details = payload.get("sourceDetails")
    if not isinstance(source_details, list) or not source_details:
        failures.append("Climate signal sourceDetails must be a non-empty list")
    elif len(source_details) < args.min_climate_source_details:
        failures.append(
            f"Climate signal sourceDetails needs at least {args.min_climate_source_details} entries"
        )

    source_status = payload.get("sourceStatus")
    if not isinstance(source_status, list) or not source_status:
        failures.append("Climate signal sourceStatus must be a non-empty list")
    else:
        validate_climate_source_status(source_status, failures)
        successful_sources = sum(
            1
            for item in source_status
            if isinstance(item, dict) and item.get("status") == "ok"
        )
        if successful_sources < args.min_climate_source_successes:
            failures.append(
                "Climate signal needs at least "
                f"{args.min_climate_source_successes} successful official sources; "
                f"found {successful_sources}"
            )


def validate_climate_source_status(payload: list[Any], failures: list[str]) -> None:
    allowed_status = {"ok", "failed", "no_signal"}
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            failures.append(f"Climate signal sourceStatus[{index}] must be an object")
            continue
        name = item.get("name")
        url = item.get("url")
        status = item.get("status")
        if not isinstance(name, str) or not name.strip():
            failures.append(f"Climate signal sourceStatus[{index}].name must be non-empty")
        if not isinstance(url, str) or not url.startswith(("https://", "http://")):
            failures.append(f"Climate signal sourceStatus[{index}].url must be HTTP(S)")
        if status not in allowed_status:
            failures.append(f"Climate signal sourceStatus[{index}].status must be one of {sorted(allowed_status)}")
        if status in {"failed", "no_signal"}:
            error = item.get("error")
            if not isinstance(error, str) or not error.strip():
                failures.append(f"Climate signal sourceStatus[{index}].error must explain non-ok status")


def validate_manifest(
    payload: dict[str, Any],
    model: dict[str, Any],
    climate: dict[str, Any],
    args: argparse.Namespace,
    bundle_dir: Path,
    failures: list[str],
) -> None:
    required = {"version", "generatedAt", "files", "qualitySummary", "appBuildSettings"}
    missing = required - payload.keys()
    if missing:
        failures.append(f"Manifest missing fields: {sorted(missing)}")
        return

    if not parse_iso8601(str(payload["version"])):
        failures.append("Manifest version must be ISO-8601")
    generated_at = parse_iso8601(str(payload["generatedAt"]))
    if not generated_at:
        failures.append("Manifest generatedAt must be ISO-8601")
    elif is_older_than(generated_at, args.max_bundle_age_hours):
        failures.append(f"Manifest generatedAt is older than {args.max_bundle_age_hours} hours")

    files = payload.get("files")
    if not isinstance(files, dict):
        failures.append("Manifest files must be an object")
        return

    model_file = files.get("modelCalibration")
    climate_file = files.get("climateSignal")
    health_file = files.get("health")
    validate_manifest_file("modelCalibration", model_file, MODEL_FILE, bundle_dir / MODEL_FILE, failures)
    validate_manifest_file("climateSignal", climate_file, CLIMATE_FILE, bundle_dir / CLIMATE_FILE, failures)
    validate_manifest_file("health", health_file, HEALTH_FILE, bundle_dir / HEALTH_FILE, failures)

    if isinstance(model_file, dict) and model_file.get("version") != model.get("version"):
        failures.append("Manifest modelCalibration.version must match model-calibration.json")
    if isinstance(model_file, dict) and model_file.get("updatedAt") != model.get("updatedAt"):
        failures.append("Manifest modelCalibration.updatedAt must match model-calibration.json")
    if isinstance(climate_file, dict) and climate_file.get("phase") != climate.get("phase"):
        failures.append("Manifest climateSignal.phase must match climate-signal.json")
    if isinstance(climate_file, dict) and climate_file.get("updatedAt") != climate.get("updatedAt"):
        failures.append("Manifest climateSignal.updatedAt must match climate-signal.json")

    build_settings = payload.get("appBuildSettings")
    if not isinstance(build_settings, dict):
        failures.append("Manifest appBuildSettings must be an object")
        return
    if build_settings.get("preferred") != "WEATHER_DATA_MANIFEST_URL":
        failures.append("Manifest appBuildSettings.preferred must be WEATHER_DATA_MANIFEST_URL")

    validate_quality_summary(payload.get("qualitySummary"), model, climate, failures)


def validate_manifest_file(
    name: str,
    payload: Any,
    expected_path: str,
    actual_path: Path,
    failures: list[str],
) -> None:
    if not isinstance(payload, dict):
        failures.append(f"Manifest files.{name} must be an object")
        return

    path = payload.get("path")
    if path != expected_path:
        failures.append(f"Manifest files.{name}.path must be {expected_path!r}")
    if isinstance(path, str) and (path.startswith("/") or ".." in Path(path).parts):
        failures.append(f"Manifest files.{name}.path must be a safe relative path")

    expected_hash = payload.get("sha256")
    if not isinstance(expected_hash, str) or not is_sha256_hex(expected_hash):
        failures.append(f"Manifest files.{name}.sha256 must be a 64-character lowercase hex SHA-256")
        return

    actual_hash = sha256_file(actual_path)
    if actual_hash != expected_hash:
        failures.append(f"Manifest files.{name}.sha256 must match {expected_path}")


def validate_quality_summary(
    payload: Any,
    model: dict[str, Any],
    climate: dict[str, Any],
    failures: list[str],
) -> None:
    if not isinstance(payload, dict):
        failures.append("Manifest qualitySummary must be an object")
        return

    entries = [entry for _, entry in iter_entries(model.get("global", []), model.get("regional", [])) if isinstance(entry, dict)]
    global_count = len(model.get("global", [])) if isinstance(model.get("global"), list) else 0
    regional_groups = [
        region for region in model.get("regional", [])
        if isinstance(region, dict) and isinstance(region.get("entries"), list)
    ]
    regional_entry_counts = [len(region["entries"]) for region in regional_groups]
    regional_count = sum(regional_entry_counts)
    sample_counts = numeric_entry_values(entries, "sampleCount")
    temperature_mae = numeric_entry_values(entries, "temperatureMAE")
    rain_brier = numeric_entry_values(entries, "rainBrierScore")
    rain_bias_correction = numeric_entry_values(entries, "rainBiasCorrection")
    sample_confidence = numeric_entry_values(entries, "sampleConfidence")
    raw_weight_multiplier = numeric_entry_values(entries, "rawWeightMultiplier")
    final_weight_delta = [abs(value - 1.0) for value in numeric_entry_values(entries, "weightMultiplier")]
    training_window = model.get("trainingWindow") if isinstance(model.get("trainingWindow"), dict) else {}

    expected = {
        "modelEntryCount": len(entries),
        "globalEntryCount": global_count,
        "regionalEntryCount": regional_count,
        "regionalGroupCount": len(regional_groups),
        "minRegionalEntryCount": min(regional_entry_counts) if regional_entry_counts else 0,
        "trainingWindowDays": training_window.get("days"),
        "minSampleCount": int(min(sample_counts)) if sample_counts else 0,
        "maxTemperatureMAE": round(max(temperature_mae), 3) if temperature_mae else None,
        "maxRainBrierScore": round(max(rain_brier), 4) if rain_brier else None,
        "maxAbsoluteRainBiasCorrection": round(max(abs(value) for value in rain_bias_correction), 4) if rain_bias_correction else None,
        "minSampleConfidence": round(min(sample_confidence), 3) if sample_confidence else None,
        "maxRawWeightMultiplier": round(max(raw_weight_multiplier), 3) if raw_weight_multiplier else None,
        "maxFinalWeightDelta": round(max(final_weight_delta), 3) if final_weight_delta else None,
        "climateSourceCount": len(climate.get("sourceDetails", [])) if isinstance(climate.get("sourceDetails"), list) else 0,
        "climateSourceAttemptCount": climate_source_status_counts(climate)["attempted"],
        "climateSourceSuccessCount": climate_source_status_counts(climate)["ok"],
        "climateSourceFailureCount": climate_source_status_counts(climate)["failed"],
        "climateSourceNoSignalCount": climate_source_status_counts(climate)["no_signal"],
        "climateConfidencePenalty": climate.get("confidencePenalty"),
        "climateUncertaintyShrinkBonus": climate.get("uncertaintyShrinkBonus"),
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            failures.append(f"Manifest qualitySummary.{field} must be {value!r}")

    status = payload.get("status")
    if status not in {"generated", "validated"}:
        failures.append("Manifest qualitySummary.status must be generated or validated")

    if payload.get("validationProfile") != "deployment-default":
        failures.append("Manifest qualitySummary.validationProfile must be deployment-default")

    required_min_sample_count = as_number(payload.get("requiredMinSampleCount"))
    if required_min_sample_count is None or required_min_sample_count < 1:
        failures.append("Manifest qualitySummary.requiredMinSampleCount must be positive")
    elif sample_counts and min(sample_counts) < required_min_sample_count:
        failures.append("Manifest qualitySummary.minSampleCount is below requiredMinSampleCount")


def validate_health(
    payload: dict[str, Any],
    manifest: dict[str, Any],
    model: dict[str, Any],
    climate: dict[str, Any],
    failures: list[str],
) -> None:
    required = {
        "status",
        "generatedAt",
        "modelCalibration",
        "climateSignal",
        "freshness",
        "modelPerformanceSummary",
        "qualitySummary",
    }
    missing = required - payload.keys()
    if missing:
        failures.append(f"Health missing fields: {sorted(missing)}")
        return

    if payload.get("status") != "ok":
        failures.append("Health status must be ok")
    if payload.get("generatedAt") != manifest.get("generatedAt"):
        failures.append("Health generatedAt must match manifest.generatedAt")

    model_health = payload.get("modelCalibration")
    if not isinstance(model_health, dict):
        failures.append("Health modelCalibration must be an object")
    else:
        if model_health.get("version") != model.get("version"):
            failures.append("Health modelCalibration.version must match model-calibration.json")
        if model_health.get("updatedAt") != model.get("updatedAt"):
            failures.append("Health modelCalibration.updatedAt must match model-calibration.json")
        if model_health.get("ttlHours") != model.get("ttlHours"):
            failures.append("Health modelCalibration.ttlHours must match model-calibration.json")

    climate_health = payload.get("climateSignal")
    if not isinstance(climate_health, dict):
        failures.append("Health climateSignal must be an object")
    else:
        if climate_health.get("phase") != climate.get("phase"):
            failures.append("Health climateSignal.phase must match climate-signal.json")
        if climate_health.get("updatedAt") != climate.get("updatedAt"):
            failures.append("Health climateSignal.updatedAt must match climate-signal.json")
        source_summary = climate_health.get("sourceStatusSummary")
        if source_summary != climate_source_status_counts(climate):
            failures.append("Health climateSignal.sourceStatusSummary must match climate-signal.json")
        if climate_health.get("sourceStatus") != climate_source_status(climate):
            failures.append("Health climateSignal.sourceStatus must match climate-signal.json")
        if climate_health.get("failedSources") != climate_source_names(climate, "failed"):
            failures.append("Health climateSignal.failedSources must match climate-signal.json")
        if climate_health.get("noSignalSources") != climate_source_names(climate, "no_signal"):
            failures.append("Health climateSignal.noSignalSources must match climate-signal.json")

    if payload.get("qualitySummary") != manifest.get("qualitySummary"):
        failures.append("Health qualitySummary must match manifest.qualitySummary")

    validate_health_freshness(payload.get("freshness"), payload, model, climate, failures)
    validate_model_performance_summary(payload.get("modelPerformanceSummary"), model, failures)


def validate_model_performance_summary(
    payload: Any,
    model: dict[str, Any],
    failures: list[str],
) -> None:
    if not isinstance(payload, dict):
        failures.append("Health modelPerformanceSummary must be an object")
        return

    expected = build_expected_model_performance_summary(model)
    if payload.get("rankingMetric") != expected["rankingMetric"]:
        failures.append("Health modelPerformanceSummary.rankingMetric must match generated metric")
    if payload.get("modelCount") != expected["modelCount"]:
        failures.append("Health modelPerformanceSummary.modelCount must match model-calibration.json")

    models = payload.get("models")
    if not isinstance(models, list):
        failures.append("Health modelPerformanceSummary.models must be a list")
        return

    expected_models = expected["models"]
    if len(models) != len(expected_models):
        failures.append("Health modelPerformanceSummary.models length must match model-calibration.json")
        return

    for index, (actual, expected_model) in enumerate(zip(models, expected_models)):
        if not isinstance(actual, dict):
            failures.append(f"Health modelPerformanceSummary.models[{index}] must be an object")
            continue
        for field, expected_value in expected_model.items():
            if actual.get(field) != expected_value:
                failures.append(
                    f"Health modelPerformanceSummary.models[{index}].{field} must be {expected_value!r}"
                )


def build_expected_model_performance_summary(model: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for _, entry in iter_entries(model.get("global", []), model.get("regional", [])):
        if not isinstance(entry, dict):
            continue
        model_name = str(entry.get("modelName", "")).strip()
        if model_name:
            grouped.setdefault(model_name, []).append(entry)

    models = [
        summarize_expected_model_performance(model_name, entries)
        for model_name, entries in grouped.items()
    ]
    ranked = sorted(
        models,
        key=lambda item: (
            item["averageCombinedError"],
            item["averageRainBrierScore"],
            item["averageTemperatureMAE"],
            item["modelName"],
        ),
    )
    return {
        "rankingMetric": "sample-weighted average combinedError; lower is better",
        "modelCount": len(ranked),
        "models": ranked,
    }


def summarize_expected_model_performance(model_name: str, entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "modelName": model_name,
        "entryCount": len(entries),
        "sampleCount": int(sum_entry_number(entries, "sampleCount")),
        "averageTemperatureMAE": round(weighted_entry_average(entries, "temperatureMAE"), 3),
        "averageRainBrierScore": round(weighted_entry_average(entries, "rainBrierScore"), 4),
        "averageRainBiasCorrection": round(weighted_entry_average(entries, "rainBiasCorrection"), 4),
        "maxAbsoluteRainBiasCorrection": round(max_absolute_entry_number(entries, "rainBiasCorrection"), 4),
        "averageCombinedError": round(weighted_entry_average(entries, "combinedError"), 4),
        "averageWeightMultiplier": round(weighted_entry_average(entries, "weightMultiplier"), 3),
    }


def weighted_entry_average(entries: list[dict[str, Any]], field: str) -> float:
    weighted_sum = 0.0
    weight_sum = 0.0
    for entry in entries:
        value = as_number(entry.get(field))
        weight = as_number(entry.get("sampleCount")) or 0.0
        if value is None or weight <= 0:
            continue
        weighted_sum += value * weight
        weight_sum += weight
    return weighted_sum / weight_sum if weight_sum else 0.0


def sum_entry_number(entries: list[dict[str, Any]], field: str) -> float:
    return sum(as_number(entry.get(field)) or 0.0 for entry in entries)


def max_absolute_entry_number(entries: list[dict[str, Any]], field: str) -> float:
    values = [abs(value) for value in (as_number(entry.get(field)) for entry in entries) if value is not None]
    return max(values) if values else 0.0


def validate_health_freshness(
    payload: Any,
    health: dict[str, Any],
    model: dict[str, Any],
    climate: dict[str, Any],
    failures: list[str],
) -> None:
    if not isinstance(payload, dict):
        failures.append("Health freshness must be an object")
        return

    refresh_cadence = as_number(payload.get("refreshCadenceHours"))
    max_bundle_age = as_number(payload.get("maxBundleAgeHours"))
    if refresh_cadence is None or refresh_cadence <= 0:
        failures.append("Health freshness.refreshCadenceHours must be positive")
    if max_bundle_age is None or max_bundle_age <= 0:
        failures.append("Health freshness.maxBundleAgeHours must be positive")
    elif refresh_cadence is not None and max_bundle_age < refresh_cadence:
        failures.append("Health freshness.maxBundleAgeHours must be >= refreshCadenceHours")

    generated_at = parse_iso8601(str(health.get("generatedAt", "")))
    model_updated_at = parse_iso8601(str(model.get("updatedAt", "")))
    ttl_hours = as_number(model.get("ttlHours"))
    next_refresh_at = parse_iso8601(str(payload.get("nextRecommendedRefreshAt", "")))
    stale_after = parse_iso8601(str(payload.get("staleAfter", "")))
    model_expires_at = parse_iso8601(str(payload.get("modelExpiresAt", "")))
    if generated_at and refresh_cadence is not None:
        expected_next = generated_at + dt.timedelta(hours=float(refresh_cadence))
        if next_refresh_at != expected_next:
            failures.append("Health freshness.nextRecommendedRefreshAt must match generatedAt + refreshCadenceHours")
    elif not next_refresh_at:
        failures.append("Health freshness.nextRecommendedRefreshAt must be ISO-8601")

    if generated_at and max_bundle_age is not None:
        expected_stale = generated_at + dt.timedelta(hours=float(max_bundle_age))
        if stale_after != expected_stale:
            failures.append("Health freshness.staleAfter must match generatedAt + maxBundleAgeHours")
    elif not stale_after:
        failures.append("Health freshness.staleAfter must be ISO-8601")

    if model_updated_at and ttl_hours is not None:
        expected_model_expiry = model_updated_at + dt.timedelta(hours=float(ttl_hours))
        if model_expires_at != expected_model_expiry:
            failures.append("Health freshness.modelExpiresAt must match model updatedAt + ttlHours")
    elif not model_expires_at:
        failures.append("Health freshness.modelExpiresAt must be ISO-8601")

    if payload.get("climateGeneratedAt") != climate.get("generatedAt"):
        failures.append("Health freshness.climateGeneratedAt must match climate-signal.json")


def numeric_entry_values(entries: list[dict[str, Any]], field: str) -> list[float]:
    values = []
    for entry in entries:
        value = as_number(entry.get(field))
        if value is not None:
            values.append(value)
    return values


def iter_entries(global_entries: list[Any], regional: list[Any]) -> list[tuple[str, Any]]:
    entries: list[tuple[str, Any]] = [(f"global[{index}]", entry) for index, entry in enumerate(global_entries)]
    for region_index, region in enumerate(regional):
        if not isinstance(region, dict):
            entries.append((f"regional[{region_index}]", region))
            continue
        region_entries = region.get("entries", [])
        if not isinstance(region_entries, list):
            entries.append((f"regional[{region_index}].entries", region_entries))
            continue
        for entry_index, entry in enumerate(region_entries):
            entries.append((f"regional[{region_index}].entries[{entry_index}]", entry))
    return entries


def count_entries(model: dict[str, Any]) -> int:
    count = len(model.get("global", [])) if isinstance(model.get("global"), list) else 0
    for region in model.get("regional", []):
        if isinstance(region, dict) and isinstance(region.get("entries"), list):
            count += len(region["entries"])
    return count


def climate_source_status_counts(payload: dict[str, Any]) -> dict[str, int]:
    counts = {
        "attempted": 0,
        "ok": 0,
        "failed": 0,
        "no_signal": 0,
    }
    for item in payload.get("sourceStatus", []):
        if not isinstance(item, dict):
            continue
        counts["attempted"] += 1
        status = item.get("status")
        if status in {"ok", "failed", "no_signal"}:
            counts[status] += 1
    return counts


def climate_source_status(payload: dict[str, Any]) -> list[dict[str, Any]]:
    statuses = []
    for item in payload.get("sourceStatus", []):
        if not isinstance(item, dict):
            continue
        status = {
            "name": item.get("name", "unknown"),
            "url": item.get("url", ""),
            "status": item.get("status", "unknown"),
        }
        if item.get("error"):
            status["error"] = item["error"]
        statuses.append(status)
    return statuses


def climate_source_names(payload: dict[str, Any], status: str) -> list[str]:
    names = []
    for item in payload.get("sourceStatus", []):
        if isinstance(item, dict) and item.get("status") == status:
            names.append(str(item.get("name", "unknown")))
    return names


def as_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_sha256_hex(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def parse_iso8601(value: str) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_iso_date(value: str) -> dt.date | None:
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        return None


def is_older_than(value: dt.datetime, max_age_hours: float, now: dt.datetime | None = None) -> bool:
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    current = now or dt.datetime.now(dt.timezone.utc)
    return current - value.astimezone(dt.timezone.utc) > dt.timedelta(hours=max_age_hours)


def report(failures: list[str]) -> None:
    print("WeatherAI data bundle failed quality gates:", file=sys.stderr)
    for failure in failures:
        print(f"- {failure}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
