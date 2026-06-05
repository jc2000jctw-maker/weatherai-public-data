#!/usr/bin/env python3
"""Build a deployable WeatherAI public data bundle."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


MODEL_FILE = "model-calibration.json"
CLIMATE_FILE = "climate-signal.json"
MANIFEST_FILE = "manifest.json"
HEALTH_FILE = "health.json"
REFRESH_CADENCE_HOURS = 24
MAX_BUNDLE_AGE_HOURS = 192


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deployable WeatherAI model calibration and climate signal JSON files."
    )
    parser.add_argument("--output-dir", default="public/weatherai-data")
    parser.add_argument("--start-date", help="UTC model-calibration start date YYYY-MM-DD.")
    parser.add_argument("--end-date", help="UTC model-calibration end date YYYY-MM-DD.")
    parser.add_argument("--lead-days", default="1,2,3,5,7")
    parser.add_argument("--min-samples", type=int, default=48)
    parser.add_argument("--ttl-hours", type=int, default=168)
    parser.add_argument("--request-timeout", type=float, default=45.0)
    parser.add_argument("--retry-count", type=int, default=3)
    parser.add_argument("--retry-backoff-seconds", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without fetching.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    model_path = output_dir / MODEL_FILE
    climate_path = output_dir / CLIMATE_FILE
    manifest_path = output_dir / MANIFEST_FILE
    health_path = output_dir / HEALTH_FILE

    model_command = [
        sys.executable,
        "tools/generate_model_calibration.py",
        "--output",
        str(model_path),
        "--lead-days",
        args.lead_days,
        "--min-samples",
        str(args.min_samples),
        "--ttl-hours",
        str(args.ttl_hours),
        "--request-timeout",
        str(args.request_timeout),
        "--retry-count",
        str(args.retry_count),
        "--retry-backoff-seconds",
        str(args.retry_backoff_seconds),
    ]
    if args.start_date:
        model_command.extend(["--start-date", args.start_date])
    if args.end_date:
        model_command.extend(["--end-date", args.end_date])

    climate_command = [
        sys.executable,
        "tools/generate_climate_signal.py",
        "--output",
        str(climate_path),
        "--request-timeout",
        str(args.request_timeout),
        "--retry-count",
        str(args.retry_count),
        "--retry-backoff-seconds",
        str(args.retry_backoff_seconds),
    ]

    if args.dry_run:
        print("Model calibration command:")
        print(" ".join(model_command))
        print("Climate signal command:")
        print(" ".join(climate_command))
        print(f"Manifest path: {manifest_path}")
        print(f"Health path: {health_path}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    run(model_command)
    run(climate_command)

    model_payload = load_json(model_path)
    climate_payload = load_json(climate_path)
    validate_model_payload(model_payload)
    validate_climate_payload(climate_payload)

    generated_at = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    quality_summary = build_quality_summary(model_payload, climate_payload, min_samples=args.min_samples)
    health = build_health_payload(
        generated_at=generated_at,
        model_payload=model_payload,
        climate_payload=climate_payload,
        quality_summary=quality_summary,
    )
    health_bytes = json_bytes(health)
    manifest = {
        "version": generated_at,
        "generatedAt": generated_at,
        "files": {
            "modelCalibration": {
                "path": MODEL_FILE,
                "sha256": sha256_file(model_path),
                "version": model_payload["version"],
                "updatedAt": model_payload["updatedAt"],
                "ttlHours": model_payload["ttlHours"],
            },
            "climateSignal": {
                "path": CLIMATE_FILE,
                "sha256": sha256_file(climate_path),
                "phase": climate_payload["phase"],
                "updatedAt": climate_payload["updatedAt"],
            },
            "health": {
                "path": HEALTH_FILE,
                "sha256": sha256_bytes(health_bytes),
            },
        },
        "qualitySummary": quality_summary,
        "appBuildSettings": {
            "preferred": "WEATHER_DATA_MANIFEST_URL",
            "WEATHER_DATA_MANIFEST_URL": MANIFEST_FILE,
            "WEATHER_MODEL_CALIBRATION_URL": MODEL_FILE,
            "WEATHER_CLIMATE_SIGNAL_URL": CLIMATE_FILE,
        },
    }
    manifest_path.write_bytes(json_bytes(manifest))
    health_path.write_bytes(health_bytes)
    print(f"Wrote deployable WeatherAI data bundle to {output_dir}")
    return 0


def run(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_quality_summary(
    model_payload: dict[str, Any],
    climate_payload: dict[str, Any],
    min_samples: int,
) -> dict[str, Any]:
    entries = model_entries(model_payload)
    sample_counts = number_values(entries, "sampleCount")
    temperature_mae = number_values(entries, "temperatureMAE")
    rain_brier = number_values(entries, "rainBrierScore")
    rain_bias_correction = number_values(entries, "rainBiasCorrection")
    sample_confidence = number_values(entries, "sampleConfidence")
    raw_weight_multiplier = number_values(entries, "rawWeightMultiplier")
    final_weight_delta = [
        abs(value - 1.0)
        for value in number_values(entries, "weightMultiplier")
    ]
    regional_entry_counts = [
        len(region.get("entries", []))
        for region in model_payload.get("regional", [])
        if isinstance(region, dict) and isinstance(region.get("entries"), list)
    ]
    regional_count = sum(
        regional_entry_counts
    )
    source_counts = climate_source_status_counts(climate_payload)

    return {
        "status": "generated",
        "validationProfile": "deployment-default",
        "modelEntryCount": len(entries),
        "globalEntryCount": len(model_payload.get("global", [])),
        "regionalEntryCount": regional_count,
        "regionalGroupCount": len(regional_entry_counts),
        "minRegionalEntryCount": min(regional_entry_counts) if regional_entry_counts else 0,
        "trainingWindowDays": model_payload.get("trainingWindow", {}).get("days"),
        "minSampleCount": int(min(sample_counts)) if sample_counts else 0,
        "maxTemperatureMAE": round(max(temperature_mae), 3) if temperature_mae else None,
        "maxRainBrierScore": round(max(rain_brier), 4) if rain_brier else None,
        "maxAbsoluteRainBiasCorrection": round(max(abs(value) for value in rain_bias_correction), 4) if rain_bias_correction else None,
        "minSampleConfidence": round(min(sample_confidence), 3) if sample_confidence else None,
        "maxRawWeightMultiplier": round(max(raw_weight_multiplier), 3) if raw_weight_multiplier else None,
        "maxFinalWeightDelta": round(max(final_weight_delta), 3) if final_weight_delta else None,
        "requiredMinSampleCount": min_samples,
        "climateSourceCount": len(climate_payload.get("sourceDetails", [])),
        "climateSourceAttemptCount": source_counts["attempted"],
        "climateSourceSuccessCount": source_counts["ok"],
        "climateSourceFailureCount": source_counts["failed"],
        "climateSourceNoSignalCount": source_counts["no_signal"],
        "climateConfidencePenalty": climate_payload.get("confidencePenalty"),
        "climateUncertaintyShrinkBonus": climate_payload.get("uncertaintyShrinkBonus"),
    }


def build_health_payload(
    generated_at: str,
    model_payload: dict[str, Any],
    climate_payload: dict[str, Any],
    quality_summary: dict[str, Any],
) -> dict[str, Any]:
    source_counts = climate_source_status_counts(climate_payload)
    return {
        "status": "ok",
        "generatedAt": generated_at,
        "modelCalibration": {
            "version": model_payload["version"],
            "updatedAt": model_payload["updatedAt"],
            "ttlHours": model_payload["ttlHours"],
        },
        "climateSignal": {
            "phase": climate_payload["phase"],
            "updatedAt": climate_payload["updatedAt"],
            "sourceStatusSummary": source_counts,
            "sourceStatus": climate_source_status(climate_payload),
            "failedSources": [
                item.get("name", "unknown")
                for item in climate_payload.get("sourceStatus", [])
                if isinstance(item, dict) and item.get("status") == "failed"
            ],
            "noSignalSources": [
                item.get("name", "unknown")
                for item in climate_payload.get("sourceStatus", [])
                if isinstance(item, dict) and item.get("status") == "no_signal"
            ],
        },
        "freshness": build_freshness_payload(generated_at, model_payload, climate_payload),
        "modelPerformanceSummary": build_model_performance_summary(model_payload),
        "qualitySummary": quality_summary,
    }


def build_freshness_payload(
    generated_at: str,
    model_payload: dict[str, Any],
    climate_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "refreshCadenceHours": REFRESH_CADENCE_HOURS,
        "maxBundleAgeHours": MAX_BUNDLE_AGE_HOURS,
        "nextRecommendedRefreshAt": add_hours(generated_at, REFRESH_CADENCE_HOURS),
        "staleAfter": add_hours(generated_at, MAX_BUNDLE_AGE_HOURS),
        "modelExpiresAt": add_hours(
            str(model_payload["updatedAt"]),
            int(model_payload["ttlHours"]),
        ),
        "climateGeneratedAt": climate_payload.get("generatedAt"),
    }


def add_hours(value: str, hours: int) -> str:
    timestamp = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    return (timestamp + dt.timedelta(hours=hours)).isoformat().replace("+00:00", "Z")


def build_model_performance_summary(model_payload: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in model_entries(model_payload):
        model_name = str(entry.get("modelName", "")).strip()
        if not model_name:
            continue
        grouped.setdefault(model_name, []).append(entry)

    models = [
        summarize_model_performance(model_name, entries)
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


def summarize_model_performance(model_name: str, entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "modelName": model_name,
        "entryCount": len(entries),
        "sampleCount": int(sum_number(entries, "sampleCount")),
        "averageTemperatureMAE": round(weighted_average(entries, "temperatureMAE"), 3),
        "averageRainBrierScore": round(weighted_average(entries, "rainBrierScore"), 4),
        "averageRainBiasCorrection": round(weighted_average(entries, "rainBiasCorrection"), 4),
        "maxAbsoluteRainBiasCorrection": round(max_absolute_number(entries, "rainBiasCorrection"), 4),
        "averageCombinedError": round(weighted_average(entries, "combinedError"), 4),
        "averageWeightMultiplier": round(weighted_average(entries, "weightMultiplier"), 3),
    }


def weighted_average(entries: list[dict[str, Any]], field: str) -> float:
    weighted_sum = 0.0
    weight_sum = 0.0
    for entry in entries:
        value = optional_float(entry.get(field))
        weight = optional_float(entry.get("sampleCount")) or 0.0
        if value is None or weight <= 0:
            continue
        weighted_sum += value * weight
        weight_sum += weight
    return weighted_sum / weight_sum if weight_sum else 0.0


def sum_number(entries: list[dict[str, Any]], field: str) -> float:
    return sum(optional_float(entry.get(field)) or 0.0 for entry in entries)


def max_absolute_number(entries: list[dict[str, Any]], field: str) -> float:
    values = [abs(value) for value in (optional_float(entry.get(field)) for entry in entries) if value is not None]
    return max(values) if values else 0.0


def model_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    entries = [entry for entry in payload.get("global", []) if isinstance(entry, dict)]
    for region in payload.get("regional", []):
        if isinstance(region, dict):
            entries.extend(entry for entry in region.get("entries", []) if isinstance(entry, dict))
    return entries


def number_values(entries: list[dict[str, Any]], field: str) -> list[float]:
    values = []
    for entry in entries:
        value = entry.get(field)
        if isinstance(value, bool):
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def optional_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def validate_model_payload(payload: dict[str, Any]) -> None:
    required = {"version", "updatedAt", "ttlHours", "sourceSummary", "global", "regional"}
    missing = required - payload.keys()
    if missing:
        raise SystemExit(f"Model calibration payload missing fields: {sorted(missing)}")


def validate_climate_payload(payload: dict[str, Any]) -> None:
    required = {
        "phase",
        "probability",
        "updatedAt",
        "summary",
        "uncertaintyShrinkBonus",
        "confidencePenalty",
        "sourceDetails",
    }
    missing = required - payload.keys()
    if missing:
        raise SystemExit(f"Climate signal payload missing fields: {sorted(missing)}")


if __name__ == "__main__":
    sys.exit(main())
