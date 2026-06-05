#!/usr/bin/env python3
"""Generate WeatherAI model calibration tables from public Open-Meteo data.

The output schema matches `ModelWeightCalibrationTable` in WeatherAI/ContentView.swift.
It is designed for a backend cron job or GitHub Actions workflow that publishes
the resulting JSON to WEATHER_MODEL_CALIBRATION_URL.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ARCHIVE_ENDPOINT = "https://archive-api.open-meteo.com/v1/archive"
RAIN_EVENT_MM = 0.1
DEFAULT_LEAD_DAYS = [1, 2, 3, 5, 7]
DEFAULT_MIN_SAMPLES = 48
DEFAULT_CONFIDENCE_SAMPLES = 96


@dataclass(frozen=True)
class Provider:
    model_name: str
    previous_runs_endpoint: str
    query_params: dict[str, str]


@dataclass(frozen=True)
class Location:
    identifier: str
    name: str
    region_id: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class Region:
    identifier: str
    name: str
    latitude_range: list[float] | None = None
    longitude_range: list[float] | None = None


@dataclass(frozen=True)
class FetchPolicy:
    timeout_seconds: float
    retry_count: int
    retry_backoff_seconds: float


@dataclass
class ScoreBucket:
    temperatures_abs_error: list[float] = field(default_factory=list)
    brier_scores: list[float] = field(default_factory=list)
    forecast_rain_probabilities: list[float] = field(default_factory=list)
    observed_rain_events: list[float] = field(default_factory=list)

    @property
    def sample_count(self) -> int:
        return min(len(self.temperatures_abs_error), len(self.brier_scores))

    @property
    def temperature_mae(self) -> float:
        return statistics.fmean(self.temperatures_abs_error) if self.temperatures_abs_error else math.nan

    @property
    def rain_brier(self) -> float:
        return statistics.fmean(self.brier_scores) if self.brier_scores else math.nan

    @property
    def rain_bias_correction(self) -> float:
        if not self.forecast_rain_probabilities or not self.observed_rain_events:
            return 0.0

        forecast_rate = statistics.fmean(self.forecast_rain_probabilities)
        observed_rate = statistics.fmean(self.observed_rain_events)
        return clamp(observed_rate - forecast_rate, -0.08, 0.08)

    @property
    def combined_error(self) -> float:
        if not self.temperatures_abs_error or not self.brier_scores:
            return math.inf

        return self.temperature_mae + self.rain_brier * 6.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate WeatherAI model calibration JSON from Open-Meteo public archives."
    )
    parser.add_argument(
        "--locations",
        default="data/model_calibration_locations.json",
        help="Provider/location config JSON.",
    )
    parser.add_argument(
        "--output",
        default="data/model_calibration_generated.json",
        help="Output calibration JSON path.",
    )
    parser.add_argument(
        "--start-date",
        help="UTC start date YYYY-MM-DD. Defaults to 35 days before today.",
    )
    parser.add_argument(
        "--end-date",
        help="UTC end date YYYY-MM-DD. Defaults to 3 days before today.",
    )
    parser.add_argument(
        "--lead-days",
        default=",".join(str(day) for day in DEFAULT_LEAD_DAYS),
        help="Comma-separated lead days to score, e.g. 1,2,3,5,7.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=DEFAULT_MIN_SAMPLES,
        help="Minimum paired hourly samples required for a model/bucket entry.",
    )
    parser.add_argument(
        "--ttl-hours",
        type=int,
        default=168,
        help="Freshness TTL to place in generated table.",
    )
    parser.add_argument(
        "--confidence-samples",
        type=int,
        default=DEFAULT_CONFIDENCE_SAMPLES,
        help="Sample count scale used to shrink weight/bias changes toward neutral.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=45.0,
        help="Seconds to wait for each public API request.",
    )
    parser.add_argument(
        "--retry-count",
        type=int,
        default=3,
        help="Attempts per public API request before failing.",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=2.0,
        help="Base exponential backoff delay between request retries.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print representative API URLs without fetching.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.locations)
    output_path = Path(args.output)
    start_date, end_date = date_window(args.start_date, args.end_date)
    lead_days = parse_lead_days(args.lead_days)

    config = load_json(config_path)
    providers = parse_providers(config)
    regions = parse_regions(config)
    locations = parse_locations(config)
    fetch_policy = FetchPolicy(
        timeout_seconds=args.request_timeout,
        retry_count=max(1, args.retry_count),
        retry_backoff_seconds=max(0, args.retry_backoff_seconds),
    )

    if args.dry_run:
        print_dry_run(providers, locations, start_date, end_date, lead_days)
        return 0

    global_scores: dict[tuple[str, int], ScoreBucket] = {}
    regional_scores: dict[tuple[str, str, int], ScoreBucket] = {}

    for location in locations:
        print(f"Scoring {location.name} ({location.region_id})")
        truth = fetch_truth(location, start_date, end_date, fetch_policy)
        for provider in providers:
            forecast = fetch_previous_runs(provider, location, start_date, end_date, lead_days, fetch_policy)
            score_forecast(
                provider=provider,
                location=location,
                truth=truth,
                forecast=forecast,
                lead_days=lead_days,
                global_scores=global_scores,
                regional_scores=regional_scores,
            )

    table = build_calibration_table(
        providers=providers,
        regions=regions,
        global_scores=global_scores,
        regional_scores=regional_scores,
        start_date=start_date,
        end_date=end_date,
        min_samples=args.min_samples,
        confidence_samples=args.confidence_samples,
        ttl_hours=args.ttl_hours,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(table, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path} with version {table['version']}")
    return 0


def date_window(start_date: str | None, end_date: str | None) -> tuple[dt.date, dt.date]:
    today = dt.datetime.now(dt.timezone.utc).date()
    start = parse_date(start_date) if start_date else today - dt.timedelta(days=35)
    end = parse_date(end_date) if end_date else today - dt.timedelta(days=3)
    if start > end:
        raise SystemExit("--start-date must be before --end-date")
    return start, end


def parse_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise SystemExit(f"Invalid date {value!r}; expected YYYY-MM-DD") from exc


def parse_lead_days(value: str) -> list[int]:
    try:
        days = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as exc:
        raise SystemExit("--lead-days must be comma-separated integers") from exc

    if not days or any(day < 1 or day > 7 for day in days):
        raise SystemExit("--lead-days must contain values from 1 to 7")
    return days


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_providers(config: dict[str, Any]) -> list[Provider]:
    providers = []
    for item in config.get("providers", []):
        providers.append(
            Provider(
                model_name=str(item["modelName"]),
                previous_runs_endpoint=str(item["previousRunsEndpoint"]),
                query_params={str(key): str(value) for key, value in item.get("queryParams", {}).items()},
            )
        )
    if not providers:
        raise SystemExit("No providers in calibration config")
    return providers


def parse_locations(config: dict[str, Any]) -> list[Location]:
    locations = []
    for item in config.get("locations", []):
        locations.append(
            Location(
                identifier=str(item["id"]),
                name=str(item["name"]),
                region_id=str(item.get("regionId", "global")),
                latitude=float(item["latitude"]),
                longitude=float(item["longitude"]),
            )
        )
    if not locations:
        raise SystemExit("No locations in calibration config")
    return locations


def parse_regions(config: dict[str, Any]) -> list[Region]:
    regions = []
    for item in config.get("regions", []):
        regions.append(
            Region(
                identifier=str(item["id"]),
                name=str(item["name"]),
                latitude_range=as_float_list(item.get("latitudeRange")),
                longitude_range=as_float_list(item.get("longitudeRange")),
            )
        )
    return regions


def as_float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    return [float(item) for item in value]


def print_dry_run(
    providers: list[Provider],
    locations: list[Location],
    start_date: dt.date,
    end_date: dt.date,
    lead_days: list[int],
) -> None:
    sample_location = locations[0]
    print(f"Date window: {start_date} to {end_date}")
    print(f"Lead days: {lead_days}")
    print("Truth URL:")
    print(build_truth_url(sample_location, start_date, end_date))
    for provider in providers:
        print(f"{provider.model_name} previous-runs URL:")
        print(build_previous_runs_url(provider, sample_location, start_date, end_date, lead_days))


def fetch_truth(
    location: Location,
    start_date: dt.date,
    end_date: dt.date,
    fetch_policy: FetchPolicy,
) -> dict[str, dict[str, float]]:
    payload = request_json(build_truth_url(location, start_date, end_date), fetch_policy)
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    temperatures = hourly.get("temperature_2m", [])
    precipitation = hourly.get("precipitation", [])

    truth: dict[str, dict[str, float]] = {}
    for index, timestamp in enumerate(times):
        temperature = value_at(temperatures, index)
        rain = value_at(precipitation, index)
        if temperature is None or rain is None:
            continue
        truth[str(timestamp)] = {"temperature": temperature, "precipitation": rain}
    return truth


def fetch_previous_runs(
    provider: Provider,
    location: Location,
    start_date: dt.date,
    end_date: dt.date,
    lead_days: list[int],
    fetch_policy: FetchPolicy,
) -> dict[str, Any]:
    return request_json(
        build_previous_runs_url(provider, location, start_date, end_date, lead_days),
        fetch_policy,
    ).get("hourly", {})


def build_truth_url(location: Location, start_date: dt.date, end_date: dt.date) -> str:
    return build_url(
        ARCHIVE_ENDPOINT,
        {
            "latitude": location.latitude,
            "longitude": location.longitude,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "hourly": "temperature_2m,precipitation",
            "timezone": "UTC",
        },
    )


def build_previous_runs_url(
    provider: Provider,
    location: Location,
    start_date: dt.date,
    end_date: dt.date,
    lead_days: list[int],
) -> str:
    hourly_variables = []
    for day in lead_days:
        hourly_variables.append(f"temperature_2m_previous_day{day}")
        hourly_variables.append(f"precipitation_previous_day{day}")

    params = {
        "latitude": location.latitude,
        "longitude": location.longitude,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": ",".join(hourly_variables),
        "timezone": "UTC",
    }
    params.update(provider.query_params)
    return build_url(provider.previous_runs_endpoint, params)


def build_url(endpoint: str, params: dict[str, Any]) -> str:
    return endpoint + "?" + urllib.parse.urlencode(params)


def request_json(url: str, fetch_policy: FetchPolicy) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "WeatherAI-calibration/1.0"})
    last_error: Exception | None = None
    for attempt in range(1, fetch_policy.retry_count + 1):
        try:
            with urllib.request.urlopen(request, timeout=fetch_policy.timeout_seconds) as response:
                if response.status < 200 or response.status >= 300:
                    raise RuntimeError(f"HTTP {response.status} for {url}")
                return json.loads(response.read().decode("utf-8"))
        except (TimeoutError, urllib.error.URLError, RuntimeError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt >= fetch_policy.retry_count:
                break
            delay = fetch_policy.retry_backoff_seconds * (2 ** (attempt - 1))
            print(f"Retrying request after {type(exc).__name__}: attempt {attempt + 1}/{fetch_policy.retry_count}", file=sys.stderr)
            time.sleep(delay)

    raise RuntimeError(f"Failed to fetch {url}: {last_error}") from last_error


def score_forecast(
    provider: Provider,
    location: Location,
    truth: dict[str, dict[str, float]],
    forecast: dict[str, Any],
    lead_days: list[int],
    global_scores: dict[tuple[str, int], ScoreBucket],
    regional_scores: dict[tuple[str, str, int], ScoreBucket],
) -> None:
    times = forecast.get("time", [])
    for lead_day in lead_days:
        temperature_values = forecast.get(f"temperature_2m_previous_day{lead_day}", [])
        precipitation_values = forecast.get(f"precipitation_previous_day{lead_day}", [])
        bucket_hours = lead_bucket_hours(lead_day)
        global_bucket = global_scores.setdefault((provider.model_name, bucket_hours), ScoreBucket())
        region_bucket = regional_scores.setdefault((location.region_id, provider.model_name, bucket_hours), ScoreBucket())

        for index, timestamp in enumerate(times):
            observation = truth.get(str(timestamp))
            forecast_temperature = value_at(temperature_values, index)
            forecast_precipitation = value_at(precipitation_values, index)
            if not observation or forecast_temperature is None or forecast_precipitation is None:
                continue

            rain_probability = precipitation_probability_proxy(forecast_precipitation)
            observed_event = 1.0 if observation["precipitation"] >= RAIN_EVENT_MM else 0.0
            temperature_error = abs(forecast_temperature - observation["temperature"])
            brier = (rain_probability - observed_event) ** 2
            for bucket in (global_bucket, region_bucket):
                bucket.temperatures_abs_error.append(temperature_error)
                bucket.brier_scores.append(brier)
                bucket.forecast_rain_probabilities.append(rain_probability)
                bucket.observed_rain_events.append(observed_event)


def value_at(values: list[Any], index: int) -> float | None:
    if index >= len(values):
        return None
    value = values[index]
    if value is None:
        return None
    return float(value)


def lead_bucket_hours(lead_day: int) -> int:
    return 72 if lead_day <= 3 else 240


def precipitation_probability_proxy(precipitation: float) -> float:
    if precipitation < 0.1:
        return 0.04
    if precipitation < 0.5:
        return 0.28
    if precipitation < 2:
        return 0.58
    if precipitation < 8:
        return 0.78
    return 0.9


def build_calibration_table(
    providers: list[Provider],
    regions: list[Region],
    global_scores: dict[tuple[str, int], ScoreBucket],
    regional_scores: dict[tuple[str, str, int], ScoreBucket],
    start_date: dt.date,
    end_date: dt.date,
    min_samples: int,
    confidence_samples: int,
    ttl_hours: int,
) -> dict[str, Any]:
    generated_at = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    version = f"openmeteo-prev-runs-{generated_at.strftime('%Y%m%dT%H%M%SZ')}"
    training_window_days = (end_date - start_date).days + 1
    lead_buckets = sorted({bucket for _, bucket in global_scores.keys()} | {72, 240})
    provider_names = [provider.model_name for provider in providers]

    global_entries = entries_from_scores(
        provider_names=provider_names,
        lead_buckets=lead_buckets,
        scores={(model, bucket): value for (model, bucket), value in global_scores.items()},
        min_samples=min_samples,
        confidence_samples=confidence_samples,
    )

    regional_entries = []
    region_by_id = {region.identifier: region for region in regions}
    for region_id in sorted({key[0] for key in regional_scores.keys()} - {"global"}):
        region = region_by_id.get(region_id)
        if region is None:
            continue
        region_entries = entries_from_scores(
            provider_names=provider_names,
            lead_buckets=lead_buckets,
            scores={(model, bucket): value for (rid, model, bucket), value in regional_scores.items() if rid == region_id},
            min_samples=min_samples,
            confidence_samples=confidence_samples,
        )
        if not region_entries:
            continue
        regional_entries.append(
            {
                "id": region.identifier,
                "name": region.name,
                "latitudeRange": region.latitude_range,
                "longitudeRange": region.longitude_range,
                "entries": region_entries,
            }
        )

    return {
        "version": version,
        "updatedAt": generated_at.isoformat().replace("+00:00", "Z"),
        "ttlHours": ttl_hours,
        "trainingWindow": {
            "startDate": start_date.isoformat(),
            "endDate": end_date.isoformat(),
            "days": training_window_days,
        },
        "sourceSummary": (
            "Generated from Open-Meteo Previous Runs forecasts scored against "
            f"Open-Meteo Historical Weather archive observations, {start_date} to {end_date} UTC. "
            f"Weight and rain-bias changes are sample-confidence shrunk toward neutral at {confidence_samples} samples."
        ),
        "global": global_entries,
        "regional": regional_entries,
    }


def entries_from_scores(
    provider_names: list[str],
    lead_buckets: list[int],
    scores: dict[tuple[str, int], ScoreBucket],
    min_samples: int,
    confidence_samples: int,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for bucket in lead_buckets:
        bucket_scores = {
            model_name: scores.get((model_name, bucket))
            for model_name in provider_names
            if scores.get((model_name, bucket)) and scores[(model_name, bucket)].sample_count >= min_samples
        }
        if not bucket_scores:
            continue

        reference_error = statistics.median(score.combined_error for score in bucket_scores.values())
        for model_name, score in bucket_scores.items():
            confidence = sample_confidence(score.sample_count, confidence_samples)
            raw_multiplier = clamp(reference_error / max(score.combined_error, 0.001), 0.65, 1.35)
            multiplier = 1 + (raw_multiplier - 1) * confidence
            rain_bias = score.rain_bias_correction * confidence
            entries.append(
                {
                    "modelName": model_name,
                    "leadTimeMaxHours": bucket,
                    "weightMultiplier": round(multiplier, 3),
                    "rainBiasCorrection": round(rain_bias, 3),
                    "sampleCount": score.sample_count,
                    "temperatureMAE": round(score.temperature_mae, 3),
                    "rainBrierScore": round(score.rain_brier, 4),
                    "combinedError": round(score.combined_error, 4),
                    "referenceError": round(reference_error, 4),
                    "sampleConfidence": round(confidence, 3),
                    "rawWeightMultiplier": round(raw_multiplier, 3),
                    "calibrationMethod": "median-relative-error-with-sample-confidence-shrinkage",
                }
            )
    return entries


def sample_confidence(sample_count: int, confidence_samples: int) -> float:
    scale = max(1, confidence_samples)
    return clamp(math.sqrt(sample_count / (sample_count + scale)), 0.15, 1.0)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


if __name__ == "__main__":
    sys.exit(main())
