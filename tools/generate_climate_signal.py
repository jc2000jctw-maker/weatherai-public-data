#!/usr/bin/env python3
"""Generate WeatherAI ClimateSignal JSON from official public climate outlooks."""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CPC_ENSO_STRENGTHS_URL = "https://cpc.ncep.noaa.gov/products/analysis_monitoring/enso/roni/strengths.php"
CPC_GLOBAL_TROPICS_URL = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/ghaz/index.php"
WMO_ENSO_UPDATE_URL = "https://wmo.int/publication-series/el-ninola-nina-updates"
INCOIS_IOD_URL = "https://services.incois.gov.in/portal/IOD"
EL_NINO_PATTERN = r"El\s*Ni(?:ñ|n)o"
LA_NINA_PATTERN = r"La\s*Ni(?:ñ|n)a"
NEUTRAL_PATTERN = r"(?:ENSO[-\s]*neutral|Neutral)"


@dataclass(frozen=True)
class FetchPolicy:
    timeout_seconds: float
    retry_count: int
    retry_backoff_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate WeatherAI ClimateSignal JSON from NOAA CPC and WMO public outlooks."
    )
    parser.add_argument("--output", default="data/climate_signal_generated.json")
    parser.add_argument("--request-timeout", type=float, default=45.0)
    parser.add_argument("--retry-count", type=int, default=3)
    parser.add_argument("--retry-backoff-seconds", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true", help="Print source URLs without fetching.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dry_run:
        print("NOAA CPC ENSO strengths:")
        print(CPC_ENSO_STRENGTHS_URL)
        print("NOAA CPC Global Tropics Hazards Outlook:")
        print(CPC_GLOBAL_TROPICS_URL)
        print("WMO ENSO update landing page:")
        print(WMO_ENSO_UPDATE_URL)
        print("INCOIS IOD monitoring:")
        print(INCOIS_IOD_URL)
        return 0

    fetch_policy = FetchPolicy(
        timeout_seconds=args.request_timeout,
        retry_count=max(1, args.retry_count),
        retry_backoff_seconds=max(0, args.retry_backoff_seconds),
    )
    fetch_results = [
        fetch_signal("NOAA CPC ENSO", CPC_ENSO_STRENGTHS_URL, parse_cpc_enso, fetch_policy),
        fetch_signal("NOAA CPC GTH", CPC_GLOBAL_TROPICS_URL, parse_global_tropics, fetch_policy),
        fetch_signal("WMO ENSO", WMO_ENSO_UPDATE_URL, parse_wmo_update, fetch_policy),
        fetch_signal("INCOIS IOD", INCOIS_IOD_URL, parse_incois_iod, fetch_policy),
    ]
    signal = combine_signals(
        [item["signal"] for item in fetch_results if item["signal"]],
        [item["sourceStatus"] for item in fetch_results],
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(signal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path} with phase {signal['phase']}")
    return 0


def fetch_signal(
    name: str,
    url: str,
    parser: Any,
    fetch_policy: FetchPolicy,
) -> dict[str, Any]:
    try:
        signal = parser(request_text(url, fetch_policy))
        if signal:
            return {
                "signal": signal,
                "sourceStatus": {
                    "name": name,
                    "url": url,
                    "status": "ok",
                },
            }
        return {
            "signal": None,
            "sourceStatus": {
                "name": name,
                "url": url,
                "status": "no_signal",
                "error": "Source fetched but no usable signal was parsed.",
            },
        }
    except Exception as exc:  # noqa: BLE001
        print(f"Skipping {name}: {exc}", file=sys.stderr)
        return {
            "signal": None,
            "sourceStatus": {
                "name": name,
                "url": url,
                "status": "failed",
                "error": str(exc),
            },
        }


def request_text(url: str, fetch_policy: FetchPolicy) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "WeatherAI-climate-signal/1.0"})
    last_error: Exception | None = None
    for attempt in range(1, fetch_policy.retry_count + 1):
        try:
            with urllib.request.urlopen(request, timeout=fetch_policy.timeout_seconds) as response:
                if response.status < 200 or response.status >= 300:
                    raise RuntimeError(f"HTTP {response.status} for {url}")
                return response.read().decode("utf-8", errors="replace")
        except (TimeoutError, urllib.error.URLError, RuntimeError) as exc:
            last_error = exc
            if attempt >= fetch_policy.retry_count:
                break
            delay = fetch_policy.retry_backoff_seconds * (2 ** (attempt - 1))
            print(f"Retrying request after {type(exc).__name__}: attempt {attempt + 1}/{fetch_policy.retry_count}", file=sys.stderr)
            time.sleep(delay)

    raise RuntimeError(f"Failed to fetch {url}: {last_error}") from last_error


def parse_cpc_enso(text: str) -> dict[str, Any] | None:
    plain = html_to_text(text)
    issued = first_match(plain, r"Issued\s+([A-Za-z]+\s+\d{4})") or "latest CPC update"
    signal = classify_enso_phase(
        {
            "el_nino": enso_probability(plain, EL_NINO_PATTERN),
            "la_nina": enso_probability(plain, LA_NINA_PATTERN),
            "neutral": enso_probability(plain, NEUTRAL_PATTERN),
        },
        source_name="NOAA CPC ENSO",
    )
    return {
        "phase": signal["phase"],
        "probability": signal["probability"],
        "updatedAt": f"NOAA CPC {issued}",
        "summary": signal["summary"],
        "uncertaintyShrinkBonus": signal["uncertaintyShrinkBonus"],
        "confidencePenalty": signal["confidencePenalty"],
        "sourceDetails": ["NOAA CPC ENSO strength probabilities", CPC_ENSO_STRENGTHS_URL],
    }


def parse_global_tropics(text: str) -> dict[str, Any] | None:
    plain = html_to_text(text)
    if "MJO" not in plain and "Global Tropics" not in plain:
        return None

    updated = first_match(plain, r"Last Updated\s*-\s*([0-9/]+)") or "latest weekly update"
    valid = first_match(plain, r"Valid\s*-\s*([0-9/]+\s*-\s*[0-9/]+)") or "current Week 2-3 outlook"
    phases = unique_matches(plain, r"Phase\s+([1-8])")
    phase = "MJO/GTH background" if not phases else f"MJO Phase {'-'.join(phases)}"
    has_cyclone = contains_any(plain, ["tropical cyclone", "cyclogenesis"])
    has_rain = contains_any(plain, ["above-normal rainfall", "above-median rainfall", "enhanced precipitation"])
    summary_parts = [f"NOAA CPC GTH 週尺度展望有效期 {valid}"]
    if has_rain:
        summary_parts.append("含熱帶降雨背景訊號")
    if has_cyclone:
        summary_parts.append("含熱帶氣旋生成背景訊號")
    return {
        "phase": phase,
        "probability": 0.62 if phases else 0.45,
        "updatedAt": f"NOAA CPC GTH {updated}",
        "summary": "，".join(summary_parts) + "。",
        "uncertaintyShrinkBonus": 0.03 if has_rain or has_cyclone else 0.015,
        "confidencePenalty": 3 if has_rain or has_cyclone else 1,
        "sourceDetails": ["NOAA CPC Global Tropics Hazards Outlook", CPC_GLOBAL_TROPICS_URL],
    }


def parse_wmo_update(text: str) -> dict[str, Any] | None:
    plain = html_to_text(text)
    if not contains_any(plain, ["El Niño", "El Nino", "La Niña", "La Nina", "ENSO"]):
        return None

    date_text = first_match(plain, r"Publication\s+([0-9]{1,2}\s+[A-Za-z]+\s+\d{4})") or "latest WMO update"
    signal = classify_enso_phase(
        {
            "el_nino": enso_probability(plain, EL_NINO_PATTERN),
            "la_nina": enso_probability(plain, LA_NINA_PATTERN),
            "neutral": enso_probability(plain, NEUTRAL_PATTERN),
        },
        source_name="WMO ENSO",
    )
    return {
        "phase": signal["phase"].replace("Watch", "Update"),
        "probability": signal["probability"],
        "updatedAt": f"WMO {date_text}",
        "summary": signal["summary"].replace("WMO ENSO", "WMO 最新 ENSO 更新"),
        "uncertaintyShrinkBonus": min(0.02, signal["uncertaintyShrinkBonus"]),
        "confidencePenalty": min(2, signal["confidencePenalty"]),
        "sourceDetails": ["WMO El Niño/La Niña Updates", WMO_ENSO_UPDATE_URL],
    }


def parse_incois_iod(text: str) -> dict[str, Any] | None:
    plain = html_to_text(text)
    if not contains_any(plain, ["Status Of IOD", "Indian Ocean Dipole", "Dipole mode Index", "Dipole Mode Index"]):
        return None

    return {
        "phase": "IOD monitoring background",
        "probability": 0.45,
        "updatedAt": "INCOIS latest IOD monitoring",
        "summary": "INCOIS-GODAS 提供印度洋偶極 IOD/DMI 監測，作為印度洋季節降雨背景訊號；目前僅低權重納入，不直接改寫逐小時天氣。",
        "uncertaintyShrinkBonus": 0.01,
        "confidencePenalty": 1,
        "sourceDetails": ["INCOIS-GODAS Indian Ocean Dipole monitoring", INCOIS_IOD_URL],
    }


def combine_signals(
    signals: list[dict[str, Any]],
    source_status: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    generated_at = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    if not signals:
        return {
            "phase": "El Niño Watch",
            "probability": 0.82,
            "updatedAt": "NOAA CPC 2026-05-14 / WMO 2026-06-02",
            "summary": "ENSO 中性但聖嬰發展機率升高，僅作季節背景校準，不直接決定逐小時天氣。",
            "uncertaintyShrinkBonus": 0.03,
            "confidencePenalty": 3,
            "sourceDetails": ["static fallback"],
            "sourceStatus": source_status or [],
            "generatedAt": generated_at,
            "generator": "tools/generate_climate_signal.py",
        }

    return {
        "phase": " + ".join(signal["phase"] for signal in signals),
        "probability": min(0.99, max(signal["probability"] for signal in signals)),
        "updatedAt": " / ".join(signal["updatedAt"] for signal in signals),
        "summary": " ".join(signal["summary"] for signal in signals),
        "uncertaintyShrinkBonus": min(0.07, sum(signal["uncertaintyShrinkBonus"] for signal in signals)),
        "confidencePenalty": min(7, sum(signal["confidencePenalty"] for signal in signals)),
        "sourceDetails": [detail for signal in signals for detail in signal["sourceDetails"]],
        "sourceStatus": source_status or [],
        "generatedAt": generated_at,
        "generator": "tools/generate_climate_signal.py",
    }


def html_to_text(value: str) -> str:
    value = re.sub(r"(?i)<br\s*/?>", "\n", value)
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    value = re.sub(r"[ \t\r\f\v]+", " ", value)
    value = re.sub(r"\n\s+", "\n", value)
    return value


def first_match(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text, re.I)
    return match.group(1).strip() if match else None


def unique_matches(text: str, pattern: str) -> list[str]:
    seen: set[str] = set()
    values: list[str] = []
    for match in re.finditer(pattern, text, re.I):
        value = match.group(1).strip()
        if value not in seen:
            seen.add(value)
            values.append(value)
    return values


def contains_any(text: str, needles: list[str]) -> bool:
    lower = text.lower()
    return any(needle.lower() in lower for needle in needles)


def enso_probability(text: str, term_pattern: str) -> float | None:
    values = [int(value) for value in re.findall(term_pattern + r"[^0-9]{0,160}(\d{1,3})\s*%", text, re.I)]
    values += [int(value) for value in re.findall(r"(\d{1,3})\s*%[^%\n]{0,160}" + term_pattern, text, re.I)]
    if not values:
        return None
    return clamp(max(values) / 100.0, 0.0, 0.99)


def classify_enso_phase(probabilities: dict[str, float | None], source_name: str) -> dict[str, Any]:
    available = {key: value for key, value in probabilities.items() if value is not None}
    if not available:
        return {
            "phase": "ENSO background",
            "probability": 0.45,
            "summary": f"{source_name} 未提供可解析的聖嬰/反聖嬰機率，僅作低權重季節背景訊號。",
            "uncertaintyShrinkBonus": 0.01,
            "confidencePenalty": 1,
        }

    phase_key, probability = max(available.items(), key=lambda item: item[1])
    if phase_key == "el_nino" and probability >= 0.5:
        return {
            "phase": "El Niño Watch",
            "probability": probability,
            "summary": f"{source_name} 機率顯示聖嬰風險偏高，作為季節背景校準。",
            "uncertaintyShrinkBonus": 0.03,
            "confidencePenalty": 3,
        }
    if phase_key == "la_nina" and probability >= 0.5:
        return {
            "phase": "La Niña Watch",
            "probability": probability,
            "summary": f"{source_name} 機率顯示反聖嬰風險偏高，作為季節背景校準。",
            "uncertaintyShrinkBonus": 0.03,
            "confidencePenalty": 3,
        }

    neutral_probability = available.get("neutral", probability)
    return {
        "phase": "ENSO Neutral",
        "probability": neutral_probability,
        "summary": f"{source_name} 未顯示明確聖嬰或反聖嬰主導，作為低權重背景訊號。",
        "uncertaintyShrinkBonus": 0.01,
        "confidencePenalty": 1,
    }


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


if __name__ == "__main__":
    sys.exit(main())
