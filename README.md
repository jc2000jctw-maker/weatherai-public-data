# WeatherAI Public Data

This repository hosts the public WeatherAI data bundle for GitHub Pages.

Published files:

- `public/weatherai-data/manifest.json`
- `public/weatherai-data/model-calibration.json`
- `public/weatherai-data/climate-signal.json`
- `public/weatherai-data/health.json`

The GitHub Actions workflow refreshes the bundle daily from public weather and climate sources, validates checksums and quality gates, then deploys `public/` to GitHub Pages.

Refresh profiles:

- `daily`: 6 representative cities, one per production region, optimized for timely scheduled updates.
- `full`: 18 representative cities, used for weekly/manual higher-coverage calibration.
