# Data Collection Guide

This project uses **hybrid automation bootstrapping** to generate high-quality training signals with minimal manual effort.

## Prerequisites

- Vercel account with an existing repository (e.g., `SushanthOO7/reactapp`).
- Playwright installed (`pip install playwright` then `playwright install chromium`).
- Logged-in storage state captured once via `playwright codegen` or `npx playwright login`:

```bash
playwright codegen https://vercel.com/dashboard --save-storage storage_state.json
```

## Recording a trajectory

```bash
python scripts/capture_session.py SushanthOO7/reactapp data/captures/run1 --storage-state storage_state.json
```

This script:

1. Launches Chromium with your saved cookies.
2. Steps through the Vercel workflow (Add → Project → search → Import → Deploy).
3. Saves full-page screenshots and a `trajectory.json` file containing DOM bounding boxes.

Repeat this command 5–10 times (vary repo names, scrolling offsets, etc.) to diversify the dataset.

## Building annotations

After recording, merge every run into a manifest:

```bash
python - <<'PY'
from pathlib import Path
from src.data_pipeline.labeling import merge_annotation_runs
roots = [Path('data/captures/run1'), Path('data/captures/run2')]
merge_annotation_runs(roots, viewport_width=1920, viewport_height=1080, output_path=Path('artifacts/annotations/vercel.json'))
PY
```

The resulting JSON is the single source of truth for training.

### Verify coordinates quickly

If you want to confirm that the bounding boxes align with the screenshots, run:

```bash
python scripts/verify_annotations.py data/captures/run1 --index 0 --output overlays/add_new.png
```

This draws the recorded box on top of the stored frame so you can visually confirm alignment before training.

## Tips

- Keep the browser window at 1920×1080 so normalized coordinates stay consistent.
- If the UI layout changes (dark/light mode, banners), capture a few extra runs to help the model adapt.
- For larger datasets, script repo permutations and environment variable toggles before invoking the capture script.
