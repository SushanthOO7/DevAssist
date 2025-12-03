# DevAssist: Vercel Deployment Agent

This repo houses a lightweight pixel-policy agent that automates the Vercel project deployment flow:

1. Click **Add New…**
2. Select **Project**
3. Search for the target GitHub repo
4. Click **Import** on the matching repo
5. Scroll if needed and press **Deploy**

The agent trains on short screen trajectories collected from the real Vercel dashboard and executes actions through Playwright.

## Repository layout

```
DevAssist/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml
├── scripts/
│   ├── capture_session.py
│   ├── train_agent.py
│   └── run_vercel_workflow.py
└── src/
    ├── data_pipeline/
    │   ├── capture.py
    │   ├── labeling.py
    │   └── dataset.py
    ├── training/
    │   ├── model.py
    │   ├── losses.py
    │   └── trainer.py
    └── inference/
        ├── policy.py
        ├── runner.py
        └── recovery.py
```

## Workflow

1. **Hybrid automation bootstrapping** (Playwright + screenshots) collects 5–10 high-quality trajectories quickly while preserving pixel-only training.
2. `scripts/train_agent.py` fits a ViT-Tiny encoder with an action decoder for click heatmaps, action class, and text tokens.
3. `scripts/run_vercel_workflow.py` loads the trained checkpoint and drives a headless (or headed) Chromium instance through the deployment steps.

Detailed instructions for data capture, annotation, training, and inference will be captured in the `/docs` section once code is finalized.
