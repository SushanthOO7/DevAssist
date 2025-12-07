# DevAssist: Vercel Deployment Agent

This repo houses a lightweight pixel-policy agent that automates the Vercel project deployment flow:

1. Click **Add New…**
2. Select **Project**
3. Search for the target GitHub repo
4. Click **Import** on the matching repo
5. Scroll if needed and press **Deploy**

The agent trains on short screen trajectories collected from the real Vercel dashboard and executes actions through Playwright.

## Workflow

1. **Hybrid automation bootstrapping** (Playwright + screenshots) collects 5–10 high-quality trajectories quickly while preserving pixel-only training.
2. It fits a ViT-Tiny encoder with an action decoder for click heatmaps, action class, and text tokens.
3. It loads the trained checkpoint and drives a headless (or headed) Chromium instance through the deployment steps.
