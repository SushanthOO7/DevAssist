"""Capture pixel-level trajectories on Vercel using Playwright automation."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

from playwright.sync_api import BrowserContext, Page, Locator, sync_playwright


@dataclass
class CaptureConfig:
    """Runtime parameters for recording a deployment trajectory."""

    repo_name: str
    output_dir: Path
    storage_state: Optional[Path] = None
    viewport_width: int = 1920
    viewport_height: int = 1080
    headless: bool = False
    wait_after_action: float = 0.8
    use_system_viewport: bool = True

    @property
    def viewport(self) -> Dict[str, int]:
        return {"width": self.viewport_width, "height": self.viewport_height}


@dataclass
class ActionRecord:
    """Single supervised sample describing one UI action."""

    step: str
    action_type: str
    target_selector: str
    timestamp: float
    screenshot: str
    bounding_box: Optional[Dict[str, float]] = None
    typed_text: Optional[str] = None


class VercelFlowCapturer:
    """Automates the Vercel deployment creation UI and records supervision."""

    DASHBOARD_URL = "https://vercel.com/dashboard"

    def __init__(self, config: CaptureConfig) -> None:
        self.config = config
        self.records: List[ActionRecord] = []
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Path:
        """Executes the full capture flow and returns the JSON log path."""
        with sync_playwright() as p:
            launch_args = {"headless": self.config.headless}
            if not self.config.headless:
                launch_args["args"] = [
                    "--start-maximized",
                    "--window-position=0,0",
                    f"--window-size={self.config.viewport_width},{self.config.viewport_height}",
                    "--disable-features=IsolateOrigins,site-per-process",
                ]
            browser = p.chromium.launch(**launch_args)
            context = self._build_context(browser)
            page = context.new_page()
            page.goto(self.DASHBOARD_URL, wait_until="domcontentloaded")
            page.wait_for_timeout(1500)

            self._click_add_new(page)
            self._click_project_button(page)
            self._search_repo(page)
            self._click_import(page)
            self._scroll_and_deploy(page)

            log_path = self._flush_records()
            context.close()
            browser.close()
            return log_path

    def _build_context(self, browser) -> BrowserContext:  # type: ignore[override]
        args = {
            "viewport": self.config.viewport,
            "device_scale_factor": 1.0,
        }
        if self.config.storage_state:
            args["storage_state"] = str(self.config.storage_state)
        return browser.new_context(**args)

    def _snapshot(
        self,
        page: Page,
        *,
        step: str,
        action_type: str,
        locator: Optional[Locator] = None,
        selector_desc: Optional[str] = None,
        typed_text: Optional[str] = None,
        bbox_override: Optional[Dict[str, float]] = None,
    ) -> None:
        file_name = f"{time.time()}_{step}.png"
        path = self.output_dir / file_name
        page.screenshot(path=str(path), full_page=False, scale="css")
        bbox = None
        bbox = None
        if bbox_override is not None:
            bbox = bbox_override
        elif locator is not None:
            try:
                bbox = locator.bounding_box()
            except Exception:
                bbox = None
        self.records.append(
            ActionRecord(
                step=step,
                action_type=action_type,
                target_selector=selector_desc or (locator.to_string() if locator else "unknown"),
                timestamp=time.time(),
                screenshot=file_name,
                bounding_box=bbox,
                typed_text=typed_text,
            )
        )
        page.wait_for_timeout(self.config.wait_after_action * 1000)

    def _click_add_new(self, page: Page) -> None:
        button = page.get_by_role("button", name="Add New")
        bbox = button.bounding_box()
        self._snapshot(page, step="add_new", action_type="click", locator=button, selector_desc="button:Add New", bbox_override=bbox)
        button.click()

    def _click_project_button(self, page: Page) -> None:
        project_btn = page.get_by_role("menuitem", name="Project")
        bbox = project_btn.bounding_box()
        self._snapshot(page, step="project_menu", action_type="click", locator=project_btn, selector_desc="menuitem:Project", bbox_override=bbox)
        project_btn.click()

    def _search_repo(self, page: Page) -> None:
        search_box = page.locator("[data-testid='import-flow-layout/search-input']")
        search_box.wait_for(state="visible", timeout=15000)
        bbox = search_box.bounding_box()
        self._snapshot(
            page,
            step="search_repo",
            action_type="type",
            locator=search_box,
            selector_desc="search-input",
            typed_text=self.config.repo_name,
            bbox_override=bbox,
        )
        search_box.click()
        search_box.fill(self.config.repo_name)

    def _click_import(self, page: Page) -> None:
        button = page.locator(
            "[data-testid='import-flow-layout/suggestion-card/suggestion/import-button']"
        ).first
        button.wait_for(state="visible", timeout=15000)
        bbox = button.bounding_box()
        self._snapshot(
            page,
            step="import",
            action_type="click",
            locator=button,
            selector_desc="import-button",
            bbox_override=bbox,
        )
        button.click()

    def _scroll_and_deploy(self, page: Page) -> None:
        body = page.locator("body")
        bbox_body = body.bounding_box()
        self._snapshot(page, step="scroll", action_type="scroll", locator=body, selector_desc="body", bbox_override=bbox_body)
        page.mouse.wheel(0, 400)
        deploy_button = self._button_by_span_text(page, "Deploy")
        bbox = deploy_button.bounding_box()
        self._snapshot(page, step="deploy", action_type="click", locator=deploy_button, selector_desc="button:Deploy", bbox_override=bbox)
        deploy_button.click()

    @staticmethod
    def _button_by_span_text(page: Page, label: str):
        span = page.locator("span.button-module__QyrFCa__content", has_text=label).first
        if not span.count():
            # fallback to accessible role in case class names change
            return page.get_by_role("button", name=label)
        return span.locator("xpath=ancestor::button[1]")

    def _flush_records(self) -> Path:
        log_path = self.output_dir / "trajectory.json"
        serializable = [asdict(record) for record in self.records]
        with log_path.open("w", encoding="utf-8") as fp:
            json.dump(serializable, fp, indent=2)
        return log_path


def capture_vercel_flow(config: CaptureConfig) -> Path:
    """Public helper for scripts."""
    capturer = VercelFlowCapturer(config)
    return capturer.run()
