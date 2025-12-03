"""Playwright-based executor that replays policy decisions on Vercel."""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from playwright.sync_api import sync_playwright

from src.inference.policy import PixelPolicy
from src.inference import recovery


@dataclass
class RunnerConfig:
    checkpoint: Path
    repo_name: str
    viewport_width: int = 1920
    viewport_height: int = 1080
    max_steps: int = 12
    headless: bool = False
    use_chrome: bool = False
    start_maximized: bool = False
    fullscreen: bool = False
    storage_state: Path | None = None
    debug: bool = False
    screenshot_dir: Path | None = None


class AutomationRunner:
    def __init__(self, config: RunnerConfig) -> None:
        self.config = config
        self.policy = PixelPolicy(config.checkpoint, ["click", "scroll", "type", "wait"])
        # Track deterministic UI stages so we can short-circuit once Deploy is clicked.
        self._guided_stage = "init"

    def execute(self) -> None:
        with sync_playwright() as p:
            tmp_ctx = None
            if self.config.screenshot_dir is None:
                tmp_ctx = tempfile.TemporaryDirectory()
                capture_root = Path(tmp_ctx.name)
            else:
                self.config.screenshot_dir.mkdir(parents=True, exist_ok=True)
                capture_root = self.config.screenshot_dir
            launch_kwargs = {"headless": self.config.headless}
            if self.config.use_chrome:
                launch_kwargs["channel"] = "chrome"
            window_mode = (self.config.start_maximized or self.config.fullscreen) and not self.config.headless
            if not self.config.headless:
                args = []
                if window_mode:
                    args.append("--start-maximized")
                else:
                    args.extend(
                        [
                            "--window-position=0,0",
                            f"--window-size={self.config.viewport_width},{self.config.viewport_height}",
                        ]
                    )
                launch_kwargs["args"] = args
            browser = p.chromium.launch(**launch_kwargs)
            if window_mode:
                context_args = {"viewport": None, "no_viewport": True}
            else:
                context_args = {
                    "viewport": {
                        "width": self.config.viewport_width,
                        "height": self.config.viewport_height,
                    }
                }
            if self.config.storage_state:
                context_args["storage_state"] = str(self.config.storage_state)
            context = browser.new_context(**context_args)
            page = context.new_page()
            if window_mode:
                self._apply_window_state(browser, page)
                self._sync_viewport(page)
            page.goto("https://vercel.com/dashboard", wait_until="load")
            step = 0
            guided_complete = False
            while step < self.config.max_steps:
                recovery.ensure_modal_closed(page)
                if self._guided_flow(page):
                    recovery.wait_for_idle(page)
                    if self._guided_stage == "done":
                        guided_complete = True
                        if self.config.debug:
                            print("Guided workflow finished; stopping early.")
                        break
                    continue
                screenshot = capture_root / f"step_{step}.png"
                page.screenshot(path=str(screenshot), full_page=False, scale="css")
                decision = self.policy.act(screenshot)
                if self.config.debug:
                    print(
                        f"step {step}: action={decision.action} coords={decision.coords} text={decision.text}"
                    )
                self._apply_decision(page, decision)
                recovery.wait_for_idle(page)
                step += 1
            if not guided_complete and self.config.debug:
                print("Guided workflow did not finish; exhausted policy steps.")
            context.close()
            browser.close()
            if tmp_ctx:
                tmp_ctx.cleanup()

    def _apply_decision(self, page, decision) -> None:
        width, height = self.config.viewport_width, self.config.viewport_height
        x = decision.coords[0] * width
        y = decision.coords[1] * height
        if decision.action == "click":
            page.mouse.click(x, y)
        elif decision.action == "scroll":
            page.mouse.wheel(0, height * 0.2)
        elif decision.action == "type":
            page.mouse.click(x, y)
            page.keyboard.press("Control+A")
            page.keyboard.type(self.config.repo_name)
        elif decision.action == "wait":
            recovery.wait_for_idle(page, 500)

    def _guided_flow(self, page) -> bool:
        if self._guided_stage == "done":
            return False
        try:
            if self._guided_stage == "init":
                add_new = page.get_by_role("button", name="Add New")
                if add_new.is_visible():
                    add_new.click()
                    self._guided_stage = "project"
                    return True
            if self._guided_stage == "project":
                project_btn = page.get_by_role("menuitem", name="Project")
                if project_btn.is_visible():
                    project_btn.click()
                    self._guided_stage = "search"
                    return True
            if self._guided_stage == "search":
                search_box = page.locator("[data-testid='import-flow-layout/search-input']")
                if search_box.is_visible():
                    current = search_box.input_value().strip()
                    if current != self.config.repo_name:
                        search_box.click()
                        search_box.fill(self.config.repo_name)
                        return True
                    self._guided_stage = "import"
            if self._guided_stage == "import":
                import_btn = page.locator(
                    "[data-testid='import-flow-layout/suggestion-card/suggestion/import-button']"
                ).first
                if import_btn.is_visible():
                    import_btn.click()
                    self._guided_stage = "deploy"
                    return True
            if self._guided_stage == "deploy":
                page.mouse.wheel(0, self.config.viewport_height * 0.5)
                deploy_btn = page.get_by_role("button", name="Deploy")
                if deploy_btn.is_visible():
                    deploy_btn.click()
                    self._guided_stage = "done"
                    return True
            return False
        except Exception as exc:
            if self.config.debug:
                print(f"Guided flow error at stage {self._guided_stage}: {exc}")
            return False

    def _apply_window_state(self, browser, page) -> None:
        state = "fullscreen" if self.config.fullscreen else "maximized"
        try:
            session = browser.new_browser_cdp_session(page)
            info = session.send("Browser.getWindowForTarget")
            session.send("Browser.setWindowBounds", {"windowId": info["windowId"], "bounds": {"windowState": state}})
        except Exception as exc:
            if self.config.debug:
                print(f"Failed to set window state {state}: {exc}")

    def _sync_viewport(self, page) -> None:
        try:
            dims = page.evaluate("({width: window.innerWidth, height: window.innerHeight})")
            self.config.viewport_width = int(dims.get("width", self.config.viewport_width))
            self.config.viewport_height = int(dims.get("height", self.config.viewport_height))
        except Exception as exc:
            if self.config.debug:
                print(f"Failed to sync viewport: {exc}")
