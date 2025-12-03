"""Rule-based helpers to stabilize automation."""
from __future__ import annotations

from playwright.sync_api import Page


def wait_for_idle(page: Page, timeout_ms: int = 2000) -> None:
    page.wait_for_timeout(timeout_ms)


def ensure_modal_closed(page: Page) -> None:
    try:
        modal = page.locator("[role='dialog'] button:has-text('Close')")
        if modal.is_visible():
            modal.click()
    except Exception:
        return
