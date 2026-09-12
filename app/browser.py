"""Playwright browser automation module for the Autonomous Lead Enrichment Agent.

Provides reusable headless Chromium browser management, safe page rendering with
JavaScript support, timeouts, and resource cleanup.
"""

from contextlib import contextmanager
from typing import Generator
from playwright.sync_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    TimeoutError as PlaywrightTimeoutError,
    Error as PlaywrightError,
    sync_playwright,
)


DEFAULT_TIMEOUT_MS = 25000
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


@contextmanager
def get_browser_session(headless: bool = True) -> Generator[Browser, None, None]:
    """Context manager for managing the Playwright Chromium browser lifecycle.

    Ensures that both the browser and Playwright process are cleaned up even if
    unhandled exceptions occur.

    Args:
        headless: Whether to run Chromium in headless mode. Defaults to True.

    Yields:
        An active Playwright Browser instance.
    """
    playwright: Playwright | None = None
    browser: Browser | None = None
    try:
        playwright = sync_playwright().start()
        browser = playwright.chromium.launch(
            headless=headless,
            args=[
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--disable-gpu",
            ],
        )
        yield browser
    finally:
        if browser:
            try:
                browser.close()
            except Exception:
                pass
        if playwright:
            try:
                playwright.stop()
            except Exception:
                pass


def fetch_page_content(
    browser: Browser,
    url: str,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    wait_after_load_ms: int = 1500,
) -> tuple[str | None, int | None, str | None]:
    """Navigates to a URL with JavaScript execution enabled and returns the rendered HTML.

    Creates an isolated BrowserContext and Page for each request to avoid state leakage,
    waits for DOM content and client hydration, and guarantees resource cleanup.

    Args:
        browser: Active Playwright Browser instance.
        url: Absolute URL to navigate to.
        timeout_ms: Navigation and execution timeout in milliseconds.
        wait_after_load_ms: Brief pause after DOMContentLoaded for client hydration.

    Returns:
        Tuple of (rendered_html, status_code, error_message).
        If successful, error_message is None. If failed, rendered_html is None.
    """
    context: BrowserContext | None = None
    page: Page | None = None

    try:
        context = browser.new_context(
            user_agent=DEFAULT_USER_AGENT,
            viewport={"width": 1280, "height": 800},
            ignore_https_errors=True,
        )
        page = context.new_page()

        response = page.goto(
            url,
            wait_until="domcontentloaded",
            timeout=timeout_ms,
        )

        status_code = response.status if response else None

        # Allow client-side frameworks (React, Next.js, Vue) to complete hydration
        if wait_after_load_ms > 0:
            page.wait_for_timeout(wait_after_load_ms)

        html_content = page.content()
        return html_content, status_code, None

    except PlaywrightTimeoutError:
        return None, None, f"Timeout after {timeout_ms}ms"
    except PlaywrightError as pe:
        return None, None, f"Playwright navigation error: {str(pe).splitlines()[0]}"
    except Exception as ex:
        return None, None, f"Unexpected browser error: {str(ex)}"
    finally:
        if page:
            try:
                page.close()
            except Exception:
                pass
        if context:
            try:
                context.close()
            except Exception:
                pass
