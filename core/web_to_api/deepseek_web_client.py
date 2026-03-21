from __future__ import annotations

import atexit
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from log import get_logger

DEFAULT_CHAT_URL = "https://chat.deepseek.com/"
DEFAULT_INPUT_SELECTORS = (
    'textarea',
    'textarea[placeholder*="DeepSeek"]',
    'textarea[placeholder*="Send"]',
    'div[contenteditable="true"]',
)
DEFAULT_SEND_BUTTON_SELECTORS = (
    'button[type="submit"]',
    'button[aria-label*="Send"]',
    'button[aria-label*="send"]',
    'button:has-text("Send")',
)
DEFAULT_ASSISTANT_MESSAGE_SELECTORS = (
    '[data-message-author-role="assistant"]',
    '[data-role="assistant"]',
    '[class*="assistant"] [class*="markdown"]',
    '.ds-markdown',
    'main article',
)
DEFAULT_LOADING_SELECTORS = (
    'button:has-text("Stop")',
    '[aria-label*="Stop"]',
    '[class*="loading"]',
    '[class*="typing"]',
)
DEFAULT_NEW_CHAT_SELECTORS = (
    'button:has-text("New chat")',
    'button:has-text("New Chat")',
    'a:has-text("New chat")',
    'a:has-text("New Chat")',
    'button:has-text("新对话")',
    'button:has-text("新建对话")',
    'a:has-text("新对话")',
    'a:has-text("新建对话")',
)


@dataclass
class DeepSeekWebClientConfig:
    chat_url: str = DEFAULT_CHAT_URL
    user_data_dir: str = ".playwright/deepseek_web"
    browser_type: str = "chromium"
    browser_channel: Optional[str] = None
    headless: bool = False
    viewport_width: int = 1440
    viewport_height: int = 960
    input_ready_timeout_ms: int = 120000
    navigation_timeout_ms: int = 60000
    reply_timeout_ms: int = 180000
    poll_interval_ms: int = 1000
    stable_rounds: int = 3
    input_selectors: tuple[str, ...] = field(default_factory=lambda: tuple(DEFAULT_INPUT_SELECTORS))
    send_button_selectors: tuple[str, ...] = field(default_factory=lambda: tuple(DEFAULT_SEND_BUTTON_SELECTORS))
    assistant_message_selectors: tuple[str, ...] = field(
        default_factory=lambda: tuple(DEFAULT_ASSISTANT_MESSAGE_SELECTORS)
    )
    loading_selectors: tuple[str, ...] = field(default_factory=lambda: tuple(DEFAULT_LOADING_SELECTORS))
    keep_alive_url: str = "about:blank"
    new_chat_url: Optional[str] = None
    new_chat_timeout_ms: int = 5000
    new_chat_selectors: tuple[str, ...] = field(default_factory=lambda: tuple(DEFAULT_NEW_CHAT_SELECTORS))


class DeepSeekWebClient:
    TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*\{.*?\}\s*</tool_call>", re.S)
    FINAL_BLOCK_RE = re.compile(r"<final>\s*.*?\s*</final>", re.S)

    # Initialize the persistent Playwright client state.
    def __init__(self, config: DeepSeekWebClientConfig):
        self._config = config
        self._playwright = None
        self._context = None
        self._page = None
        self._lock = threading.RLock()
        atexit.register(self.close)

    # Open a fresh chat page, send the prompt, and return the reply.
    def ask(self, prompt: str) -> str:
        logger = get_logger()
        with self._lock:
            page, timeout_error_cls = self._ensure_page()
            target_url = self._config.new_chat_url or self._config.chat_url

            logger.info("Open DeepSeek chat page: {}", target_url)
            page.goto(target_url, wait_until="domcontentloaded")
            self._maybe_start_new_chat(page)

            input_box = self._wait_for_input_box(page, timeout_error_cls)
            previous_reply = self._extract_latest_reply_text(page)
            if previous_reply:
                logger.warning(
                    "DeepSeek web page already contains assistant content before sending the prompt. "
                    "If you need API-like stateless behavior, set deepseek_web.new_chat_url or "
                    "deepseek_web.new_chat_selectors."
                )
            logger.info("Previous reply preview: {}", self._preview_text(previous_reply))

            self._fill_prompt(page, input_box, prompt)
            self._send_prompt(page, input_box)
            reply = self._wait_for_reply(page, previous_reply)
            reply = self._normalize_reply_text(reply)
            logger.info("DeepSeek web reply preview: {}", self._preview_text(reply))
            return reply

    # Close the cached Playwright page, context, and driver.
    def close(self):
        with self._lock:
            page = self._page
            context = self._context
            playwright = self._playwright
            self._page = None
            self._context = None
            self._playwright = None

        try:
            if page is not None and not page.is_closed():
                page.close()
        except Exception:
            pass
        try:
            if context is not None:
                context.close()
        except Exception:
            pass
        try:
            if playwright is not None:
                playwright.stop()
        except Exception:
            pass

    # Resolve the browser profile directory to an absolute path.
    def _resolve_user_data_dir(self) -> Path:
        path = Path(self._config.user_data_dir)
        if path.is_absolute():
            return path
        # Keep relative profile paths stable regardless of the current working directory.
        return self._project_root() / path

    # Return the project root used for relative profile paths.
    @staticmethod
    def _project_root() -> Path:
        return Path(__file__).resolve().parents[2]

    # Reuse or create a live page for browser automation.
    def _ensure_page(self):
        if self._has_live_page():
            return self._page, self._timeout_error_class()

        context = self._ensure_context()
        page = context.pages[0] if context.pages else context.new_page()
        page.set_default_timeout(self._config.navigation_timeout_ms)
        self._page = page
        return page, self._timeout_error_class()

    # Reuse or launch the persistent browser context.
    def _ensure_context(self):
        if self._has_live_context():
            return self._context

        logger = get_logger()
        user_data_dir = self._resolve_user_data_dir()
        user_data_dir.mkdir(parents=True, exist_ok=True)
        logger.info("DeepSeek web user data dir: {}", str(user_data_dir))

        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError(
                "Playwright is not installed. Run `pip install playwright` and `playwright install chromium`."
            ) from exc

        playwright = sync_playwright().start()
        browser_launcher = getattr(playwright, self._config.browser_type, None)
        if browser_launcher is None:
            playwright.stop()
            raise ValueError(f"Unsupported browser_type: {self._config.browser_type}")

        launch_kwargs = {
            "user_data_dir": str(user_data_dir),
            "headless": self._config.headless,
            "viewport": {
                "width": self._config.viewport_width,
                "height": self._config.viewport_height,
            },
        }
        if self._config.browser_channel:
            launch_kwargs["channel"] = self._config.browser_channel

        try:
            context = browser_launcher.launch_persistent_context(**launch_kwargs)
        except Exception as exc:
            playwright.stop()
            raise RuntimeError(
                "Failed to launch DeepSeek web browser profile at "
                f"{user_data_dir}. The profile may be locked or corrupted. "
                "Try closing other Chrome/Playwright processes or set "
                "`deepseek_web.user_data_dir` to a fresh directory."
            ) from exc

        self._playwright = playwright
        self._context = context
        self._page = context.pages[0] if context.pages else context.new_page()
        self._page.set_default_timeout(self._config.navigation_timeout_ms)
        try:
            if self._config.keep_alive_url:
                self._page.goto(self._config.keep_alive_url, wait_until="domcontentloaded")
        except Exception:
            pass
        return context

    # Check whether the cached browser context is still usable.
    def _has_live_context(self) -> bool:
        if self._context is None:
            return False
        try:
            self._context.pages
            return True
        except Exception:
            return False

    # Check whether the cached browser page is still usable.
    def _has_live_page(self) -> bool:
        if self._page is None:
            return False
        try:
            return not self._page.is_closed()
        except Exception:
            return False

    # Load Playwright's timeout exception type.
    def _timeout_error_class(self):
        try:
            from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        except ImportError as exc:
            raise RuntimeError(
                "Playwright is not installed. Run `pip install playwright` and `playwright install chromium`."
            ) from exc
        return PlaywrightTimeoutError

    # Try to click the new chat control before sending.
    def _maybe_start_new_chat(self, page):
        deadline = time.monotonic() + (self._config.new_chat_timeout_ms / 1000)
        while time.monotonic() < deadline:
            button = self._find_first_visible(page, self._config.new_chat_selectors)
            if button is None:
                page.wait_for_timeout(250)
                continue
            try:
                button.click()
                page.wait_for_timeout(500)
                return True
            except Exception:
                page.wait_for_timeout(250)
        return False

    # Wait until an input box becomes available.
    def _wait_for_input_box(self, page, timeout_error_cls):
        deadline = time.monotonic() + (self._config.input_ready_timeout_ms / 1000)
        last_error = None
        while time.monotonic() < deadline:
            locator = self._find_first_visible(page, self._config.input_selectors)
            if locator is not None:
                return locator
            page.wait_for_timeout(1000)
        raise timeout_error_cls(
            "Unable to find the DeepSeek input box. Check login state or adjust input selectors."
        ) from last_error

    # Return the first visible locator that matches the selectors.
    def _find_first_visible(self, page, selectors: Iterable[str]):
        for selector in selectors:
            locator = page.locator(selector)
            count = locator.count()
            if count == 0:
                continue
            for index in range(count):
                candidate = locator.nth(index)
                try:
                    if candidate.is_visible():
                        return candidate
                except Exception:
                    continue
        return None

    # Fill the prompt into the chat input.
    def _fill_prompt(self, page, input_box, prompt: str):
        try:
            input_box.click()
            input_box.fill(prompt)
            return
        except Exception:
            pass

        input_box.click()
        try:
            page.keyboard.press("Control+A")
            page.keyboard.press("Backspace")
        except Exception:
            pass
        page.keyboard.insert_text(prompt)

    # Submit the current prompt through the UI.
    def _send_prompt(self, page, input_box):
        send_button = self._find_send_button(page, input_box)
        if send_button is not None:
            send_button.click()
            return
        input_box.press("Enter")

    # Find the best visible send button near the input.
    def _find_send_button(self, page, input_box):
        try:
            form = input_box.locator("xpath=ancestor::form[1]")
            button = self._find_first_visible(form, self._config.send_button_selectors)
            if button is not None:
                return button
        except Exception:
            pass
        return self._find_first_visible(page, self._config.send_button_selectors)

    # Wait until the assistant reply stabilizes.
    def _wait_for_reply(self, page, previous_reply: str) -> str:
        deadline = time.monotonic() + (self._config.reply_timeout_ms / 1000)
        last_text = ""
        stable_hits = 0
        seen_new_reply = False

        while time.monotonic() < deadline:
            current = self._extract_latest_reply_text(page)
            has_loading = self._has_loading_indicator(page)

            if current and current != previous_reply:
                seen_new_reply = True
                if current == last_text and not has_loading:
                    stable_hits += 1
                else:
                    stable_hits = 0
                    last_text = current

                if stable_hits >= self._config.stable_rounds:
                    return current.strip()

            page.wait_for_timeout(self._config.poll_interval_ms)

        if seen_new_reply and last_text:
            return last_text.strip()

        raise TimeoutError("Timed out while waiting for the DeepSeek web reply.")

    # Check whether the page is still generating a reply.
    def _has_loading_indicator(self, page) -> bool:
        indicator = self._find_first_visible(page, self._config.loading_selectors)
        return indicator is not None

    # Extract the latest visible assistant message text.
    def _extract_latest_reply_text(self, page) -> str:
        texts = []
        for selector in self._config.assistant_message_selectors:
            locator = page.locator(selector)
            count = locator.count()
            for index in range(count):
                node = locator.nth(index)
                try:
                    if not node.is_visible():
                        continue
                    text = node.inner_text().strip()
                except Exception:
                    continue
                if text:
                    texts.append(text)

            if texts:
                break

        unique_texts = []
        seen = set()
        for text in texts:
            if text in seen:
                continue
            seen.add(text)
            unique_texts.append(text)

        return unique_texts[-1] if unique_texts else ""

    # Clean duplicated browser text before parsing.
    def _normalize_reply_text(self, text: str) -> str:
        normalized = (text or "").replace("\r\n", "\n").strip()
        if not normalized:
            return ""

        # Some page layouts expose the same assistant block twice in one container.
        lines = [line.strip() for line in normalized.split("\n") if line.strip()]
        if lines:
            canonical_lines = [self._canonicalize_text(line) for line in lines]
            if len(set(canonical_lines)) == 1:
                return lines[0]

            deduped_lines = []
            deduped_canonical = []
            for line, canonical in zip(lines, canonical_lines):
                if deduped_canonical and deduped_canonical[-1] == canonical:
                    continue
                deduped_lines.append(line)
                deduped_canonical.append(canonical)
            candidate = "\n".join(deduped_lines).strip()
            if candidate:
                normalized = candidate

        deduped_lines = normalized.split("\n")
        if len(deduped_lines) % 2 == 0:
            midpoint = len(deduped_lines) // 2
            first_half = [line.strip() for line in deduped_lines[:midpoint]]
            second_half = [line.strip() for line in deduped_lines[midpoint:]]
            if first_half == second_half:
                return "\n".join(deduped_lines[:midpoint]).strip()

        normalized = self._collapse_duplicate_tag_blocks(normalized, self.TOOL_CALL_BLOCK_RE)
        normalized = self._collapse_duplicate_tag_blocks(normalized, self.FINAL_BLOCK_RE)

        return normalized

    # Collapse repeated tagged blocks into one copy.
    def _collapse_duplicate_tag_blocks(self, text: str, pattern) -> str:
        blocks = [match.strip() for match in pattern.findall(text or "")]
        if len(blocks) >= 2 and len(set(blocks)) == 1:
            return blocks[0]
        return text

    # Normalize text for duplicate detection.
    def _canonicalize_text(self, text: str) -> str:
        return re.sub(r"[\s\u200b\u200c\u200d\ufeff]+", "", text or "")

    # Build a compact preview for browser logs.
    @staticmethod
    def _preview_text(value: str, limit: int = 300) -> str:
        text = "" if value is None else str(value)
        text = text.replace("\r", "\\r").replace("\n", "\\n")
        return text if len(text) <= limit else text[:limit] + "...(truncated)"
