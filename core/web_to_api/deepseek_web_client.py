from __future__ import annotations

import re
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


class DeepSeekWebClient:
    TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*\{.*?\}\s*</tool_call>", re.S)
    FINAL_BLOCK_RE = re.compile(r"<final>\s*.*?\s*</final>", re.S)

    def __init__(self, config: DeepSeekWebClientConfig):
        self._config = config

    def ask(self, prompt: str) -> str:
        logger = get_logger()
        user_data_dir = self._resolve_user_data_dir()
        user_data_dir.mkdir(parents=True, exist_ok=True)

        logger.info("DeepSeek web user data dir: {}", str(user_data_dir))

        try:
            from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError(
                "Playwright is not installed. Run `pip install playwright` and `playwright install chromium`."
            ) from exc

        with sync_playwright() as playwright:
            browser_launcher = getattr(playwright, self._config.browser_type, None)
            if browser_launcher is None:
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

            context = browser_launcher.launch_persistent_context(**launch_kwargs)
            try:
                page = context.pages[0] if context.pages else context.new_page()
                page.set_default_timeout(self._config.navigation_timeout_ms)
                logger.info("Open DeepSeek chat page: {}", self._config.chat_url)
                page.goto(self._config.chat_url, wait_until="domcontentloaded")

                input_box = self._wait_for_input_box(page, PlaywrightTimeoutError)
                previous_reply = self._extract_latest_reply_text(page)
                logger.info("Previous reply preview: {}", self._preview_text(previous_reply))

                self._fill_prompt(page, input_box, prompt)
                self._send_prompt(page, input_box)
                reply = self._wait_for_reply(page, previous_reply)
                reply = self._normalize_reply_text(reply)
                logger.info("DeepSeek web reply preview: {}", self._preview_text(reply))
                return reply
            finally:
                context.close()

    def _resolve_user_data_dir(self) -> Path:
        path = Path(self._config.user_data_dir)
        if path.is_absolute():
            return path
        return Path.cwd() / path

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

    def _send_prompt(self, page, input_box):
        send_button = self._find_send_button(page, input_box)
        if send_button is not None:
            send_button.click()
            return
        input_box.press("Enter")

    def _find_send_button(self, page, input_box):
        try:
            form = input_box.locator("xpath=ancestor::form[1]")
            button = self._find_first_visible(form, self._config.send_button_selectors)
            if button is not None:
                return button
        except Exception:
            pass
        return self._find_first_visible(page, self._config.send_button_selectors)

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

    def _has_loading_indicator(self, page) -> bool:
        indicator = self._find_first_visible(page, self._config.loading_selectors)
        return indicator is not None

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

    def _collapse_duplicate_tag_blocks(self, text: str, pattern) -> str:
        blocks = [match.strip() for match in pattern.findall(text or "")]
        if len(blocks) >= 2 and len(set(blocks)) == 1:
            return blocks[0]
        return text

    def _canonicalize_text(self, text: str) -> str:
        return re.sub(r"[\s\u200b\u200c\u200d\ufeff]+", "", text or "")

    @staticmethod
    def _preview_text(value: str, limit: int = 300) -> str:
        text = "" if value is None else str(value)
        text = text.replace("\r", "\\r").replace("\n", "\\n")
        return text if len(text) <= limit else text[:limit] + "...(truncated)"
