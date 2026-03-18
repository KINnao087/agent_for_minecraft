from __future__ import annotations

import os

from core.web_to_api.deepseek_web_client import DeepSeekWebClient, DeepSeekWebClientConfig


def build_provider(config: dict):
    provider_name = str(config.get("provider") or "deepseek_api").strip().lower()

    if provider_name == "deepseek_api":
        from core.web_to_api.deepseek_api_provider import DeepSeekApiProvider

        api_config = config.get("deepseek_api") or {}
        api_key = (
            os.environ.get("DEEPSEEK_API_KEY")
            or api_config.get("api_key")
            or config.get("api_key")
        )
        if not api_key:
            raise RuntimeError("Missing DeepSeek API key. Set DEEPSEEK_API_KEY or config.api_key.")
        base_url = api_config.get("base_url") or "https://api.deepseek.com"
        temperature = float(api_config.get("temperature", 0.2))
        return DeepSeekApiProvider(api_key=api_key, base_url=base_url, temperature=temperature)

    if provider_name == "deepseek_web":
        from core.web_to_api.deepseek_web_provider import DeepSeekWebProvider

        web_config = config.get("deepseek_web") or {}
        defaults = DeepSeekWebClientConfig()
        client_config = DeepSeekWebClientConfig(
            chat_url=os.environ.get("DEEPSEEK_WEB_CHAT_URL") or web_config.get("chat_url") or defaults.chat_url,
            user_data_dir=(
                os.environ.get("DEEPSEEK_WEB_USER_DATA_DIR")
                or web_config.get("user_data_dir")
                or defaults.user_data_dir
            ),
            browser_type=web_config.get("browser_type") or defaults.browser_type,
            browser_channel=web_config.get("browser_channel"),
            headless=bool(web_config.get("headless", defaults.headless)),
            viewport_width=int(web_config.get("viewport_width", defaults.viewport_width)),
            viewport_height=int(web_config.get("viewport_height", defaults.viewport_height)),
            input_ready_timeout_ms=int(web_config.get("input_ready_timeout_ms", defaults.input_ready_timeout_ms)),
            navigation_timeout_ms=int(web_config.get("navigation_timeout_ms", defaults.navigation_timeout_ms)),
            reply_timeout_ms=int(web_config.get("reply_timeout_ms", defaults.reply_timeout_ms)),
            poll_interval_ms=int(web_config.get("poll_interval_ms", defaults.poll_interval_ms)),
            stable_rounds=int(web_config.get("stable_rounds", defaults.stable_rounds)),
            input_selectors=tuple(web_config.get("input_selectors") or defaults.input_selectors),
            send_button_selectors=tuple(web_config.get("send_button_selectors") or defaults.send_button_selectors),
            assistant_message_selectors=tuple(
                web_config.get("assistant_message_selectors") or defaults.assistant_message_selectors
            ),
            loading_selectors=tuple(web_config.get("loading_selectors") or defaults.loading_selectors),
        )

        return DeepSeekWebProvider(DeepSeekWebClient(client_config))

    raise ValueError(f"Unsupported provider: {provider_name}")
