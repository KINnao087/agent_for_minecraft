from __future__ import annotations

import os

from core.web_to_api.deepseek_web_client import DeepSeekWebClient, DeepSeekWebClientConfig


# Normalize provider aliases to api or web.
def normalize_provider_name(provider_name: str | None) -> str:
    value = str(provider_name or "api").strip().lower()
    aliases = {
        "api": "api",
        "deepseek_api": "api",
        "web": "web",
        "deepseek_web": "web",
    }
    try:
        return aliases[value]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported provider: {provider_name}. Supported values: api, web."
        ) from exc


# Build the configured chat provider instance.
def build_provider(config: dict):
    provider_name = normalize_provider_name(config.get("provider"))

    if provider_name == "api":
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

    if provider_name == "web":
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
            keep_alive_url=web_config.get("keep_alive_url") or defaults.keep_alive_url,
            new_chat_url=web_config.get("new_chat_url"),
            new_chat_timeout_ms=int(web_config.get("new_chat_timeout_ms", defaults.new_chat_timeout_ms)),
            new_chat_selectors=tuple(web_config.get("new_chat_selectors") or defaults.new_chat_selectors),
        )

        return DeepSeekWebProvider(DeepSeekWebClient(client_config))

    raise ValueError(f"Unsupported provider: {provider_name}. Supported values: api, web.")
