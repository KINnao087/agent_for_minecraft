def deepseek_chat(client, model, tool_defs, messages, stream_thinking_callback=None):
    return client.chat(
        model=model,
        tool_defs=tool_defs,
        messages=messages,
        stream_thinking_callback=stream_thinking_callback,
    )
