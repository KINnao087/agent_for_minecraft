from core.token_counter import count_message_tokens


# Sum non-system message tokens for the current history.
def calc_total_tokens(messages, model: str):
    return sum(count_message_tokens(m, model) for m in messages if m.get("role") != "system")


# Keep the most recent messages within the token budget.
def trim_messages(messages, keep_last, model: str, cached_total_tokens=None, return_total=False):
    if cached_total_tokens is not None and cached_total_tokens <= keep_last:
        return (messages, cached_total_tokens) if return_total else messages

    if not messages:
        return ([], 0) if return_total else []

    system_messages = [m for m in messages if m.get("role") == "system"]
    other = [m for m in messages if m.get("role") != "system"]
    if not other:
        return (system_messages, 0) if return_total else system_messages

    window = []
    total = 0
    i = len(other) - 1

    while i >= 0:
        m = other[i]
        role = m.get("role")

        if role == "tool" and i - 1 >= 0 and other[i - 1].get("role") == "assistant":
            pair = [other[i - 1], other[i]]
            pair_tokens = count_message_tokens(pair[0], model) + count_message_tokens(pair[1], model)
            if total + pair_tokens <= keep_last:
                window[0:0] = pair
                total += pair_tokens
                i -= 2
                continue
            break

        mt = count_message_tokens(m, model)
        if total + mt <= keep_last:
            window.insert(0, m)
            total += mt
            i -= 1
            continue

        i -= 1
        continue

    if not window:
        last = other[-1]
        if last.get("role") == "tool" and len(other) >= 2 and other[-2].get("role") == "assistant":
            window = [other[-2], other[-1]]
        else:
            window = [last]
        total = sum(count_message_tokens(m, model) for m in window if m.get("role") != "system")

    trimmed = system_messages + window
    return (trimmed, total) if return_total else trimmed


# Append one message and optionally trim the history.
def append_message(messages, message, total_tokens, keep_last, model: str, auto_trim=False):
    messages.append(message)
    if message.get("role") != "system":
        total_tokens += count_message_tokens(message, model)

    if auto_trim:
        messages, total_tokens = trim_messages(
            messages,
            keep_last=keep_last,
            model=model,
            cached_total_tokens=total_tokens,
            return_total=True,
        )
    return messages, total_tokens

