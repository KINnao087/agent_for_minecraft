import base64

def decode_base64_utf8(encoded: str) -> str:
    decoded = base64.b64decode(encoded)
    return decoded.decode("utf-8")