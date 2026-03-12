import argparse
import json
import socketserver
import threading
from typing import Dict, List

from agent import build_initial_session, run_agent_and_get_reply
from log import get_logger


def _preview_text(value, limit=500):
    text = "" if value is None else str(value)
    text = text.replace("\r", "\\r").replace("\n", "\\n")
    return text if len(text) <= limit else text[:limit] + "...(truncated)"


def _preview_json(value, limit=500):
    try:
        text = json.dumps(value, ensure_ascii=False)
    except Exception:
        text = str(value)
    return _preview_text(text, limit=limit)


class SessionManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._sessions: Dict[str, List[dict]] = {}

    def get(self, session_id: str):
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = build_initial_session()
                self._sessions[session_id] = session
            return session.copy()

    def set(self, session_id: str, session):
        with self._lock:
            self._sessions[session_id] = session.copy()

    def reset(self, session_id: str):
        with self._lock:
            self._sessions[session_id] = build_initial_session()


SESSION_MANAGER = SessionManager()


class JsonLineTCPHandler(socketserver.StreamRequestHandler):
    def handle(self):
        logger = get_logger()
        client_addr = f"{self.client_address[0]}:{self.client_address[1]}"
        logger.info("socket client connected: {}", client_addr)
        self.connection.settimeout(1.0)
        buffer = b""

        while True:
            try:
                chunk = self.connection.recv(4096)
            except TimeoutError:
                if buffer:
                    logger.info(
                        "socket waiting for newline from {}: buffered {} byte(s), preview={}",
                        client_addr,
                        len(buffer),
                        _preview_text(buffer.decode("utf-8", errors="replace")),
                    )
                continue

            if not chunk:
                if buffer.strip():
                    logger.warning(
                        "socket client {} disconnected with unterminated payload, treating buffer as one request",
                        client_addr,
                    )
                    self._process_and_respond(client_addr, buffer)
                break

            logger.info(
                "socket chunk from {}: {} byte(s), preview={}",
                client_addr,
                len(chunk),
                _preview_text(chunk.decode("utf-8", errors="replace")),
            )
            buffer += chunk

            while b"\n" in buffer:
                raw, buffer = buffer.split(b"\n", 1)
                if not raw.strip():
                    logger.info("socket blank line ignored from {}", client_addr)
                    continue
                self._process_and_respond(client_addr, raw)

        logger.info("socket client disconnected: {}", client_addr)

    def _process_and_respond(self, client_addr: str, raw: bytes):
        logger = get_logger()
        logger.info("socket raw request from {}: {}", client_addr, _preview_text(raw.decode("utf-8", errors="replace")))

        try:
            request = json.loads(raw.decode("utf-8").strip())
            logger.info("socket parsed request from {}: {}", client_addr, _preview_json(request))
            response = self.process_request(request)
        except json.JSONDecodeError as exc:
            logger.error("invalid json from {}: {}", client_addr, str(exc))
            response = {"ok": False, "error": f"invalid json: {exc}"}
        except Exception as exc:
            logger.error("socket request failed: {}", str(exc))
            response = {"ok": False, "error": str(exc)}

        logger.info("socket response to {}: {}", client_addr, _preview_json(response))
        self.wfile.write((json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8"))
        self.wfile.flush()

    def process_request(self, request: dict):
        logger = get_logger()
        action = request.get("action", "chat")
        session_id = str(request.get("session_id") or "default")
        logger.info("process request action={}, session_id={}", action, session_id)

        if action == "ping":
            logger.info("heartbeat received for session {}", session_id)
            return {"ok": True, "action": "pong", "session_id": session_id}

        if action == "reset":
            SESSION_MANAGER.reset(session_id)
            logger.info("session reset completed: {}", session_id)
            return {"ok": True, "action": "reset", "session_id": session_id}

        if action != "chat":
            logger.warning("unsupported action received: {}", action)
            return {"ok": False, "error": f"unsupported action: {action}"}

        task = request.get("task")
        if not isinstance(task, str) or not task.strip():
            logger.warning("invalid task for session {}: {}", session_id, _preview_text(task))
            return {"ok": False, "error": "task must be a non-empty string"}

        logger.info("chat task for session {}: {}", session_id, _preview_text(task))
        session = SESSION_MANAGER.get(session_id)
        logger.info("session {} loaded with {} message(s)", session_id, len(session))
        reply, updated_session = run_agent_and_get_reply(
            task=task,
            max_steps=int(request.get("max_steps", 18)),
            enable_thinking_stream=bool(request.get("stream", False)),
            session=session,
        )
        SESSION_MANAGER.set(session_id, updated_session)
        logger.info("session {} updated to {} message(s)", session_id, len(updated_session))
        logger.info("final reply for session {}: {}", session_id, _preview_text(reply))

        return {
            "ok": True,
            "action": "chat",
            "session_id": session_id,
            "reply": reply,
        }


def main():
    parser = argparse.ArgumentParser(description="JSON socket server for agent_for_minecraft")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5050)
    args = parser.parse_args()

    logger = get_logger()
    with socketserver.ThreadingTCPServer((args.host, args.port), JsonLineTCPHandler) as server:
        logger.info("socket server listening on {}:{}", args.host, args.port)
        server.serve_forever()


if __name__ == "__main__":
    main()
