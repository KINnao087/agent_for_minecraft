import argparse
import json
import socketserver
import threading
from typing import Dict, List

from agent import build_initial_session, run_agent_and_get_reply
from log import get_logger


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

        while True:
            raw = self.rfile.readline()
            if not raw:
                break

            try:
                request = json.loads(raw.decode("utf-8").strip())
                response = self.process_request(request)
            except json.JSONDecodeError as exc:
                response = {"ok": False, "error": f"invalid json: {exc}"}
            except Exception as exc:
                logger.error("socket request failed: {}", str(exc))
                response = {"ok": False, "error": str(exc)}

            self.wfile.write((json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8"))
            self.wfile.flush()

        logger.info("socket client disconnected: {}", client_addr)

    def process_request(self, request: dict):
        action = request.get("action", "chat")
        session_id = str(request.get("session_id") or "default")

        if action == "reset":
            SESSION_MANAGER.reset(session_id)
            return {"ok": True, "action": "reset", "session_id": session_id}

        if action != "chat":
            return {"ok": False, "error": f"unsupported action: {action}"}

        task = request.get("task")
        if not isinstance(task, str) or not task.strip():
            return {"ok": False, "error": "task must be a non-empty string"}

        session = SESSION_MANAGER.get(session_id)
        reply, updated_session = run_agent_and_get_reply(
            task=task,
            max_steps=int(request.get("max_steps", 18)),
            enable_thinking_stream=bool(request.get("stream", False)),
            session=session,
        )
        SESSION_MANAGER.set(session_id, updated_session)

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
