"""A line-oriented socket so anything can drive the face.

TCP rather than a callback API: the events come from outside this process —
a CI hook, a shell script, a game — and `nc` or one line of curl-equivalent is
a lower bar than importing a Python module.

    printf 'hurt 25\\n' | nc 127.0.0.1 8777
"""

import queue
import socket
import threading

PORT = 8777


class EventFeed:
    """Accepts `<name> [amount]` lines and hands them to whoever asks."""

    def __init__(self, port: int = PORT, host: str = "127.0.0.1"):
        self.events: queue.Queue[tuple[str, int]] = queue.Queue()
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, port))
        self.server.listen(4)
        self.port = port
        self.closed = False
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self) -> None:
        while not self.closed:
            try:
                conn, _ = self.server.accept()
            except OSError:
                return
            threading.Thread(target=self._read, args=(conn,), daemon=True).start()

    def _read(self, conn: socket.socket) -> None:
        with conn, conn.makefile("r") as stream:
            for line in stream:
                parts = line.split()
                if not parts:
                    continue
                amount = int(parts[1]) if len(parts) > 1 and parts[1].lstrip("-").isdigit() else 20
                self.events.put((parts[0].lower(), amount))

    def drain(self) -> list[tuple[str, int]]:
        out = []
        while True:
            try:
                out.append(self.events.get_nowait())
            except queue.Empty:
                return out

    def close(self) -> None:
        self.closed = True
        self.server.close()
