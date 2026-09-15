"""Drives the Swift IOBluetooth helper over its stdio line protocol."""

import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import IO

BIN = Path(__file__).resolve().parent.parent / "bin" / "divoom-bridge"


class BridgeError(RuntimeError):
    pass


def _run(args: list[str], timeout: float = 30) -> list[str]:
    if not BIN.exists():
        raise BridgeError(f"{BIN} is missing — run ./build.sh")
    done = subprocess.run([str(BIN), *args], capture_output=True, text=True, timeout=timeout)
    if done.returncode != 0:
        raise BridgeError(done.stderr.strip() or f"exit {done.returncode}")
    return [line for line in done.stdout.splitlines() if line.strip()]


def paired() -> list[tuple[str, str, str]]:
    rows = []
    for line in _run(["list"]):
        address, state, name = line.split("\t", 2)
        rows.append((address, state, name))
    return rows


def services(address: str) -> list[str]:
    return _run(["services", address], timeout=40)


class Link:
    """An open RFCOMM channel. Writes are hex lines, replies come back async."""

    def __init__(self, address: str, channel: int = 1, keep_audio: bool = False, verbose: bool = False):
        if not BIN.exists():
            raise BridgeError(f"{BIN} is missing — run ./build.sh")

        args = [str(BIN), "connect", address, str(channel)]
        if keep_audio:
            args.append("--keep-audio")

        self.verbose = verbose
        self.events: queue.Queue[str] = queue.Queue()
        self.proc = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            text=True, bufsize=1,
        )
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None
        self.stdin: IO[str] = self.proc.stdin
        self.stdout: IO[str] = self.proc.stdout
        threading.Thread(target=self._pump, daemon=True).start()

        first = self._await(("READY", "ERR"), timeout=25)
        if first.startswith("ERR"):
            raise BridgeError(first[4:])
        self.mtu = int(first.split()[1])

    def _pump(self) -> None:
        for line in self.stdout:
            self.events.put(line.strip())
        self.events.put("CLOSED")

    def _await(self, prefixes: tuple[str, ...], timeout: float) -> str:
        deadline = timeout
        while True:
            try:
                line = self.events.get(timeout=deadline)
            except queue.Empty:
                raise BridgeError(f"timed out waiting for {'/'.join(prefixes)}")
            if self.verbose and not line.startswith(prefixes):
                print(f"  · {line}")
            if line.startswith(prefixes):
                return line
            if line == "CLOSED":
                raise BridgeError("channel closed by the device")

    def send(self, *messages: bytes, timeout: float = 10, gap: float = 0.0) -> None:
        for index, msg in enumerate(messages):
            if gap and index:
                time.sleep(gap)
            if self.verbose:
                print(f"  → {msg.hex()}")
            self.stdin.write(msg.hex() + "\n")
            self.stdin.flush()
            reply = self._await(("OK", "ERR"), timeout)
            if reply.startswith("ERR"):
                raise BridgeError(reply[4:])

    def drain(self) -> list[str]:
        """Whatever the device volunteered since the last call."""
        out = []
        while True:
            try:
                out.append(self.events.get_nowait())
            except queue.Empty:
                return out

    def close(self) -> None:
        try:
            self.stdin.close()
            self.proc.wait(timeout=3)
        except Exception:
            self.proc.kill()
