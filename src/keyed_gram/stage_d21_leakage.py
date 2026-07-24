"""D2.1 高熵 canary 生成与应用层持久化扫描。"""

from __future__ import annotations

import base64
import secrets
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


CANARY_MARKER = "SYN" + "-D21-"


@dataclass(slots=True)
class CanaryFactory:
    _issued: set[str] = field(default_factory=set, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def issue(self) -> str:
        with self._lock:
            for _ in range(32):
                value = CANARY_MARKER + secrets.token_hex(16).upper()
                if value not in self._issued:
                    self._issued.add(value)
                    return value
        raise RuntimeError("无法生成唯一 D2.1 canary")


def canary_variants(canary: str) -> frozenset[bytes]:
    raw = canary.encode("ascii")
    values = {
        raw,
        raw.lower(),
        raw.upper(),
        raw.hex().encode("ascii"),
        base64.b64encode(raw),
        base64.urlsafe_b64encode(raw).rstrip(b"="),
        b" ".join(bytes((item,)) for item in raw),
        raw.replace(b"-", b""),
    }
    return frozenset(value for value in values if value)


def count_canary_occurrences(
    payloads: Iterable[bytes | str],
    canaries: Iterable[str],
) -> int:
    encoded = [
        value.encode("utf-8") if isinstance(value, str) else bytes(value)
        for value in payloads
    ]
    variants = {
        variant
        for canary in canaries
        for variant in canary_variants(canary)
    }
    return sum(payload.count(variant) for payload in encoded for variant in variants)


def scan_paths(paths: Iterable[Path], canaries: Iterable[str]) -> int:
    payloads: list[bytes] = []
    for root in paths:
        if not root.exists():
            continue
        if root.is_file():
            payloads.append(root.read_bytes())
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file():
                payloads.append(path.read_bytes())
    return count_canary_occurrences(payloads, canaries)
