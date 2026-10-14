from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class Treatment:
    codec: str
    setting_id: str
    settings: dict[str, object]
    lossless: bool


class Codec(Protocol):
    name: str

    def treatments(
        self, values: list[int] | None = None, options: Mapping[str, object] | None = None
    ) -> list[Treatment]: ...

    def encode(self, image: np.ndarray, settings: dict[str, object]) -> bytes: ...

    def decode(self, data: bytes) -> np.ndarray: ...

    def implementation_metadata(self) -> dict[str, str]: ...
