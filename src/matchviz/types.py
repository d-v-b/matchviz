from typing import Literal
from typing_extensions import TypedDict


class Tx(TypedDict):
    scale: float
    trans: float


class Coords(TypedDict):
    x: Tx
    y: Tx
    z: Tx


class TileCoordinate(TypedDict):
    x: int
    y: int
    z: int
    ch: Literal["488", "561"]
