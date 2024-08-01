from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

NeuroglancerViewerStyle = Literal["images_combined", "images_split"]

neuroglancer_view_styles: tuple[NeuroglancerViewerStyle, ...] = (
    "images_combined",
    "images_split",
)


@dataclass
class Described:
    description: str


@dataclass
class FileName(Described):
    name: str


fnames: dict[NeuroglancerViewerStyle, FileName] = {
    "images_combined": FileName(
        description="All tiles combined", name="images_combined.json"
    ),
    "images_split": FileName(
        description="One layer per tile", name="images_split.json"
    ),
}
