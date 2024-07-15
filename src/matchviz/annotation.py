from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Sequence
import neuroglancer
import numpy as np
from neuroglancer.write_annotations import AnnotationWriter
import fsspec
import os
from typing_extensions import Self
import json
import time
import logging
import io

pool = ThreadPoolExecutor(max_workers=36)


def cp(buf_in, fs_out, fname):
    with fs_out.open(fname, mode="wb") as fh:
        fh.write(buf_in.read())
    return "ok"


def write_line_annotations(
    path: str,
    lines: Iterable[tuple[tuple[float, ...], ...]],
    coordinate_space: neuroglancer.CoordinateSpace,
    *,
    point_color: tuple[int, int, int, int] = (255, 255, 255, 255),
    line_color: tuple[int, int, int, int] = (255, 255, 255, 255),
) -> None:
    """
    Save a collection of paired points (i.e., lines) as neuroglancer precomputed annotations.

    Lines will be stored with a `point_color` attribute, which defaults to (255, 255, 255, 255),
    and a `line_color` attribute, which defaults to (255, 255, 255, 255)
    """
    writer = AnnotationWriterFSSpec(
        coordinate_space=coordinate_space,
        annotation_type="line",
        properties=[
            neuroglancer.AnnotationPropertySpec(id="line_color", type="rgba"),
            neuroglancer.AnnotationPropertySpec(id="point_color", type="rgba"),
        ],
    )

    [
        writer.add_line(*points, point_color=point_color, line_color=line_color)
        for points in lines
    ]
    writer.write(path)


def write_point_annotations(
    path: str,
    *,
    points: np.ndarray | list[Sequence[float]],
    ids: np.ndarray | list[Sequence[float]],
    coordinate_space: neuroglancer.CoordinateSpace,
    point_size=10,
    point_color: tuple[int, int, int, int] = (127, 255, 127, 255),
) -> None:
    """
    Save a collection of points as neuroglancer precomputed annotations.

    Points will be stored with the following attributes:
    - `id`, which is the id of each point,
    - `size` attribute, which defaults to 10,
    - `point_color`, which defaults to (127, 255, 127, 255)
    """
    writer = AnnotationWriterFSSpec(
        coordinate_space=coordinate_space,
        annotation_type="point",
        properties=[
            neuroglancer.AnnotationPropertySpec(id="id", type="int32"),
            neuroglancer.AnnotationPropertySpec(id="size", type="float32"),
            neuroglancer.AnnotationPropertySpec(id="point_color", type="rgba"),
        ],
    )

    for _id, point in zip(ids, points):
        writer.add_point(point, id=_id, size=point_size, point_color=point_color)
    writer.write(path)


class AnnotationWriterFSSpec(AnnotationWriter):
    """
    An AnnotationWriter that uses FSSpec for writing, enabling cloud storage.
    """

    def write(self: Self, path: str) -> None:
        logger = logging.getLogger(__name__)

        fut_map = {}
        metadata = {
            "@type": "neuroglancer_annotations_v1",
            "dimensions": self.coordinate_space.to_json(),
            "lower_bound": [float(x) for x in self.lower_bound],
            "upper_bound": [float(x) for x in self.upper_bound],
            "annotation_type": self.annotation_type,
            "properties": [p.to_json() for p in self.properties],
            "relationships": [
                {"id": relationship, "key": f"rel_{relationship}"}
                for relationship in self.relationships
            ],
            "by_id": {
                "key": "by_id",
            },
            "spatial": [
                {
                    "key": "spatial0",
                    "grid_shape": [1] * self.rank,
                    "chunk_size": [
                        max(1, float(x)) for x in self.upper_bound - self.lower_bound
                    ],
                    "limit": len(self.annotations),
                },
            ],
        }

        if path.startswith("s3://"):
            fs, _ = fsspec.url_to_fs(path)
        else:
            fs, _ = fsspec.url_to_fs(path, auto_mkdir=True)

        spatial_path = os.path.join(
            path, "spatial0", "_".join("0" for _ in range(self.rank))
        )

        start = time.time()
        logger.info("Writing info...")

        with fs.open(os.path.join(path, "info"), mode="w") as f:
            f.write(json.dumps(metadata))

        logger.info(f"Preparing to write annotations to {spatial_path}...")
        spatial_buf = io.BytesIO()
        self._serialize_annotations(spatial_buf, self.annotations)
        spatial_buf.seek(0)

        with fs.open(spatial_path, mode="wb") as f_rem:
            f_rem.write(spatial_buf.read())

        logger.info(f'Preparing to write annotations to {os.path.join(path, "by_id")}')

        for annotation in self.annotations:
            by_id_path = os.path.join(path, "by_id", str(annotation.id))

            id_buf = io.BytesIO()
            self._serialize_annotation(id_buf, annotation)
            id_buf.seek(0)
            fut = pool.submit(cp, id_buf, fs, by_id_path)
            fut_map[fut] = by_id_path

        logger.info(
            f'Preparing to write relationships to {os.path.join(path, "rel_*")}'
        )
        for i, relationship in enumerate(self.relationships):
            rel_index = self.related_annotations[i]

            for segment_id, anns in rel_index.items():
                rel_path = os.path.join(path, f"rel_{relationship}", str(segment_id))
                rel_buf = io.BytesIO()
                self._serialize_annotations(rel_buf, anns)
                rel_buf.seek(0)
                fut = pool.submit(cp, rel_buf, fs, rel_path)
                fut_map[fut] = rel_path

        for result in as_completed(fut_map.keys()):
            _ = result.result()

        elapsed = time.time() - start
        logger.info(f"Completed saving annotation in {elapsed:.4f} s")
