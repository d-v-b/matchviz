from concurrent.futures import ThreadPoolExecutor, as_completed
import neuroglancer
import numpy as np
from neuroglancer.write_annotations import AnnotationWriter
import fsspec
import os
from typing_extensions import Self
import json
from tqdm import tqdm
import logging
import io


def write_line_annotations(
    path: str,
    lines: np.ndarray,
    coordinate_space: neuroglancer.CoordinateSpace,
    rgba: tuple[int, int, int, int] = (255, 255, 255, 255),
) -> None:
    writer = AnnotationWriterFSSpec(
        coordinate_space=coordinate_space,
        annotation_type="line",
        properties=[
            neuroglancer.AnnotationPropertySpec(id="line_color", type="rgba"),
            neuroglancer.AnnotationPropertySpec(id="point_color", type="rgba"),
        ],
    )

    [writer.add_line(*points, point_color=rgba, line_color=rgba) for points in lines]
    writer.write(path)


pool = ThreadPoolExecutor(max_workers=36)


def cp(buf_in, fs_out, fname):
    with fs_out.open(fname, mode="wb") as fh:
        fh.write(buf_in.read())
    return "ok"


def write_point_annotations(
    path: str,
    points: np.ndarray,
    coordinate_space: neuroglancer.CoordinateSpace,
    rgba: tuple[int, int, int, int] = (127, 255, 127, 255),
) -> None:
    writer = AnnotationWriterFSSpec(
        coordinate_space=coordinate_space,
        annotation_type="point",
        properties=[
            neuroglancer.AnnotationPropertySpec(id="size", type="float32"),
            neuroglancer.AnnotationPropertySpec(id="point_color", type="rgba"),
        ],
    )

    [writer.add_point(c, size=10, point_color=rgba) for c in points]
    writer.write(path)


class AnnotationWriterFSSpec(AnnotationWriter):
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
            fs = fsspec.filesystem("s3")
        else:
            fs = fsspec.filesystem("local", auto_mkdir=True)

        spatial_path = os.path.join(
            path, "spatial0", "_".join("0" for _ in range(self.rank))
        )

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

            for segment_id, annotations in rel_index.items():
                rel_path = os.path.join(path, f"rel_{relationship}", str(segment_id))
                rel_buf = io.BytesIO()
                self._serialize_annotations(rel_buf, annotations)
                rel_buf.seek(0)
                fut = pool.submit(cp, rel_buf, fs, rel_path)
                fut_map[fut] = rel_path

        for result in tqdm(as_completed(fut_map.keys())):
            _ = result.result()
