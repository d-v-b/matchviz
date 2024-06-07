import json
import fsspec
from neuroglancer.write_annotations import AnnotationWriter
import s3fs
import os 
from typing_extensions import Self

class AnnotationWriterFSSpec(AnnotationWriter):
    def write(self: Self, path: str) -> None:
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

        if path.startswith('s3'):
            fs = fsspec.filesystem('s3')
        else:
            fs = fsspec.filesystem('local')
        
        with fs.open(os.path.join(path, "info"), mode='w') as f:
            f.write(json.dumps(metadata))
        
        with fs.open(
            os.path.join(path, "spatial0", "_".join("0" for _ in range(self.rank))),
            mode="wb",
        ) as f:
            self._serialize_annotations(f, self.annotations)

        for annotation in self.annotations:
            with fs.open(os.path.join(path, "by_id", str(annotation.id)), mode="wb") as f:
                self._serialize_annotation(f, annotation)

        for i, relationship in enumerate(self.relationships):
            rel_index = self.related_annotations[i]
            for segment_id, annotations in rel_index.items():
                with fs.open(
                    os.path.join(path, f"rel_{relationship}", str(segment_id)), mode="wb"
                ) as f:
                    self._serialize_annotations(f, annotations)