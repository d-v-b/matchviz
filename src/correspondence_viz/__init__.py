# SPDX-FileCopyrightText: 2024-present Davis Vann Bennett <davis.v.bennett@gmail.com>
#
# SPDX-License-Identifier: MIT
from __future__ import annotations
from typing import Annotated, Any, Literal
import neuroglancer.coordinate_space
import numpy as np
import zarr
import neuroglancer
import os 
from pydantic import BaseModel, BeforeValidator, Field
import tempfile
import atexit
import shutil
from pydantic_zarr.v2 import GroupSpec, ArraySpec
from pydantic_bigstitcher import SpimData2
from pydantic_ome_ngff.v04.multiscale import MultiscaleMetadata
from s3fs import S3FileSystem

from typing_extensions import TypedDict
from neuroglancer.write_annotations import AnnotationWriter

def write_point_annotations(
    path: str,
    points: np.ndarray,
    coordinate_space: neuroglancer.CoordinateSpace,
    rgba: tuple[int, int, int, int] = (255, 255, 255, 255),
):
    writer = AnnotationWriter(
        coordinate_space=coordinate_space,
        annotation_type="point",
        properties=[
            neuroglancer.AnnotationPropertySpec(id="size", type="float32"),
            neuroglancer.AnnotationPropertySpec(id="cell_type", type="uint16"),
            neuroglancer.AnnotationPropertySpec(id="point_color", type="rgba"),
        ],
    )

    [writer.add_point(c, size=10, cell_type=16, point_color=rgba) for c in points]
    writer.write(path)

def parse_idmap(data: dict[str, int]) -> dict[tuple[int, int, str], int]:
    """
    convert {'0,1,beads': 0} to {(0, 1, "beads"): 0}
    """
    parts = map(lambda k: k.split(','), data.keys())
    # convert first two elements to int, leave the last as str
    parts_normalized = map(lambda v: (int(v[0], int[v[1], v[2]])), parts)
    return dict(zip(parts_normalized, data.values()))

class InterestPointsGroupMeta(BaseModel):
    list_version: str = Field(alias='list version')
    pointcloud: str
    type: str

class CorrespondencesGroupMeta(BaseModel):
    correspondences: str
    idmap: Annotated[dict[tuple[int, int, str], int], BeforeValidator(parse_idmap)]

class InterestPointsMembers(TypedDict):
    """
    id is a num_points X 1 array of integer IDs
    loc is a num_points X ndim array of locations in work coordinates
    """
    id: ArraySpec
    loc: ArraySpec

class TileCoordinate(TypedDict):
    x: int
    y: int
    z: int
    ch: Literal['488', '561']

def resolve_image_path(coord: TileCoordinate) -> str:
    """
    {'x': 0, 'y': 0, 'ch': 561'} -> "tile_x_0000_y_0002_z_0000_ch_488.zarr/"
    """
    cx = coord["x"]
    cy = coord["y"]
    cz = coord["z"]
    cch = coord['ch']
    return f'tile_x_{cx:04}_y_{cy:04}_z_{cz:04}_ch_{cch}'

class PointsGroup(GroupSpec[InterestPointsGroupMeta, InterestPointsMembers]):
    members: InterestPointsMembers
    ...

def visualize_points(bdv_url: str):
    bdv_xml = S3FileSystem(anon=True).cat_file(bdv_url)
    bdv_model = SpimData2.from_xml(bdv_xml)
    alignment_url = bdv_model.sequence_description.image_loader.zarr
    assert False

def create_annotation_layer_from_points(
        dataset: str, 
        alignment_id: str):
    """
    e.g. dataset = 'exaSPIM_3163606_2023-11-17_12-54-51'
        alignment_id = 'alignment_2024-01-09_05-00-44'
    """
    coord: TileCoordinate = {'x': 0, 'y': 0, 'z': 0, 'ch': '488'}
    alignment_url = f's3://aind-open-data/{dataset}_{alignment_id}/interestpoints.n5/'
    tile_name = resolve_image_path(coord)
    image_url = f's3://aind-open-data/{dataset}/SPIM.ome.zarr/{tile_name}.zarr'
    multi_meta = MultiscaleMetadata(**zarr.open(image_url).attrs.asdict()['multiscales'][0])
    scale = multi_meta.datasets[0].coordinateTransformations[0].scale
    trans = multi_meta.datasets[0].coordinateTransformations[1].translation
    base_coords = {axis.name: {'scale': s, 'translation': t} for axis, s, t in zip(multi_meta.axes, scale, trans)}
    alignment_store = zarr.N5FSStore(alignment_url)
    timepoint_id = 0
    setup_id = 0
    prefix = f'tpId_{timepoint_id}_viewSetupId_{setup_id}'
    matches = zarr.open_group(
        store=alignment_store, 
        path=os.path.join(prefix, 'beads', 'correspondences'), 
        mode='r')
    points = zarr.open_group(
        store=alignment_store, 
        path=os.path.join(prefix, 'beads', 'interestpoints'), 
        mode='r')
    # point_ids = np.array(points['id'])
    point_loc_stored = points['loc']
    # points are saved as x, y, z triplets
    point_loc = np.zeros((point_loc_stored.shape[0], point_loc_stored.shape[1] + 1), dtype=point_loc_stored.dtype)
    point_loc[:,:-1] = point_loc_stored[:] 
    # apply base scaling and translation
    for idx, dim in enumerate(('x','y','z')):
        local_scale = base_coords[dim]['scale']
        local_trans = base_coords[dim]['translation']
        point_loc[:, idx] += local_trans / local_scale
        #point_loc[:, idx] += local_trans
    
    annotation_scales = [base_coords[dim]['scale'] for dim in ('x','y','z','t')]
    annotation_units = ['um', 'um', 'um', 's']
    output_scales = [100, 100, 100, 1]
    output_units = ['nm', 'nm', 'nm', 'ms']
    annotation_space = neuroglancer.CoordinateSpace(
        names=['x','y','z','t'],
        scales=annotation_scales,
        units=annotation_units
        )

    image_coordinate_space = neuroglancer.CoordinateSpace(
        names=['z','y','x','t'], 
        scales=output_scales,
        units=output_units)

    write_point_annotations(
        f'{tile_name}.precomputed', 
        points = point_loc,
        coordinate_space=annotation_space, 
        rgba=(0, 255, 255, 255))

    img_layer = neuroglancer.ImageLayer(
        source=f'zarr://{image_url}',
        layer_dimensions=image_coordinate_space)
    ann_layer = neuroglancer.AnnotationLayer(
        source=f'precomputed://http://localhost:3000/{tile_name}.precomputed',
        )

    viewer = neuroglancer.Viewer()
    with viewer.txn() as s:
        s.dimensions=image_coordinate_space
        s.layers['points'] = ann_layer
        s.layers["image"] = img_layer
    breakpoint()
    return viewer