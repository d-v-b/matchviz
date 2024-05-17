# SPDX-FileCopyrightText: 2024-present Davis Vann Bennett <davis.v.bennett@gmail.com>
#
# SPDX-License-Identifier: MIT
from __future__ import annotations
from typing import Annotated, Any
import numpy as np
import zarr
import neuroglancer
import os 
from pydantic import BaseModel, BeforeValidator, Field

from pydantic_zarr.v2 import GroupSpec, ArraySpec

from typing_extensions import TypedDict


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

class PointsGroup(GroupSpec[InterestPointsGroupMeta, InterestPointsMembers]):
    members: InterestPointsMembers
    ...

dataset = 'exaSPIM_3163606_2023-11-17_12-54-51'
image_path = f's3://aind-open-data/{dataset}'
alignment_id = 'alignment_2024-01-09_05-00-44'
alignment_path = f's3://aind-open-data/{dataset}/_{alignment_id}/interestpoints.n5/'
alignment_store = zarr.N5FSStore(alignment_path, anon = True)
corresp = zarr.open_group(alignment_store, path='tpId_0_viewSetupId_0/beads/correspondences')
ips = zarr.open_group(alignment_store, path='tpId_0_viewSetupId_0/beads/interestpoints')


def create_annotation_layer_from_points(url: str):
    store = zarr.N5FSStore(os.path.join(url, 'interestpoints.n5/'))
    setup_id = 0
    tp_id = 0
    prefix = f'tpId_{tp_id}_viewSetupId_{setup_id}'
    matches = zarr.open_group(store, path=os.path.join(prefix, '/beads/correspondences'))
    points = zarr.open_group(store, path=os.path.join(prefix, '/beads/interestpoints'))
    point_ids, point_locs = np.array(points['id']), np.array(points['loc'])
    img_layer = neuroglancer.ImageLayer(source=f'n5://{}')
    ann_layer = neuroglancer.AnnotationLayer(points=point_locs)