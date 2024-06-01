# SPDX-FileCopyrightText: 2024-present Davis Vann Bennett <davis.v.bennett@gmail.com>
#
# SPDX-License-Identifier: MIT
from __future__ import annotations
from typing import Annotated, Any, Literal, Sequence
import neuroglancer.coordinate_space
import numpy as np
import zarr
import neuroglancer
import os 
from pydantic import BaseModel, BeforeValidator, Field
from pydantic_zarr.v2 import GroupSpec, ArraySpec
from pydantic_bigstitcher import SpimData2
from pydantic_ome_ngff.v04.multiscale import MultiscaleMetadata
from s3fs import S3FileSystem
import re
from typing_extensions import TypedDict
from neuroglancer.write_annotations import AnnotationWriter

def write_point_annotations(
    path: str,
    points: np.ndarray,
    coordinate_space: neuroglancer.CoordinateSpace,
    rgba: tuple[int, int, int, int] = (127, 255, 127, 255),
) -> None:
    writer = AnnotationWriter(
        coordinate_space=coordinate_space,
        annotation_type="point",
        properties=[
            neuroglancer.AnnotationPropertySpec(id="size", type="float32"),
            neuroglancer.AnnotationPropertySpec(id="point_color", type="rgba"),
        ],
    )

    [writer.add_point(c, size=10, point_color=rgba) for c in points]
    writer.write(path)

def write_line_annotations(
        path: str,
        lines: np.ndarray,
        coordinate_space: neuroglancer.CoordinateSpace,
        rgba: tuple[int, int, int, int] = (127,255,127,255)) -> None:
    ...

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

def tile_coord_to_image_name(coord: TileCoordinate) -> str:
    """
    {'x': 0, 'y': 0, 'ch': 561'} -> "tile_x_0000_y_0002_z_0000_ch_488.zarr/"
    """
    cx = coord["x"]
    cy = coord["y"]
    cz = coord["z"]
    cch = coord['ch']
    return f'tile_x_{cx:04}_y_{cy:04}_z_{cz:04}_ch_{cch}'

def image_name_to_tile_coord(image_name: str) -> TileCoordinate:
    coords: TileCoordinate = {}
    for index_str in ('x', 'y', 'z', 'ch'):    
        prefix = f'_{index_str}_'
        matcher = re.compile(f'{prefix}[0-9]*')
        matches = matcher.findall(image_name)
        if len(matches) > 1:
            raise ValueError(f'Too many matches! The string {image_name} is ambiguous.')
        substr = matches[0][len(prefix):]
        if index_str == 'ch':
            coords[index_str] = substr
        else:
            coords[index_str] = int(substr)
    return coords

class PointsGroup(GroupSpec[InterestPointsGroupMeta, InterestPointsMembers]):
    members: InterestPointsMembers
    ...

from pydantic_bigstitcher import ViewSetup

def parse_bigstitcher_xml_from_s3(base_url: str) -> SpimData2:
    xml_url = os.path.join(base_url, 'bigstitcher.xml')
    bs_xml = S3FileSystem(anon=True).cat_file(xml_url)
    bs_model = SpimData2.from_xml(bs_xml)
    return bs_model

def get_tilegroup_s3_url(model: SpimData2) -> str:
    bucket = model.sequence_description.image_loader.s3bucket
    image_root_url = model.sequence_description.image_loader.zarr.path
    return os.path.join(f's3://{bucket}', image_root_url)

def save_interest_points(bs_model: SpimData2, base_url: str, out_prefix: str):
    tilegroup_url = get_tilegroup_s3_url(bs_model)

    view_setup_dict: dict[str, ViewSetup] = {v.ident: v for v in bs_model.sequence_description.view_setups.view_setup}

    for file in bs_model.view_interest_points.data:
        setup_id = file.setup
        tile_name = view_setup_dict[setup_id].name
        alignment_url = os.path.join(base_url, 'interestpoints.n5', file.path)
        image_url = os.path.join( 
                tilegroup_url, 
                f'{tile_name}.zarr')

        save_points_tile(
            tile_name=tile_name,
            image_url=image_url,
            alignment_url=alignment_url,
            out_prefix=out_prefix
            )


def get_url(node: zarr.Group | zarr.Array) -> str:
    store = node.store
    if hasattr(store, "path"):
        if hasattr(store, "fs"):
            if isinstance(store.fs.protocol, Sequence):
                protocol = store.fs.protocol[0]
            else:
                protocol = store.fs.protocol
        else:
            protocol = "file"

        # fsstore keeps the protocol in the path, but not s3store
        if "://" in store.path:
            store_path = store.path.split("://")[-1]
        else:
            store_path = store.path
        return f"{protocol}://{os.path.join(store_path, node.path)}"
    else:
        msg = (
            f"The store associated with this object has type {type(store)}, which "
            "cannot be resolved to a url"
        )
        raise ValueError(msg)

def create_neuroglancer_state(
                    image_url: str,
                    points_host: str,
                    points_path: str,
                    layer_per_tile: bool = False,
                    wavelength: str = '488'
                    ):
    from neuroglancer import ImageLayer, AnnotationLayer, ViewerState, CoordinateSpace
    image_group = zarr.open(store=image_url, path='')

    image_sources = {}
    points_sources = {}
    space = CoordinateSpace(
    names=['z','y','x'], 
    scales=[100,] * 3, 
    units=['nm',] * 3)
    state = ViewerState(dimensions=space)
    image_shader_controls = {'normalized': {'range': [0, 255], "window": [0, 255]}}
    annotation_shader = r'void main(){setPointMarkerSize(prop_size()); setColor(prop_point_color());}'
    for name, sub_group in filter(lambda kv: f'ch_{wavelength}' in kv[0], image_group.groups()):
        image_sources[name] = f'zarr://{get_url(sub_group)}'
        points_fname = name.removesuffix('.zarr') + '.precomputed'
        points_sources[name] = os.path.join(f'precomputed://{points_host}', points_path, points_fname)

    if layer_per_tile:
        for name, im_source in image_sources:
            point_source = points_sources[name]
            state.layers.append(
                name=name, 
                layer=ImageLayer(
                    source=im_source,
                    shaderControls=image_shader_controls
                ))
            state.layers.append(name=name, layer=AnnotationLayer(source=point_source, shader=annotation_shader))
    else:
        state.layers.append(
            name="images", 
            layer=ImageLayer(
                source=list(image_sources.values()),
                shader_controls=image_shader_controls))
        state.layers.append(
            name="points",
            layer=AnnotationLayer(source=list(points_sources.values()), shader=annotation_shader))

    return state

def save_points_tile(
        tile_name: str,
        image_url: str,
        alignment_url: str,
        out_prefix: str,):
    """
    e.g. dataset = 'exaSPIM_3163606_2023-11-17_12-54-51'
        alignment_id = 'alignment_2024-01-09_05-00-44'

    N5 is organized according to the structure defined here: https://github.com/PreibischLab/multiview-reconstruction/blob/a566bf4d6d35a7ab00d976a8bf46f1615b34b2d0/src/main/java/net/preibisch/mvrecon/fiji/spimdata/interestpoints/InterestPointsN5.java#L54
    """
    multi_meta = MultiscaleMetadata(**zarr.open(image_url).attrs.asdict()['multiscales'][0])
    scale = multi_meta.datasets[0].coordinateTransformations[0].scale
    trans = multi_meta.datasets[0].coordinateTransformations[1].translation
    base_coords = {axis.name: {'scale': s, 'translation': t} for axis, s, t in zip(multi_meta.axes, scale, trans)}
    alignment_store = zarr.N5FSStore(alignment_url)

    matches = zarr.open_group(
        store=alignment_store, 
        path='correspondences', 
        mode='r')
    points = zarr.open_group(
        store=alignment_store, 
        path='interestpoints', 
        mode='r')

    point_loc_stored = points['loc']
    # points are saved as x, y, z triplets
    point_loc = np.zeros((point_loc_stored.shape[0], point_loc_stored.shape[1] + 1), dtype=point_loc_stored.dtype)
    point_loc[:,:-1] = point_loc_stored[:] 

    # apply base translation, scaling will be handled by the coordinate space of the layer
    for idx, dim in enumerate(('x','y','z')):
        local_scale = base_coords[dim]['scale']
        local_trans = base_coords[dim]['translation']
        point_loc[:, idx] += local_trans / local_scale
    
    annotation_scales = [base_coords[dim]['scale'] for dim in ('x','y','z','t')]
    annotation_units = ['um', 'um', 'um', 's']
    annotation_space = neuroglancer.CoordinateSpace(
        names=['x','y','z','t'],
        scales=annotation_scales,
        units=annotation_units
        )
    point_color = tile_coordinate_to_rgba(image_name_to_tile_coord(tile_name))
    write_point_annotations(
        f'{out_prefix}/{tile_name}.precomputed', 
        points = point_loc,
        coordinate_space=annotation_space, 
        rgba=point_color)

def tile_coordinate_to_rgba(coord: TileCoordinate) -> tuple[int, int, int, int]:
    """
    generate an RGBA value from a tile coordinate
    """
    mod_map = {}
    for key in ('x','y','z'):
        mod_map[key] = coord[key] % 2
    lut = {
        (0, 0, 0) : ((255,0,0,255)),
        (1, 0, 0) : ((0,255,0,255)),
        (0, 1, 0) : ((0,0,255,255)),
        (1, 1, 0) : ((255,255,0,255)),
        (0, 0, 1) : ((0,255,255,255)),
        (1, 0, 1) : ((191,191,191,255)),
        (0, 1, 1) : ((0,128,128,255)),
        (1, 1, 1) : ((128,128,0,255)),
        }
    
    return lut[tuple(mod_map.values())]