# SPDX-FileCopyrightText: 2024-present Davis Vann Bennett <davis.v.bennett@gmail.com>
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Annotated, Literal, Sequence, cast
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
import polars as pl
from pydantic_bigstitcher import ViewSetup
from matchviz.annotation import write_line_annotations
import logging
import fsspec


class Tx(TypedDict):
    scale: float
    trans: float


# we assume here that there's no need to parametrize t
class Coords(TypedDict):
    x: Tx
    y: Tx
    z: Tx


def parse_idmap(data: dict[str, int]) -> dict[tuple[int, int, str], int]:
    """
    convert {'0,1,beads': 0} to {(0, 1, "beads"): 0}
    """
    parts = map(lambda k: k.split(","), data.keys())
    # convert first two elements to int, leave the last as str
    parts_normalized = map(lambda v: (int(v[0]), int(v[1]), v[2]), parts)
    return dict(zip(parts_normalized, data.values()))


class InterestPointsGroupMeta(BaseModel):
    list_version: str = Field(alias="list version")
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


# we cannot use these classes without overriding from_zarr to deal with n5 not storing
# attributes.json files in intermediate groups
class PointsGroup(GroupSpec[InterestPointsGroupMeta, InterestPointsMembers]):
    members: InterestPointsMembers
    ...


class TileCoordinate(TypedDict):
    x: int
    y: int
    z: int
    ch: Literal["488", "561"]


def tile_coord_to_image_name(coord: TileCoordinate) -> str:
    """
    {'x': 0, 'y': 0, 'ch': 561'} -> "tile_x_0000_y_0002_z_0000_ch_488.zarr/"
    """
    cx = coord["x"]
    cy = coord["y"]
    cz = coord["z"]
    cch = coord["ch"]
    return f"tile_x_{cx:04}_y_{cy:04}_z_{cz:04}_ch_{cch}"


def image_name_to_tile_coord(image_name: str) -> TileCoordinate:
    coords = {}
    for index_str in ("x", "y", "z", "ch"):
        prefix = f"_{index_str}_"
        matcher = re.compile(f"{prefix}[0-9]*")
        matches = matcher.findall(image_name)
        if len(matches) > 1:
            raise ValueError(f"Too many matches! The string {image_name} is ambiguous.")
        substr = matches[0][len(prefix) :]
        if index_str == "ch":
            coords[index_str] = substr
        else:
            coords[index_str] = int(substr)
    coords_out: TileCoordinate = cast(TileCoordinate, coords)
    return coords_out


def parse_bigstitcher_xml_from_s3(base_url: str) -> SpimData2:
    xml_url = os.path.join(base_url, "bigstitcher.xml")
    bs_xml = S3FileSystem(anon=True).cat_file(xml_url)
    bs_model = SpimData2.from_xml(bs_xml)
    return bs_model


def get_tilegroup_s3_url(model: SpimData2) -> str:
    bucket = model.sequence_description.image_loader.s3bucket
    image_root_url = model.sequence_description.image_loader.zarr.path
    return os.path.join(f"s3://{bucket}", image_root_url)


def translate_points(coords: Coords, points: np.ndarray):
    """
    Apply a translation
    """
    # transform one column at a time
    for idx, dim in enumerate(("x", "y", "z")):
        local_scale = coords[dim]["scale"]  # type: ignore
        local_trans = coords[dim]["translation"]  # type: ignore
        points[:, idx] += local_trans / local_scale


def load_points(
    interest_points_url: str, coords: Coords | None
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    e.g. interestpoints.n5/tpId_0_viewSetupId_3/beads
    """
    store = zarr.N5FSStore(interest_points_url, mode="r")
    interest_points_group = zarr.open_group(
        store=store, path="interestpoints", mode="r"
    )
    correspondences_group = zarr.open_group(
        store=store, path="correspondences", mode="r"
    )
    # points are saved as [num_points, [x, y, z]]
    loc = interest_points_group["loc"][:]

    if coords is not None:
        translate_points(coords, loc)

    ids = interest_points_group["id"][:]
    intensity = interest_points_group["intensities"][:]

    idmap_parsed = parse_idmap(correspondences_group.attrs["idMap"])
    # map from pair index to image id
    remap = {value: key[1] for key, value in idmap_parsed.items()}
    # matches are saved as [num_matches, [point_self, point_other, match_id]]
    matches = correspondences_group["data"][:]
    # replace the pair id value with an actual image index reference in the last column
    matches[:, -1] = np.vectorize(remap.get)(matches[:, -1])

    return pl.DataFrame(
        {"id": ids, "loc_xyz": loc, "intensity": intensity}
    ), pl.DataFrame(
        {
            "point_self": matches[:, 0],
            "point_other": matches[:, 1],
            "id_other": matches[:, 2],
        }
    )


def get_tile_coords(bs_model: SpimData2) -> dict[int, Coords]:
    """
    Get the coordinates of all the tiles referenced in bigstitcher xml data. Returns a dict with int
    keys (id numbers of tiles) and Coords values ()
    """
    tile_coords: dict[int, Coords] = {}
    tilegroup_url = get_tilegroup_s3_url(bs_model)
    view_setup_dict: dict[str, ViewSetup] = {
        v.ident: v for v in bs_model.sequence_description.view_setups.view_setup
    }
    for file in bs_model.view_interest_points.data:
        setup_id = file.setup
        tile_name = view_setup_dict[setup_id].name
        image_url = os.path.join(tilegroup_url, f"{tile_name}.zarr")
        multi_meta = MultiscaleMetadata(
            **zarr.open_group(image_url, mode="r").attrs.asdict()["multiscales"][0]
        )
        scale = multi_meta.datasets[0].coordinateTransformations[0].scale
        trans = multi_meta.datasets[0].coordinateTransformations[1].translation
        tile_coords[int(setup_id)] = {
            axis.name: {"scale": s, "translation": t}
            for axis, s, t in zip(multi_meta.axes, scale, trans)
        }  # type: ignore
    return tile_coords


def save_interest_points(bs_model: SpimData2, base_url: str, out_prefix: str):
    """
    Save interest points for all tiles as collection of neuroglancer precomputed annotations. One
    collection of annotations will be generated per image described in the bigstitcher metadata under
    the directory name <out_prefix>/<image_name>.precomputed
    """

    view_setup_dict: dict[int, ViewSetup] = {
        int(v.ident): v for v in bs_model.sequence_description.view_setups.view_setup
    }
    # generate a coordinate grid for all the images
    tile_coords: dict[int, Coords] = get_tile_coords(bs_model=bs_model)

    for file in bs_model.view_interest_points.data:
        setup_id = int(file.setup)
        tile_name = view_setup_dict[setup_id].name
        # todo: use pydantic zarr models to formalize this path
        alignment_url = os.path.join(
            base_url, "interestpoints.n5", file.path.split("/")[0]
        )

        save_points_tile(
            vs_id=setup_id,
            tile_name=tile_name,
            alignment_url=alignment_url,
            out_prefix=out_prefix,
            tile_coords=tile_coords,
        )


def get_url(node: zarr.Group | zarr.Array) -> str:
    """
    Get a URL from a zarr array or group pointing to its location in storage
    """
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
    points_url: str,
    layer_per_tile: bool = False,
):
    from neuroglancer import ImageLayer, AnnotationLayer, ViewerState, CoordinateSpace

    image_group = zarr.open_group(store=image_url, path="", mode="r")
    image_sources = {}
    points_sources = {}
    space = CoordinateSpace(
        names=["z", "y", "x"],
        scales=[
            100,
        ]
        * 3,
        units=[
            "nm",
        ]
        * 3,
    )
    points_fs, points_path = fsspec.url_to_fs(points_url)

    state = ViewerState(dimensions=space)
    image_shader_controls = {"normalized": {"range": [0, 255], "window": [0, 255]}}
    annotation_shader = r"void main(){setColor(prop_point_color());}"

    for name, sub_group in image_group.groups():
        point_dir = name.removesuffix(".zarr") + ".precomputed"
        point_url = f"{points_url}/{point_dir}"
        if points_fs.exists(os.path.join(points_path, point_dir)):
            image_sources[name] = f"zarr://{get_url(sub_group)}"
            points_sources[name] = f"precomputed://{point_url}"

    if layer_per_tile:
        for name, im_source in image_sources:
            point_source = points_sources[name]
            state.layers.append(
                name=name,
                layer=ImageLayer(
                    source=im_source, shaderControls=image_shader_controls
                ),
            )
            state.layers.append(
                name=name,
                layer=AnnotationLayer(source=point_source, shader=annotation_shader),
            )
    else:
        state.layers.append(
            name="images",
            layer=ImageLayer(
                source=list(image_sources.values()),
                shader_controls=image_shader_controls,
            ),
        )
        state.layers.append(
            name="points",
            layer=AnnotationLayer(
                source=list(points_sources.values()), shader=annotation_shader
            ),
        )

    return state


def save_points_tile(
    vs_id: int,
    tile_name: str,
    alignment_url: str,
    out_prefix: str,
    tile_coords: dict[int, Coords],
):
    """
    e.g. dataset = 'exaSPIM_3163606_2023-11-17_12-54-51'
        alignment_id = 'alignment_2024-01-09_05-00-44'

    N5 is organized according to the structure defined here: https://github.com/PreibischLab/multiview-reconstruction/blob/a566bf4d6d35a7ab00d976a8bf46f1615b34b2d0/src/main/java/net/preibisch/mvrecon/fiji/spimdata/interestpoints/InterestPointsN5.java#L54
    """

    # base_coords = tile_coords[vs_id]

    logger = logging.getLogger(__name__)
    # remove trailing slash
    alignment_url = alignment_url.rstrip("/")
    alignment_store = zarr.N5FSStore(alignment_url)
    base_coords = tile_coords[vs_id]
    match_group = zarr.open_group(
        store=alignment_store, path="beads/correspondences", mode="r"
    )

    id_map = parse_idmap(match_group.attrs.asdict()["idMap"])
    # the idMap attribute uses 0 instead of the actual setup id for the self in this metadata.
    # normalizing replaces that 0 with the actual setup id.
    id_map_normalized = {(vs_id, *key[1:]): value for key, value in id_map.items()}

    # tuple of view_setup ids to load
    to_access: tuple[int, ...] = (vs_id,)
    points_map = {}
    matches_map = {}

    for key in id_map_normalized:
        to_access += (key[1],)

    for other_id in to_access:
        new_name = f"tpId_0_viewSetupId_{other_id}"
        new_url = os.path.join(os.path.split(alignment_url)[0], new_name, "beads")

        points_data, match_data = load_points(
            interest_points_url=new_url, coords=tile_coords[other_id]
        )
        points_map[other_id] = points_data
        matches_map[other_id] = match_data

    annotation_scales = [base_coords[dim]["scale"] for dim in ("x", "y", "z", "t")]  # type: ignore
    annotation_units = ["um", "um", "um", "s"]
    annotation_space = neuroglancer.CoordinateSpace(
        names=["x", "y", "z", "t"], scales=annotation_scales, units=annotation_units
    )

    point_color = tile_coordinate_to_rgba(image_name_to_tile_coord(tile_name))

    line_starts = []
    line_stops = []
    point_map_self = points_map[vs_id]

    for row in matches_map[vs_id].rows():
        logger.info("saving tile id {vs_id}")
        point_self, point_other, id_other = row
        line_start = (
            point_map_self.get_column("loc_xyz")[point_self]
            .clone()
            .extend(pl.Series([0.0]))
        )
        loc_xyz_other = points_map[id_other].get_column("loc_xyz")
        try:
            line_stop = loc_xyz_other[point_other].clone().extend(pl.Series([0.0]))
        except IndexError:
            print(f"indexing error with {point_other} into vs_id {id_other}")
            line_stop = line_start
        # add the fake time coordinate
        line_starts.append(line_start)
        line_stops.append(line_stop)

    lines_loc = tuple(zip(*(line_starts, line_stops)))
    write_line_annotations(
        f"{out_prefix}/{tile_name}.precomputed",
        lines=lines_loc,
        coordinate_space=annotation_space,
        rgba=point_color,
    )


def tile_coordinate_to_rgba(coord: TileCoordinate) -> tuple[int, int, int, int]:
    """
    generate an RGBA value from a tile coordinate. This ensures that adjacent tiles have different
    colors. It's not a nice lookup table by any measure.
    """
    mod_map = {}
    for key in ("x", "y", "z"):
        mod_map[key] = coord[key] % 2  # type: ignore
    lut = {
        (0, 0, 0): ((255, 0, 0, 255)),
        (1, 0, 0): ((0, 255, 0, 255)),
        (0, 1, 0): ((0, 0, 255, 255)),
        (1, 1, 0): ((255, 255, 0, 255)),
        (0, 0, 1): ((0, 255, 255, 255)),
        (1, 0, 1): ((191, 191, 191, 255)),
        (0, 1, 1): ((0, 128, 128, 255)),
        (1, 1, 1): ((128, 128, 0, 255)),
    }

    return lut[tuple(mod_map.values())]
