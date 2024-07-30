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
from matchviz.annotation import write_line_annotations, write_point_annotations
import logging
import fsspec
from neuroglancer import ImageLayer, AnnotationLayer, ViewerState, CoordinateSpace
from matchviz.neuroglancer_styles import NeuroglancerViewerStyle
from functools import reduce
from concurrent.futures import ThreadPoolExecutor, as_completed
from matplotlib import colors
import matplotlib.pyplot as plt
from xarray_ome_ngff import read_multiscale_group

pool = ThreadPoolExecutor(max_workers=16)

OUT_PREFIX = "tile_alignment_visualization"


class Tx(TypedDict):
    scale: float
    trans: float


# we assume here that there's no need to parametrize t
class Coords(TypedDict):
    x: Tx
    y: Tx
    z: Tx


def plot_points(points_df: pl.DataFrame, image_group_path: str):
    images_xarray = read_multiscale_group(
        zarr.open_group(image_group_path, mode="r"),
        array_wrapper={"name": "dask_array", "config": {"chunks": "auto"}},
    )

    loc_xyz = np.array(points_df["loc_xyz"].to_list())

    fig, axs = plt.subplots(ncols=2, nrows=2, dpi=200, figsize=(8, 8))

    dims = ("x", "y", "z")
    img = images_xarray["4"].drop_vars(("t", "c")).squeeze()
    pairs = ("z", "y"), None, ("z", "x"), ("x", "y")

    for idx, pair in enumerate(pairs):
        if pair is not None:
            plot_x, plot_y = sorted(pair)
            axis = axs.ravel()[idx]
            proj_dim = tuple(set(dims) - set(pair))[0]
            proj = img.max((proj_dim)).compute()
            proj.name = f"proj_{proj_dim}"
            proj.plot.imshow(
                x=plot_x,
                y=plot_y,
                ax=axis,
                robust=True,
                norm=colors.LogNorm(),
                cmap="gray_r",
            )

            axis.scatter(
                loc_xyz[:, dims.index(plot_x)],
                loc_xyz[:, dims.index(plot_y)],
                marker="o",
                facecolor="y",
                edgecolor="y",
                alpha=0.1,
            )

            axis.set_xlabel(plot_x)
            axis.set_ylabel(plot_y)

    fig.savefig("foo.svg")


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


def read_bigstitcher_xml(url: str) -> SpimData2:
    fs, path = fsspec.url_to_fs(url)
    bs_xml = fs.cat_file(path)
    bs_model = SpimData2.from_xml(bs_xml)
    return bs_model


def get_tilegroup_s3_url(model: SpimData2) -> str:
    bucket = model.sequence_description.image_loader.s3bucket
    image_root_url = model.sequence_description.image_loader.zarr.path
    return os.path.join(f"s3://{bucket}", image_root_url)


def translate_point(point: Sequence[float], params: Sequence[float]):
    return np.add(point, params)


def scale_point(point: Sequence[float], params: Sequence[float]):
    return np.multiply(point, params)


def translate_points(points_df: pl.DataFrame, coords: Coords):
    """
    Apply a translation
    """
    col = "loc_xyz"
    dims = ("x", "y", "z")
    col_index = points_df.columns.index(col)
    local_scale = [coords[dim]["scale"] for dim in dims]
    local_trans = [coords[dim]["trans"] for dim in dims]

    trans = np.divide(local_trans, local_scale)  # type: ignore
    new_col = translate_point(points_df[col].to_list(), trans)

    return points_df.clone().replace_column(
        col_index, pl.Series(name=col, values=new_col.tolist())
    )


def scale_points(points_df: pl.DataFrame, coords: Coords):
    col = "loc_xyz"
    dims = ("x", "y", "z")
    col_index = points_df.columns.index(col)

    scale = [coords[dim]["scale"] for dim in dims]  # type: ignore
    new_col = scale_point(points_df[col].to_list(), scale)

    return points_df.clone().replace_column(
        col_index, pl.Series(name=col, values=new_col.tolist())
    )


def transform_points(coords: Coords, points: np.ndarray):
    translate_points(coords, points)
    scale_points(coords, points)


def load_points(url: str) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """
    Load interest points and optionally correspondences from a bigstitcher-formatted n5 group as
    polars dataframes.
    """
    logger = logging.getLogger(__name__)
    store = zarr.N5FSStore(url, mode="r")
    interest_points_group = zarr.open_group(
        store=store, path="interestpoints", mode="r"
    )

    if "id" not in interest_points_group:
        raise ValueError(
            f"Failed to find expected n5 dataset at {get_url(interest_points_group)}/id"
        )
    if "loc" not in interest_points_group:
        raise ValueError(
            f"Failed to find expected n5 dataset at {get_url(interest_points_group)}/loc"
        )

    correspondences_group = zarr.open_group(
        store=store, path="correspondences", mode="r"
    )

    matches_exist = "data" in correspondences_group
    if not matches_exist:
        logger.info("No matches found.")

    # points are saved as [num_points, [x, y, z]]
    loc = interest_points_group["loc"][:]

    ids = interest_points_group["id"][:]
    ids_list = ids.squeeze().tolist()

    if matches_exist:
        idmap_parsed = parse_idmap(correspondences_group.attrs["idMap"])
        # get the self id, might not be robust
        match = re.search(r"viewSetupId_(\d+)", url)
        if match is None:
            raise ValueError(f"Could not infer id_self from {url}")

        id_self = int(match.group(1))

        # map from pair index to image id
        remap = {value: key[1] for key, value in idmap_parsed.items()}

        # matches are saved as [num_matches, [point_self, point_other, match_id]]
        matches = np.array(correspondences_group["data"])

        # replace the pair id value with an actual image index reference in the last column
        matches[:, -1] = np.vectorize(remap.get)(matches[:, -1])

        match_result = pl.DataFrame(
            {
                "point_self": matches[:, 0],
                "point_other": matches[:, 1],
                "id_self": [id_self] * matches.shape[0],
                "id_other": matches[:, 2],
            }
        )
    else:
        match_result = None

    return pl.DataFrame({"id": ids_list, "loc_xyz": loc}), match_result


def ome_ngff_to_coords(url: str) -> Coords:
    multi_meta = MultiscaleMetadata(
        **zarr.open_group(url, mode="r").attrs.asdict()["multiscales"][0]
    )
    scale = multi_meta.datasets[0].coordinateTransformations[0].scale
    trans = multi_meta.datasets[0].coordinateTransformations[1].translation
    return {
        axis.name: {"scale": s, "trans": t}
        for axis, s, t in zip(multi_meta.axes, scale, trans)
    }  # type: ignore


def get_tile_coords(bs_model: SpimData2) -> dict[int, Coords]:
    """
    Get the coordinates of all the tiles referenced in bigstitcher xml data. Returns a dict with int
    keys (id numbers of tiles) and Coords values ()
    """
    tile_coords: dict[int, Coords] = {}
    tilegroup_url = get_tilegroup_s3_url(bs_model)

    view_setup_dict: dict[str, ViewSetup] = {
        v.ident: v for v in bs_model.sequence_description.view_setups.view_setups
    }

    for file in bs_model.view_interest_points.data:
        setup_id = file.setup
        tile_name = view_setup_dict[setup_id].name
        image_url = os.path.join(tilegroup_url, f"{tile_name}.zarr")
        _coords = ome_ngff_to_coords(image_url)
        tile_coords[int(setup_id)] = _coords

    return tile_coords


def save_interest_points(bs_model: SpimData2, base_url: str, out_prefix: str):
    """
    Save interest points for all tiles as collection of neuroglancer precomputed annotations. One
    collection of annotations will be generated per image described in the bigstitcher metadata under
    the directory name <out_prefix>/<image_name>.precomputed
    """

    view_setup_dict: dict[int, ViewSetup] = {
        int(v.ident): v for v in bs_model.sequence_description.view_setups.view_setups
    }

    # generate a coordinate grid for all the images
    tile_coords: dict[int, Coords] = get_tile_coords(bs_model=bs_model)
    if bs_model.view_interest_points is None:
        raise ValueError("No view interest points were found in the bigsti")
    for file in bs_model.view_interest_points.data:
        setup_id = int(file.setup)
        tile_name = view_setup_dict[setup_id].name
        # todo: use pydantic zarr models to formalize this path
        alignment_url = os.path.join(
            base_url, "interestpoints.n5", file.path.split("/")[0]
        )

        save_annotations(
            image_id=setup_id,
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


def get_percentiles(image_url: str) -> tuple[int, int]:
    """
    Get the 5th and 95th percentiles from of the smallest array in a multiscale group
    """
    group = zarr.open_group(image_url, mode="r")
    arrays_sorted = sorted(group.arrays(), key=lambda kv: np.prod(kv[1].shape))
    smallest = arrays_sorted[0][1][:]
    return np.percentile(smallest, (5, 95))


def get_histogram_bounds_batch(group_urls: tuple[str, ...]):
    return tuple(pool.map(get_percentiles, group_urls))


def create_neuroglancer_state(
    image_url: str,
    points_url: str,
    style: NeuroglancerViewerStyle,
):
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

    state = ViewerState(
        dimensions=space, cross_section_scale=1000, projection_scale=500_000
    )

    # read the smallest images in the pyramid
    subgroups = tuple(image_group.groups())
    subgroup_urls = tuple(get_url(g) for _, g in subgroups)
    bounds = get_histogram_bounds_batch(subgroup_urls)
    histogram_bounds = {k: v for k, v in zip(image_group.group_keys(), bounds)}

    for fname, sub_group in subgroups:
        subgroup_url = get_url(sub_group)
        point_dir = fname.removesuffix(".zarr") + ".precomputed"
        point_url = f"{points_url}/{point_dir}"
        if points_fs.exists(os.path.join(points_path, point_dir)):
            image_sources[fname] = f"zarr://{subgroup_url}"
            points_sources[fname] = f"precomputed://{point_url}"

    # bias the histogram towards the brighter values
    hist_min, hist_max = reduce(
        lambda old, new: (max(old[0], new[0]), max(old[1], new[1])),
        histogram_bounds.values(),
    )
    window_min = int(hist_min) - abs(int(hist_min) - int(hist_max)) // 3
    if window_min < 0:
        window_min = 0
    window_max = int(hist_max) + abs(int(hist_min) - int(hist_max)) // 3
    image_shader_controls = {
        "normalized": {
            "range": [int(hist_min), int(hist_max)],
            "window": [window_min, window_max],
        }
    }
    annotation_shader = r"void main(){setColor(prop_point_color());}"

    if style == "images_split":
        for fname, im_source in image_sources.items():
            point_source = points_sources[fname]
            coordinate = image_name_to_tile_coord(fname)
            name_base = f"x={coordinate['x']}, y={coordinate['y']}, z={coordinate['z']}, ch={coordinate['ch']}"
            color = tile_coordinate_to_rgba(coordinate)
            color_str = "#{0:02x}{1:02x}{2:02x}".format(*color)
            shader = (
                "#uicontrol invlerp normalized()\n"
                f'#uicontrol vec3 color color(default="{color_str}")\n'
                "void main(){{emitRGB(color * normalized());}}"
            )

            state.layers.append(
                name=f"{name_base}/img",
                layer=ImageLayer(
                    source=im_source,
                    shaderControls=image_shader_controls,
                    shader=shader,
                ),
            )
            state.layers.append(
                name=f"{name_base}/geom",
                layer=AnnotationLayer(source=point_source, shader=annotation_shader),
            )
    elif style == "images_combined":
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
    else:
        msg = f"Style {style} not recognized. Style must be one of 'images_combined' or 'images_split'"
        raise ValueError(msg)
    return state


def save_annotations(
    image_id: int,
    tile_name: str,
    alignment_url: str,
    out_prefix: str,
    tile_coords: dict[int, Coords],
):
    """
    Load points and correspondences (matches) between a single tile an all other tiles, and save as neuroglancer
    precomputed annotations.

        e.g. dataset = 'exaSPIM_3163606_2023-11-17_12-54-51'
        alignment_id = 'alignment_2024-01-09_05-00-44'

    N5 is organized according to the structure defined here: https://github.com/PreibischLab/multiview-reconstruction/blob/a566bf4d6d35a7ab00d976a8bf46f1615b34b2d0/src/main/java/net/preibisch/mvrecon/fiji/spimdata/interestpoints/InterestPointsN5.java#L54

    If matches are not point, then just the interest points will be saved.
    """

    logger = logging.getLogger(__name__)
    logger.setLevel("INFO")
    logger.info(f"Saving annotations for image id {image_id}")
    points_url = os.path.join(out_prefix, f"points/{tile_name}.precomputed")
    lines_url = os.path.join(out_prefix, f"matches/{tile_name}.precomputed")

    # remove trailing slash
    alignment_url = alignment_url.rstrip("/")
    alignment_store = zarr.N5FSStore(alignment_url)
    base_coords = tile_coords[image_id]

    match_group = zarr.open_group(
        store=alignment_store, path="beads/correspondences", mode="r"
    )

    to_access: tuple[int, ...] = (image_id,)
    id_map_normalized = {}
    # tuple of view_setup ids to load
    points_map: dict[int, pl.DataFrame] = {}
    matches_map: dict[int, None | pl.DataFrame] = {}

    matches_exist = "data" in match_group

    if matches_exist:
        id_map = parse_idmap(match_group.attrs.asdict()["idMap"])
        # the idMap attribute uses 0 instead of the actual setup id for the self in this metadata.
        # normalizing replaces that 0 with the actual setup id.
        id_map_normalized = {
            (image_id, *key[1:]): value for key, value in id_map.items()
        }

    for key in id_map_normalized:
        to_access += (key[1],)

    for img_id in to_access:
        new_name = f"tpId_0_viewSetupId_{img_id}"
        new_url = os.path.join(os.path.split(alignment_url)[0], new_name, "beads")

        coords = tile_coords[img_id]
        points_data, match_data = load_points(url=new_url)
        points_data = translate_points(points_data, coords)

        points_map[img_id] = points_data
        matches_map[img_id] = match_data

    annotation_scales = [base_coords[dim]["scale"] for dim in ("x", "y", "z", "t")]  # type: ignore
    annotation_units = ["um", "um", "um", "s"]
    annotation_space = neuroglancer.CoordinateSpace(
        names=["x", "y", "z", "t"], scales=annotation_scales, units=annotation_units
    )

    point_color = tile_coordinate_to_rgba(image_name_to_tile_coord(tile_name))

    line_starts: list[tuple[float, float, float]] = []
    line_stops: list[tuple[float, float, float]] = []
    point_map_self = points_map[image_id]

    # save points for self
    # pad with 0 for time coordinate
    point_data = [(*p, 0.0) for p in point_map_self.get_column("loc_xyz").to_list()]
    id_data = point_map_self.get_column("id").to_list()
    write_point_annotations(
        points_url,
        points=point_data,
        ids=id_data,
        coordinate_space=annotation_space,
        point_color=point_color,
    )
    if matches_map[image_id] is not None:
        match_entry = matches_map[image_id]
        if match_entry is None:
            raise ValueError(f"Missing match data for {image_id}")
        match_entry = cast(pl.DataFrame, match_entry)
        for row in match_entry.rows():
            point_self, point_other, id_other = row
            row_self = point_map_self.row(
                by_predicate=(pl.col("id") == point_self), named=True
            )
            line_start = (
                *row_self["loc_xyz"],
                0.0,
            )  # add a 0 for the time coordinate
            if len(line_start) != 4:
                msg = f"Wrong number of elements in line_start ({len(line_start)})"
                raise ValueError(msg)
            line_start = cast(tuple[float, float, float], line_start)
            try:
                row_other = points_map[id_other].row(
                    by_predicate=(pl.col("id") == point_other), named=True
                )
                line_stop = (
                    *row_other["loc_xyz"],
                    0.0,
                )  # add a 0 for the time coordinate
                if len(line_stop) != 4:
                    msg = f"Wrong number of elements in line_start ({len(line_start)})"
                line_stop = cast(tuple[float, float, float], line_stop)

            except pl.exceptions.NoRowsReturnedError:
                logger.info(f"indexing error with {point_other} into vs_id {id_other}")
                line_stop = line_start

            line_starts.append(line_start)
            line_stops.append(line_stop)

        lines_loc = tuple(zip(*(line_starts, line_stops)))
        write_line_annotations(
            lines_url,
            lines=lines_loc,
            coordinate_space=annotation_space,
            point_color=point_color,
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


def summarize_match(match: pl.DataFrame) -> pl.DataFrame:
    """
    Summarize a dataframe of matches read from the bigstitcher n5 output by grouping by the id
    """
    return (
        match.group_by("id_self", "id_other")
        .agg(pl.col("point_self").count())
        .rename({"point_self": "num_matches"})
    )


def summarize_matches(
    bs_model: SpimData2, matches_dict: dict[str, pl.DataFrame]
) -> pl.DataFrame:
    individual = {k: summarize_match(v) for k, v in matches_dict.items()}
    bs_image_names = tuple(
        map(lambda v: v.rstrip("/").split("/")[-2], individual.keys())
    )
    tile_coords = get_tile_coords(bs_model)
    coords_tokenized = tokenize_tile_coords(tile_coords)

    # include the file name and the normalized tile coordinate to each column
    individual_augmented = {}
    for idx, kv in enumerate(individual.items()):
        k, v = kv
        individual_augmented[k] = v.with_columns(
            bs_image_name=pl.lit(bs_image_names[idx]),
            tile_coords_tokenized=pl.lit(coords_tokenized[idx]),
        )
    return pl.concat(individual_augmented.values()).sort("id_self")


def tokenize(data: Sequence[float]) -> Sequence[int]:
    uniques = sorted(np.unique(data).tolist())
    return [uniques.index(d) for d in data]


def fetch_all_matches(
    n5_interest_points_url: str,
) -> dict[str, pl.DataFrame | BaseException]:
    """
    Load all the match data from the n5 datasets containing it. Takes a url to an n5 group
    emitted by bigstitcher for storing interest points, e.g.
    s3://bucket/dataset/interestpoints.n5/.

    This function uses a thread pool to speed things up.
    """
    fs, path = fsspec.url_to_fs(n5_interest_points_url)
    all_beads = ("s3://" + v for v in fs.glob(os.path.join(path, "*/beads/")))

    matches_dict = {}
    futures_dict = {}

    for bead_path in all_beads:
        fut = pool.submit(load_points, bead_path)
        futures_dict[fut] = bead_path

    for result in as_completed(futures_dict):
        key = futures_dict[result]
        try:
            _, matches = result.result()
            matches_dict[key] = matches
        except BaseException:
            matches_dict = result.exception

    return matches_dict


def tokenize_tile_coords(tile_coords: dict[int, Coords]) -> tuple[tuple[int, int], ...]:
    """
    Convert positions in world coordinates to positions in a space where the only coordinates
    allowed are tuples of integers.
    """

    tile_positions = {
        k: (v["x"]["trans"], v["y"]["trans"]) for k, v in tile_coords.items()
    }
    tile_x, tile_y = zip(*tuple(t for t in tile_positions.values()))
    tile_coords_normed = tuple(zip(tokenize(tile_x), tokenize(tile_y)))
    return tile_coords_normed
