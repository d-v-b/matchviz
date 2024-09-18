from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import re
import time
from typing import Annotated, cast, TYPE_CHECKING
from typing_extensions import TypedDict

from pydantic import BaseModel, BeforeValidator, Field
from pydantic_zarr.v2 import ArraySpec, GroupSpec

from matchviz.annotation import write_line_annotations, write_point_annotations
from matchviz.core import (
    get_url,
    ome_ngff_to_coords,
    parse_url,
    tile_coordinate_to_rgba,
    tokenize_tile_coords,
    translate_points,
)
from matchviz.types import Coords, TileCoordinate

if TYPE_CHECKING:
    pass
import zarr
import structlog
import fsspec
import numpy as np
import polars as pl
from pydantic_bigstitcher import SpimData2, ViewSetup
from yarl import URL
import neuroglancer


def read_bigstitcher_xml(url: URL) -> SpimData2:
    fs, path = fsspec.url_to_fs(str(url))
    bs_xml = fs.cat_file(path)
    bs_model = SpimData2.from_xml(bs_xml)
    return bs_model


def get_tilegroup_url(model: SpimData2) -> URL:
    breakpoint()
    if hasattr(model.sequence_description.image_loader, "s3bucket"):
        bucket = model.sequence_description.image_loader.s3bucket
        image_root_path = model.sequence_description.image_loader.zarr.path
        return URL.build(scheme="s3", authority=bucket, path=image_root_path)
    else:
        return URL(model.sequence_description.image_loader.n5.path)

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


def parse_idmap(data: dict[str, int]) -> dict[tuple[int, int, str], int]:
    """
    convert {'0,1,beads': 0} to {(0, 1, "beads"): 0}
    """
    parts = map(lambda k: k.split(","), data.keys())
    # convert first two elements to int, leave the last as str
    parts_normalized = map(lambda v: (int(v[0]), int(v[1]), v[2]), parts)
    return dict(zip(parts_normalized, data.values()))


def tile_coord_to_image_name(coord: TileCoordinate) -> str:
    """
    {'x': 0, 'y': 0, 'ch': 561'} -> "tile_x_0000_y_0002_z_0000_ch_488.zarr/"
    """
    cx = coord["x"]
    cy = coord["y"]
    cz = coord["z"]
    cch = coord["ch"]
    return f"tile_x_{cx:04}_y_{cy:04}_z_{cz:04}_ch_{cch}"


def load_points_all(url: str) -> dict[str, tuple[pl.DataFrame, pl.DataFrame]]:
    """
    Load interest points and optionally correspondences from a bigstitcher-formatted n5 group as
    polars dataframes for all tiles.
    """
    raise NotImplementedError()
    log = structlog.get_logger(url=url, name=__name__)

    store = zarr.N5FSStore(url=url, mode="r")
    stored = {}
    for name, group in store.groups():
        stored[name] = {""}


def parse_matches(
    *, name: str, data: np.ndarray, id_map: dict[tuple[int, int, str], int]
):
    """
    Convert a name, match data, and an id mapping to a polars dataframe that contains
    pairwise image matching information.
    """
    data_copy = data.copy()

    # get the self id, might not be robust
    match = re.search(r"viewSetupId_(\d+)", name)
    if match is None:
        raise ValueError(f"Could not infer id_self from {name}")

    id_self = int(match.group(1))

    # map from pair index to image id
    remap = {value: key[1] for key, value in id_map.items()}

    # replace the pair id value with an actual image index reference in the last column
    data_copy[:, -1] = np.vectorize(remap.get)(data[:, -1])

    match_result = pl.DataFrame(
        {
            "point_self": data_copy[:, 0],
            "point_other": data_copy[:, 1],
            "id_self": [id_self] * data_copy.shape[0],
            "id_other": data_copy[:, 2],
        }
    )
    return match_result


def load_points_tile(url: str | URL) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """
    Load interest points and optionally correspondences from a bigstitcher-formatted n5 group as
    polars dataframes for a single tile.
    """
    log = structlog.get_logger()
    url_parsed = str(url)
    store = zarr.N5FSStore(url_parsed, mode="r")
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
        log.info(f"No matches found for {url_parsed}.")

    # points are saved as [num_points, [x, y, z]]
    loc = interest_points_group["loc"][:]

    ids = interest_points_group["id"][:]
    ids_list = ids.squeeze().tolist()

    if matches_exist:
        id_map = parse_idmap(correspondences_group.attrs["idMap"])
        matches = np.array(correspondences_group["data"])
        match_result = parse_matches(name=url_parsed, data=matches, id_map=id_map)
    else:
        match_result = None

    return pl.DataFrame({"id": ids_list, "loc_xyz": loc}), match_result


def get_tile_coords(bs_model: SpimData2) -> dict[int, Coords]:
    """
    Get the coordinates of all the tiles referenced in bigstitcher xml data. Returns a dict with int
    keys (id numbers of tiles) and Coords values ()
    """
    tile_coords: dict[int, Coords] = {}
    tilegroup_url = get_tilegroup_url(bs_model)

    view_setup_dict: dict[str, ViewSetup] = {
        v.ident: v for v in bs_model.sequence_description.view_setups.view_setups
    }
    # TODO: don't hard-code zarr stuff in here, lets use the bigsticher metadata itself instead.
    for file in bs_model.view_interest_points.data:
        setup_id = file.setup
        tile_name = view_setup_dict[setup_id].name
        image_url = tilegroup_url.joinpath(f"{tile_name}.zarr")
        _coords = ome_ngff_to_coords(image_url)
        tile_coords[int(setup_id)] = _coords

    return tile_coords


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


class PointsGroup(GroupSpec[InterestPointsGroupMeta, InterestPointsMembers]):
    members: InterestPointsMembers
    ...


def save_annotations(
    *,
    image_id: int,
    tile_name: str,
    alignment_url: URL | str,
    dest_url: URL | str,
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

    log = structlog.get_logger(tile_name=tile_name)
    start = time.time()
    log.info(f"Begin saving annotations for image id {image_id}")
    dest_url_parsed = parse_url(dest_url)
    points_url = dest_url_parsed.joinpath(f"points/{tile_name}.precomputed")
    lines_url = dest_url_parsed.joinpath(f"matches/{tile_name}.precomputed")
    alignment_url_parsed = parse_url(alignment_url)

    log.info(f"Saving points to {points_url}")
    log.info(f"Saving matches to {lines_url}")
    # remove trailing slash
    alignment_store = zarr.N5FSStore(str(alignment_url_parsed).rstrip("/"))

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
        log.info("Found matches.")
        id_map = parse_idmap(match_group.attrs.asdict()["idMap"])
        # the idMap attribute uses 0 instead of the actual setup id for the self in this metadata.
        # normalizing replaces that 0 with the actual setup id.
        id_map_normalized = {
            (image_id, *key[1:]): value for key, value in id_map.items()
        }
    else:
        log.info("No matches found.")
    for key in id_map_normalized:
        to_access += (key[1],)

    for img_id in to_access:
        new_name = f"tpId_0_viewSetupId_{img_id}"
        new_url = alignment_url_parsed.parent.joinpath(new_name, "beads")
        coords = tile_coords[img_id]
        points_data, match_data = load_points_tile(url=new_url)
        points_data = translate_points(points_data, coords)

        points_map[img_id] = points_data
        matches_map[img_id] = match_data

    annotation_scales = [base_coords[dim]["scale"] for dim in ("x", "y", "z", "t")]  # type: ignore
    annotation_units = ["um", "um", "um", "s"]
    annotation_space = neuroglancer.CoordinateSpace(
        names=["x", "y", "z", "t"], scales=annotation_scales, units=annotation_units
    )

    point_color = tile_coordinate_to_rgba(image_name_to_tile_coord(tile_name))

    line_starts: list[tuple[float, float, float, float]] = []
    line_stops: list[tuple[float, float, float, float]] = []
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
        log.info(f"Saving matches to {lines_url}.")
        match_entry = matches_map[image_id]
        if match_entry is None:
            raise ValueError(f"Missing match data for {image_id}")
        match_entry = cast(pl.DataFrame, match_entry)
        for row in match_entry.rows():
            point_self, point_other, id_self, id_other = row
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
            line_start = cast(tuple[float, float, float, float], line_start)
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
                line_stop = cast(tuple[float, float, float, float], line_stop)

            except pl.exceptions.NoRowsReturnedError:
                log.info(f"indexing error with {point_other} into vs_id {id_other}")
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
    log.info(f"Completed saving points / matches after {time.time() - start:0.4f}s.")


def save_interest_points(*, bs_model: SpimData2, alignment_url: URL, dest: URL):
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
        raise ValueError(
            "No view interest points were found in the bigstitcher xml file."
        )

    for file in bs_model.view_interest_points.data:
        setup_id = int(file.setup)
        tile_name = view_setup_dict[setup_id].name
        # todo: use pydantic zarr models to formalize this path
        fname = file.path.split("/")[0]
        _alignment_url = alignment_url.joinpath(f"interestpoints.n5/{fname}")
        save_annotations(
            image_id=setup_id,
            tile_name=tile_name,
            alignment_url=_alignment_url,
            dest_url=dest,
            tile_coords=tile_coords,
        )


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
    # convert absolute bead paths to bigstitcher image names
    
    tile_coords = get_tile_coords(bs_model)
    coords_tokenized = tokenize_tile_coords(tile_coords)
    
    bs_image_names = tuple(
        map(lambda v: v.parts[-2], matches_dict.keys())
    )
    viewsetup_ids = tuple(
        map(lambda v: int(v.split("viewSetupId_")[-1]), bs_image_names)
    )

    # get base image metadata from bigstitcher xml
    bs_view_setups_by_id = {
        int(k.ident): k for k in bs_model.sequence_description.view_setups.view_setups
    }

    # polars dataframes keyed by bigstitcher image names
    individual_summaries = {
        viewsetup_ids[index]: summarize_match(v)
        for index, v in enumerate(matches_dict.values())
    }

    # include the file name and the normalized tile coordinate to each column
    individual_augmented = {}

    for idx, (k, v) in enumerate(individual_summaries.items()):
        individual_augmented[k] = v.with_columns(
            image_name=pl.lit(bs_image_names[idx],
            tile_coords = pl.lit(coords_tokenized[idx]))
        )
    return (
        pl.concat(individual_augmented.values())
        .select(
            [
                pl.col("id_self"),
                pl.col("x_self"),
                pl.col("y_self"),
                pl.col("z_self"),
                pl.col("id_other"),
                pl.col("x_other"),
                pl.col("y_other"),
                pl.col("z_other"),
                pl.col("num_matches"),
            ]
        )
        .sort("id_self")
    )


def fetch_all_matches(
    n5_interest_points_url: URL, 
    pool: ThreadPoolExecutor
) -> dict[str, pl.DataFrame | BaseException | None]:
    """
    Load all the match data from the n5 datasets containing it. 
    Takes a URL to an n5 group emitted by bigstitcher for storing interest points, e.g.
    s3://bucket/dataset/interestpoints.n5/.

    This function uses a thread pool to speed things up.
    """
    fs, path = fsspec.url_to_fs(str(str(n5_interest_points_url)))
    all_beads = (n5_interest_points_url.with_path(v) for v in fs.glob(os.path.join(path, "*/*")))

    matches_dict: dict[str, pl.DataFrame | BaseException | None] = {}
    futures_dict = {}

    for bead_path in all_beads:
        fut = pool.submit(load_points_tile, bead_path)
        futures_dict[fut] = bead_path

    for result in as_completed(futures_dict):
        key = futures_dict[result]
        try:
            _, matches = result.result()
            matches_dict[key] = matches
        except BaseException:
            matches_dict[key] = result.exception()

    return matches_dict
