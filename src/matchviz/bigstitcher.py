from __future__ import annotations
from itertools import accumulate, product
import pydantic_bigstitcher.transform
from zarr import N5FSStore
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import re
import time
from typing import Annotated, Any, Iterable, Literal, cast, TYPE_CHECKING, Iterable
from typing_extensions import TypedDict, deprecated
from pydantic import BaseModel, BeforeValidator, Field
from pydantic_zarr.v2 import ArraySpec, GroupSpec
from functools import partial
from matchviz.annotation import write_line_annotations, write_point_annotations
import random
import colorsys
from matchviz.core import (
    get_url,
    parse_url,
    tile_coordinate_to_rgba,
    translate_points_xyz,
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
import pydantic_bigstitcher
from yarl import URL
import neuroglancer
np.random.seed(0)
def read_bigstitcher_xml(url: URL, anon: bool = True) -> SpimData2:
    fs, path = fsspec.url_to_fs(str(url), anon=anon)
    bs_xml = fs.cat_file(path)
    bs_model = SpimData2.from_xml(bs_xml)
    return bs_model

def view_bdv(bs_model: SpimData2, host: URL | None = None):
    state = bdv_to_neuroglancer(bs_model, host)
    viewer = neuroglancer.Viewer()
    viewer.set_state(state)
    print(viewer)

def bdv_to_neuroglancer(
        xml_path: URL, 
        *,
        anon: bool=True,
        host: URL | None = None,
        view_setups: Literal["all"] | Iterable[int] = 'all',
        channels: Literal["all"] | Iterable[int] = 'all',
        display_settings: dict[str, Any]) -> neuroglancer.ViewerState:
    
    bs_model = read_bigstitcher_xml(xml_path, anon=anon)
    unit = bs_model.sequence_description.view_setups.elements[0].voxel_size.unit
    scales = tuple(map(float, bs_model.sequence_description.view_setups.elements[0].voxel_size.size.split(" ")))
    dimension_names = ["z", "y", "x"]
    units = [unit] * len(dimension_names)

    output_space = neuroglancer.CoordinateSpace(
        names=dimension_names,
        scales=[
            scales[0],
        ] * len(dimension_names),
        units=units,
    )

    state = neuroglancer.ViewerState(
        dimensions=output_space
    )

    if bs_model.base_path.path == '.':
        if host is None:
            base_path = xml_path.parent
        else:
            base_path = host

    else:
        raise ValueError(f'Base path {bs_model.base_path.path} is not supported')
    
    image_loader = bs_model.sequence_description.image_loader
    
    if image_loader.fmt == "bdv.multimg.zarr":
        image_format = "zarr"
        if image_loader.zarr.type == 'absolute':
            if hasattr(image_loader, "s3bucket"):
                prefix = URL(f's3://{image_loader.s3bucket}')
            else:
                if host is not None:
                    prefix=host
                else:
                    prefix=URL('file:///')
            container_root = prefix / image_loader.zarr.path
        else:
            container_root = base_path / image_loader.zarr.path

    elif image_loader.fmt == "bdv.n5":
        image_format = "n5"
        if image_loader.n5.type == 'absolute':
            if hasattr(image_loader, "s3bucket"):
                prefix = f's3://{image_loader.s3bucket}'
            else:
                if host is not None:
                    prefix=host
                else:
                    prefix=URL('file:///')
            container_root = URL(prefix) / image_loader.zarr.path
        else:
            container_root = base_path / image_loader.n5.path
    else:
        raise ValueError(f"Unknown image format: {bs_model.sequence_description.image_loader.fmt}")

    for view_reg in bs_model.view_registrations.elements:
        vs_id = view_reg.setup

        if view_setups != 'all' and int(vs_id) not in view_setups:
            continue

        maybe_view_setups = tuple(
            filter(
                lambda v: v.ident == vs_id, 
                bs_model.sequence_description.view_setups.elements
                )
            )
        
        if len(maybe_view_setups) > 1:
            raise ValueError(f"Ambiguous setup id {vs_id}.")
        
        view_setup = maybe_view_setups[0]

        if channels != 'all' and int(view_setup.attributes.channel) not in channels:
            continue

        if image_format == 'zarr':
            image_name = view_setup.name
            image_path = container_root / f'{image_name}.zarr'
        else:
            image_name = f'setup{view_reg.setup}/timepoint{view_reg.timepoint}'
            image_path = container_root / image_name

        # bigstitcher stores the transformations in reverse order, i.e. the 
        # last transform in the series is the first one in the list

        tforms = tuple(t.to_transform() for t in reversed(view_reg.view_transforms))
        # for zarr data, remove the translation to nominal grid transform
        tforms_filtered = tuple(filter(lambda v: v.name != "Translation to Nominal Grid", tforms))
        hoaffines = [t.transform for t in tforms_filtered]
        
        composed_hoaffs = tuple(accumulate(hoaffines, partial(compose_hoaffines, dimensions=dimension_names)))

        # neuroglancer expects ndim x ndim + 1 matrix
        matrix = hoaffine_to_array(
            composed_hoaffs[-1], 
            dimensions=dimension_names
            )[:-1]
        
        # convert translations into the output coordinate space
        base_scale_dict = {dim: float(x) for x, dim in zip(view_setup.voxel_size.size.split(' '), ('x','y','z')) }
        scales = [base_scale_dict[dim] for dim in dimension_names]
        scales = [min(scales)] * len(scales)
        image_name = ''

        image_source = f'{image_format}://{image_path}'
        h,s,l = random.random(), 0.5 + random.random()/2.0, 0.4 + random.random()/5.0
        color = [int(256*i) for i in colorsys.hls_to_rgb(h,l,s)]
        color_str = "#{0:02x}{1:02x}{2:02x}".format(*color)
        shader = (
            "#uicontrol invlerp normalized()\n"
            f'#uicontrol vec3 color color(default="{color_str}")\n'
            "void main(){{emitRGB(color * normalized());}}"
        )
        image_shader_controls = {
        "normalized": {
            "range": [display_settings['start'], display_settings['stop']],
            "window": [display_settings['min'], display_settings['max']],
        }
    }
        input_space = neuroglancer.CoordinateSpace(
            names=dimension_names,
            units=units,
            scales=scales
        )
        
        image_transform = neuroglancer.CoordinateSpaceTransform(
            output_dimensions=input_space,
            input_dimensions=input_space,
            source_rank=len(dimension_names),
            matrix=matrix)
        # TODO: choose a source class based on whether a host was provided, so that 
        # local files can potentially be viewed.
        state.layers.append(
            name=image_name,
            layer=neuroglancer.ImageLayer(
                source=neuroglancer.LayerDataSource(
                    url=image_source, 
                    transform=image_transform),
                shader_controls=image_shader_controls,
                shader=shader,
                blend='additive'
            ))

    return state

# TODO: use this instead of the raw tuple
@dataclass(frozen=True)
class IdMap:
    id_self: int
    id_other: int
    matches_path: str


def get_tilegroup_url(model: SpimData2) -> URL:
    if hasattr(model.sequence_description.image_loader, "s3bucket"):
        bucket = model.sequence_description.image_loader.s3bucket
        image_root_path = model.sequence_description.image_loader.zarr.path
        return URL.build(scheme="s3", authority=bucket, path=image_root_path)
    else:
        # this will be a relative URL
        return URL(model.sequence_description.image_loader.n5.path)


@deprecated(
    "This functionality is deprecated. Use image metadata or the bigstitcher xml file instead."
)
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


@deprecated(
    "This functionality is deprecated. Use image metadata or the bigstitcher xml file instead."
)
def tile_coord_to_image_name(coord: TileCoordinate) -> str:
    """
    {'x': 0, 'y': 0, 'ch': 561'} -> "tile_x_0000_y_0002_z_0000_ch_488.zarr/"
    """
    cx = coord["x"]
    cy = coord["y"]
    cz = coord["z"]
    cch = coord["ch"]
    return f"tile_x_{cx:04}_y_{cy:04}_z_{cz:04}_ch_{cch}"


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


def load_points_tile(
    *, store: str | URL | N5FSStore, path: str, anon: bool = True
) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """
    Load interest points and optionally correspondences from a bigstitcher-formatted n5 group as
    polars dataframes for a single tile.
    """
    log = structlog.get_logger()

    if isinstance(store, (str, URL)):
        store_parsed = zarr.N5FSStore(str(store), mode="r", anon=anon)
    else:
        store_parsed = store

    interest_points_group = zarr.open_group(
        store=store_parsed, path=f"{path}/interestpoints", mode="r"
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
        store=store_parsed, path=f"{path}/correspondences", mode="r"
    )

    matches_exist = "data" in correspondences_group
    if not matches_exist:
        log.info(f"No matches found for {get_url(store_parsed)}.")

    # points are saved as [num_points, [x, y, z]]
    loc = interest_points_group["loc"][:]

    ids = interest_points_group["id"][:]
    ids_list = ids.squeeze().tolist()

    if matches_exist:
        id_map = parse_idmap(correspondences_group.attrs["idMap"])
        matches = np.array(correspondences_group["data"])
        match_result = parse_matches(name=path, data=matches, id_map=id_map)
    else:
        match_result = None
    return pl.DataFrame(
        {"id": ids_list, "loc_xyz": loc},
        schema={"id": pl.Int64, "loc_xyz": pl.Array(pl.Float64, 3)},
    ), match_result


def get_tile_coords(bs_model: SpimData2, nominal_grid=True) -> dict[int, Coords]:
    """
    Get the source coordinates of the tiles referenced in bigstitcher xml data. Returns a dict with int
    keys (id numbers of tiles) and Coords values ()
    """
    # get all transformations for all view_setups
    coords_by_view_setup: dict[int, Coords] = {}
    for vs, vr in zip(
        bs_model.sequence_description.view_setups.view_setups,
        bs_model.view_registrations.elements,
    ):
        transforms = tuple(v.to_transform() for v in vr.view_transforms)
        # note that we reverse the order of the transformations, because bigsticher appends new transformations
        # to the front of the list, but we want to iterate with the first applied transform first.
        translation_array = np.array(
            tuple(tuple(t.transform.translation.values()) for t in reversed(transforms))
        )
        translation_array_summed = np.cumsum(translation_array, axis=0)
        if nominal_grid:
            for row in translation_array_summed:
                if not np.all(row == 0):
                    final_translation = row
                    break
        else:
            final_translation = translation_array_summed[-1]
        bs_voxel_size_xyz = tuple(map(float, vs.voxel_size.size.split(" ")))
        coords_by_view_setup[int(vs.ident)] = {
            "z": {"trans": final_translation[2], "scale": bs_voxel_size_xyz[2]},
            "y": {"trans": final_translation[1], "scale": bs_voxel_size_xyz[1]},
            "x": {"trans": final_translation[0], "scale": bs_voxel_size_xyz[0]},
        }
    return coords_by_view_setup


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
    bs_model: SpimData2,
    image_id: int,
    alignment_url: URL | str,
    dest_url: URL | str,
):
    """
    Load points and correspondences (matches) between a single tile an all other tiles, and save as neuroglancer
    precomputed annotations.

        e.g. dataset = 'exaSPIM_3163606_2023-11-17_12-54-51'
        alignment_id = 'alignment_2024-01-09_05-00-44'

    N5 is organized according to the structure defined here: https://github.com/PreibischLab/multiview-reconstruction/blob/a566bf4d6d35a7ab00d976a8bf46f1615b34b2d0/src/main/java/net/preibisch/mvrecon/fiji/spimdata/interestpoints/InterestPointsN5.java#L54

    If matches are not found, then just the interest points will be saved.
    """

    ips_by_setup = {int(v.setup): v for v in bs_model.view_interest_points.data}
    view_setups_by_id = {int(v.ident): v for v in bs_model.sequence_description.view_setups.view_setups}
    
    image_name = view_setups_by_id[image_id].name

    log = structlog.get_logger(image_name=image_name)
    start = time.time()
    log.info(f"Begin saving annotations for image id {image_id}")
    tile_coords: dict[int, Coords] = get_tile_coords(bs_model=bs_model)
    interest_points_url = parse_url(alignment_url) / "interestpoints.n5"
    dest_url_parsed = parse_url(dest_url)
    points_url = dest_url_parsed.joinpath(f"points/{image_name}.precomputed")
    lines_url = dest_url_parsed.joinpath(f"matches/{image_name}.precomputed")

    log.info(f"Saving points to {points_url}")
    log.info(f"Saving matches to {lines_url}")

    ip_store = zarr.N5FSStore(str(interest_points_url))
    base_coords = tile_coords[image_id]

    ip_path = bs_model.view_interest_points.data[image_id].path
    match_group = zarr.open_group(
        store=ip_store, path=f"{ip_path}/correspondences", mode="r"
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
        coords = tile_coords[img_id]
        _path = ips_by_setup[img_id].path
        points_data, match_data = load_points_tile(
            store=ip_store, path=_path, anon=True
        )
        points_transformed = translate_points_xyz(
            points_array=points_data["loc_xyz"].to_numpy(), coords=coords
        )
        points_data = points_data.with_columns(
            loc_xyz=pl.Series(points_transformed, dtype=pl.Array(pl.Float64, 3))
        )

        points_map[img_id] = points_data
        matches_map[img_id] = match_data

    annotation_scales = [base_coords[dim]["scale"] for dim in ("x", "y", "z")]  # type: ignore
    annotation_units = ["um", "um", "um", "s"]
    annotation_space = neuroglancer.CoordinateSpace(
        names=["x", "y", "z"], scales=annotation_scales, units=annotation_units
    )

    try:
        point_color = tile_coordinate_to_rgba(image_name_to_tile_coord(image_name))
    except IndexError:
        point_color = (125,125,125,125)

    line_starts: list[tuple[float, float, float, float]] = []
    line_stops: list[tuple[float, float, float, float]] = []
    point_map_self = points_map[image_id]

    point_data = points_map[image_id]
    id_data = point_map_self.get_column("id").to_list()
    write_point_annotations(
        points_url,
        points=point_data["loc_xyz"].to_numpy(),
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
            line_start = row_self["loc_xyz"]
            try:
                row_other = points_map[id_other].row(
                    by_predicate=(pl.col("id") == point_other), named=True
                )
                line_stop = row_other["loc_xyz"]

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


def save_interest_points(
    *,
    bs_model: SpimData2,
    alignment_url: URL,
    dest: URL,
    image_names: Iterable[str] | None = None,
):
    """
    Save interest points for all tiles as collection of neuroglancer precomputed annotations. One
    collection of annotations will be generated per image described in the bigstitcher metadata under
    the directory name <out_prefix>/<image_name>.precomputed
    """

    view_setup_by_ident: dict[int, ViewSetup] = {
        int(v.ident): v for v in bs_model.sequence_description.view_setups.view_setups
    }

    ips_by_setup = {int(v.setup): v for v in bs_model.view_interest_points.data}
    
    if image_names is None:
        image_names_parsed = [
            view_setup_by_ident[ip_id].name for ip_id in ips_by_setup
        ]
    else:
        image_names_parsed = image_names

    # generate a coordinate grid for all the images

    if bs_model.view_interest_points is None:
        raise ValueError(
            "No view interest points were found in the bigstitcher xml file."
        )

    for setup_id in ips_by_setup:
        save_annotations(
            bs_model=bs_model,
            image_id=setup_id,
            alignment_url=alignment_url,
            dest_url=dest,
        )


def fetch_summarize_matches(
    *, bigstitcher_xml: URL, pool: ThreadPoolExecutor, anon: bool = True
) -> pl.DataFrame:

    log = structlog.get_logger()
    bs_model = read_bigstitcher_xml(bigstitcher_xml)
    interest_points_url = bigstitcher_xml.parent.joinpath("interestpoints.n5")
    all_matches = fetch_all_matches(
        bs_model=bs_model,
        n5_interest_points_url=interest_points_url,
        pool=pool,
        anon=anon,
    )
    if len(all_matches) == 0:
        raise ValueError("No matches found!")
    valid_matches = {}
    for key, value in all_matches.items():
        if isinstance(value, BaseException):
            log.info(f"An exception occurred when accessing {key}")
            log.exception(value)
        else:
            valid_matches[key] = value

    summarized = summarize_matches(bs_model=bs_model, matches_dict=valid_matches)
    return summarized


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
    tile_coords = get_tile_coords(bs_model)

    image_names = tuple(
        v.name for v in bs_model.sequence_description.view_setups.view_setups
    )

    # polars dataframes keyed by bigstitcher image names
    individual_summaries = {}
    for v in matches_dict.values():
        individual_summaries[v["id_self"][0]] = summarize_match(v)

    # include the file name and the normalized tile coordinate to each column
    individual_augmented = {}
    for k, v in individual_summaries.items():
        tcoord = tile_coords[k]["x"], tile_coords[k]["y"], tile_coords[k]["z"]
        individual_augmented[k] = v.with_columns(
            image_name=pl.lit(image_names[k]),
            x_coord_self=pl.lit(tcoord[0]['trans']),
            y_coord_self=pl.lit(tcoord[1]['trans']),
            z_coord_self=pl.lit(tcoord[2]['trans']),
        )
    result = (
        pl.concat(individual_augmented.values())
        .select(
            [
                pl.col("image_name"),
                pl.col("id_self"),
                pl.col("id_other"),
                pl.col("num_matches"),
                pl.col("x_coord_self"),
                pl.col("y_coord_self"),
                pl.col("z_coord_self"),
            ]
        )
        .sort("id_self")
    )
    return result


def fetch_all_matches(
    *,
    bs_model: SpimData2,
    n5_interest_points_url: URL,
    pool: ThreadPoolExecutor,
    anon: bool = True,
) -> dict[str, pl.DataFrame | BaseException | None]:
    """
    Load all the match data from the n5 datasets containing it.
    Takes a URL to an n5 group emitted by bigstitcher for storing interest points, e.g.
    s3://bucket/dataset/interestpoints.n5/.

    This function uses a thread pool to speed things up.
    """
    vips = bs_model.view_interest_points
    matches_dict: dict[str, pl.DataFrame | BaseException | None] = {}
    futures_dict = {}

    for ip in vips.data:
        fut = pool.submit(
            load_points_tile,
            store=n5_interest_points_url,
            path=ip.path,
            anon=anon,
        )
        futures_dict[fut] = ip.path

    for result in as_completed(futures_dict):
        key = futures_dict[result]
        try:
            _, matches = result.result()
            matches_dict[key] = matches
        except BaseException:
            matches_dict[key] = result.exception()

    return matches_dict

def hoaffine_to_array(
        tx: pydantic_bigstitcher.transform.HoAffine[Any],
        dimensions: tuple[str, ...]) -> np.ndarray:
    out = np.eye((len(dimensions) + 1))
    affine = affine_to_array(tx.affine, dimensions)
    translation = translate_to_array(tx.translation, dimensions)
    out[:-1, :-1] = affine
    out[:-1,-1] = translation
    return out

def affine_to_array(
        tx: pydantic_bigstitcher.transform.MatrixMap[Any],
        dimensions: tuple[str, ...]) -> np.ndarray:
    assert set(dimensions) == set(tx.keys())
    out = np.zeros((
        len(tx), ) * 2)
    
    for idx, (dim_outer, dim_inner) in enumerate(product(dimensions, dimensions)):
        out.ravel()[idx] = tx[dim_outer][dim_inner]
    
    return out

def array_to_affine(array: np.ndarray, dimensions: tuple[str, ...]) -> pydantic_bigstitcher.transform.MatrixMap[Any]:
    return dict(zip(dimensions, (array_to_translate(row, dimensions) for row in array)))

def translate_to_array(
        tx: pydantic_bigstitcher.transform.VectorMap[Any],
        dimensions: tuple[str, ...]) -> np.ndarray:
    assert set(dimensions) == set(tx.keys())
    out = np.zeros(len(tx))
    for idx, dim in enumerate(dimensions):
        out[idx] = tx[dim]
    return out
        
def array_to_translate(array: np.ndarray, dimensions: tuple[str, ...]) -> pydantic_bigstitcher.transform.VectorMap[Any]:
    return dict(zip(dimensions, array))        

def compose_hoaffines(
        tx_a: pydantic_bigstitcher.transform.HoAffine[Any], 
        tx_b: pydantic_bigstitcher.transform.HoAffine[Any],
        *,
        dimensions: tuple[str, ...]
        ) -> pydantic_bigstitcher.transform.HoAffine[Any]:

        homogeneous_matrix = hoaffine_to_array(tx_b, dimensions=dimensions) @ hoaffine_to_array(tx_a, dimensions=dimensions)

        new_affine = homogeneous_matrix[:-1, :-1]
        new_trans = homogeneous_matrix[:-1,-1]

        return pydantic_bigstitcher.transform.HoAffine(
            affine=array_to_affine(
                new_affine, dimensions=dimensions), 
                translation=array_to_translate(new_trans, dimensions=dimensions))
            