from __future__ import annotations
from itertools import accumulate
import pydantic_bigstitcher.transform
import xarray
from zarr import N5FSStore
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
import re
import time
from typing import Annotated, Any, Literal, cast, TYPE_CHECKING, Iterable
from typing_extensions import TypedDict, deprecated
from pydantic import BaseModel, BeforeValidator, Field
from pydantic_zarr.v2 import ArraySpec, GroupSpec
from matchviz.annotation import write_line_annotations, write_point_annotations
import random
import colorsys
from matchviz.core import (
    get_store_url,
    get_url,
    parse_url,
    tile_coordinate_to_rgba,
)
from matchviz.schemas import IP_SCHEMA, MATCH_SCHEMA
from matchviz.transform import (
    apply_hoaffine,
    array_to_hoaffine,
    compose_hoaffines,
    hoaffine_to_array,
)
from matchviz.types import TileCoordinate
from pydantic_bigstitcher.transform import HoAffine
from pydantic_bigstitcher import ZarrImageLoader

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
from neuroglancer.static_file_server import StaticFileServer
from pydantic_bigstitcher.transform import Axes as T_XYZ
from xarray_ome_ngff.v04.multiscale import read_multiscale_group

xyz = ("x", "y", "z")
random.seed(0)


@dataclass(frozen=True, slots=True, kw_only=True)
class SetupTimepoint:
    setup: str
    timepoint: int


def read_bigstitcher_xml(url: URL) -> SpimData2:
    fs, path = fsspec.url_to_fs(str(url))
    bs_xml = fs.cat_file(path)
    bs_model = SpimData2.from_xml(bs_xml)
    return bs_model


def spimdata_to_neuroglancer(
    xml_path: URL,
    *,
    host: URL | None = None,
    transform_index: int = -1,
    view_setups: Iterable[str] | None = None,
    channels: Iterable[int] | None = None,
    interest_points: Literal["points", "matches"] | None = None,
    display_settings: dict[str, Any] | None = None,
    bind_address="127.0.0.1",
) -> neuroglancer.ViewerState:
    bs_model = read_bigstitcher_xml(xml_path)
    sample_vs = bs_model.sequence_description.view_setups.elements[0]
    unit = sample_vs.voxel_size.unit
    scales = base_scale_dict = {
        dim: float(x) for x, dim in zip(sample_vs.voxel_size.size.split(" "), xyz)
    }
    dimension_names_out = xyz[::-1]
    units = [unit] * len(dimension_names_out)
    points_map = {}
    output_space = neuroglancer.CoordinateSpace(
        names=dimension_names_out,
        scales=[scales[k] for k in dimension_names_out],
        units=units,
    )

    annotation_shader = """void main() {
  setColor(prop_color());
  setPointMarkerSize(prop_size());
}"""

    state = neuroglancer.ViewerState(dimensions=output_space)

    if bs_model.base_path.path == ".":
        if host is None:
            server = StaticFileServer(
                str(xml_path.parent.path), bind_address=bind_address, daemon=True
            )
            base_url = URL(server.url)
        else:
            server = None
            base_url = URL(host)

    else:
        raise ValueError(f"Base path {bs_model.base_path.path} is not supported")

    image_loader = bs_model.sequence_description.image_loader
    image_format: Literal["n5", "zarr"]
    image_format = image_loader.fmt.split(".")[-1]
    if getattr(image_loader, image_format).type == "absolute":
        if hasattr(image_loader, "s3bucket"):
            prefix = URL(f"s3://{image_loader.s3bucket}")
        else:
            prefix = base_url
        container_root = prefix / getattr(image_loader, image_format).path
    else:
        container_root = base_url / getattr(image_loader, image_format).path

    for view_reg in bs_model.view_registrations.elements:
        vs_id = view_reg.setup

        if view_setups is not None and vs_id not in view_setups:
            continue

        maybe_view_setups = tuple(
            filter(
                lambda v: v.ident == vs_id,
                bs_model.sequence_description.view_setups.elements,
            )
        )

        if len(maybe_view_setups) > 1:
            raise ValueError(f"Ambiguous setup id {vs_id}.")

        view_setup = maybe_view_setups[0]

        if channels is not None and int(view_setup.attributes.channel) not in channels:
            continue

        if image_format == "zarr":
            image_name = view_setup.name
            image_path = container_root / f"{image_name}.zarr"
        else:
            image_name = f"setup{view_reg.setup}/timepoint{view_reg.timepoint}"
            image_path = container_root / image_name

        # bigstitcher stores the transformations in reverse order, i.e. the
        # last transform in the series is the first one in the list

        tforms = get_transforms(bs_model)[view_setup.ident]
        # for zarr data, replace the translation to nominal grid transform because neuroglancer
        # infers the position from zarr metadata

        extra_points_transform = array_to_hoaffine(np.eye(4), dimensions=xyz)

        if image_format == "zarr":
            tforms_filtered = tuple(
                filter(lambda v: v.name != "Translation to Nominal Grid", tforms)
            )

            extra_points_transform = [
                t for t in tforms if t.name == "Translation to Nominal Grid"
            ][0].transform
        else:
            tforms_filtered = tforms

        tforms_indexed = [
            *tforms_filtered[:transform_index],
            tforms_filtered[transform_index],
        ]

        final_transform = compose_transforms(tforms_indexed)

        # neuroglancer expects ndim x ndim + 1 matrix
        matrix = hoaffine_to_array(final_transform, dimensions=dimension_names_out)[:-1]

        # convert translations into the output coordinate space
        base_scale_dict = {
            dim: float(x)
            for x, dim in zip(view_setup.voxel_size.size.split(" "), ("x", "y", "z"))
        }
        input_scales = [base_scale_dict[dim] for dim in dimension_names_out]
        image_name = vs_id

        image_source = f"{image_format}://{image_path}"
        hue, sat, lum = (
            random.random(),
            0.5 + random.random() / 2.0,
            0.4 + random.random() / 5.0,
        )
        color = [int(256 * i) for i in colorsys.hls_to_rgb(hue, lum, sat)]
        color_str = "#{0:02x}{1:02x}{2:02x}".format(*color)
        shader = (
            "#uicontrol invlerp normalized()\n"
            f'#uicontrol vec3 color color(default="{color_str}")\n'
            "void main(){{emitRGB(color * normalized());}}"
        )
        if display_settings is not None:
            image_shader_controls = {
                "normalized": {
                    "range": [display_settings["start"], display_settings["stop"]],
                    "window": [display_settings["min"], display_settings["max"]],
                }
            }
        else:
            image_shader_controls = None
        input_space = neuroglancer.CoordinateSpace(
            names=dimension_names_out, units=units, scales=input_scales
        )

        image_transform = neuroglancer.CoordinateSpaceTransform(
            output_dimensions=input_space,
            input_dimensions=input_space,
            source_rank=len(dimension_names_out),
            matrix=matrix,
        )
        image_layer = neuroglancer.ImageLayer(
            source=neuroglancer.LayerDataSource(
                url=image_source, transform=image_transform
            ),
            shader_controls=image_shader_controls,
            shader=shader,
            blend="additive",
        )
        state.layers.append(name=image_name, layer=image_layer)
        # TODO: choose a source class based on whether a host was provided, so that
        # local files can potentially be viewed.

        annotation_layer = []

        if interest_points == "points":
            points_data = read_interest_points(
                bs_model=bs_model,
                image_id=vs_id,
                store=xml_path.parent / "interestpoints.n5",
            )

            points_affine = compose_hoaffines(extra_points_transform, final_transform)

            points_transform = neuroglancer.CoordinateSpaceTransform(
                output_dimensions=input_space,
                input_dimensions=input_space,
                source_rank=len(dimension_names_out),
                matrix=hoaffine_to_array(points_affine, dimensions=dimension_names_out)[
                    :-1
                ],
            )

            # points_data = points_data.with_columns(loc_xyz_transformed=locs_transformed)
            points_map[vs_id] = points_data
            annotation_layer = neuroglancer.LocalAnnotationLayer(
                transform=points_transform,
                dimensions=output_space,
                annotation_properties=[
                    neuroglancer.AnnotationPropertySpec(
                        id="color",
                        type="rgb",
                        default="green",
                    ),
                    neuroglancer.AnnotationPropertySpec(
                        id="size",
                        type="float32",
                        default=10,
                    ),
                ],
                annotations=[
                    neuroglancer.PointAnnotation(
                        id=idx, point=point, props=[color_str, 10]
                    )
                    for idx, point in enumerate(
                        points_data["point_loc_xyz"].to_numpy()[:, ::-1]
                    )
                ],
                shader=annotation_shader,
            )

            state.layers.append(name="interest_points", layer=annotation_layer)

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


def parse_idmap(data: dict[str, str]) -> dict[tuple[str, str, str], str]:
    """
    convert {'0,1,beads': 0} to {('0', '1', "beads"): '0'}
    """
    parts = map(lambda k: k.split(","), data.keys())
    # convert first two elements to int, leave the last as str
    parts_normalized = map(lambda v: (v[0], v[1], v[2]), parts)
    return dict(zip(parts_normalized, data.values()))


def parse_matches(
    *, name: str, data: np.ndarray, id_map: dict[tuple[str, str, str], str]
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
    remap = {int(value): int(key[1]) for key, value in id_map.items()}

    # replace the pair id value with an actual image index reference in the last column
    data_copy[:, -1] = np.vectorize(remap.get)(data[:, -1])

    match_result = pl.DataFrame(
        schema=MATCH_SCHEMA,
        data={
            "point_id_self": data_copy[:, 0],
            "point_id_other": data_copy[:, 1],
            "image_id_self": [id_self] * data_copy.shape[0],
            "image_id_other": data_copy[:, 2],
        },
        strict=False,
    )
    return match_result


def parse_interest_point_params(params: str) -> dict[str, str]:
    """
    Convert a semi-structured string in the ViewInterestPoints element of bigstitcher xml documents
    to a key: value pair.
    A string like 'DOG (Spark) s=2.0 t=0.005' will be mapped to `{'s': '2.0', 't': '0.005'}`
    """
    split_ws = params.split(" ")
    result: dict[str, str] = {}

    for x in split_ws:
        split_param = x.split("=")
        if len(split_param) == 2:
            result[split_param[0]] = split_param[1]

    return result


def read_interest_points(
    *,
    bs_model: SpimData2,
    image_id: str,
    store: str | URL | N5FSStore,
):
    """
    Load interest points and optionally correspondences from a bigstitcher-formatted n5 group as
    polars dataframes for a single image. In the event that bigstitcher performed downsampling before
    interest point detection, the interest points coordinates will be returned in the coordinate
    system of the full-resolution image.
    """

    log = structlog.get_logger()
    ips_by_setup = {v.setup: v for v in bs_model.view_interest_points.elements}
    path = ips_by_setup[image_id].path
    # these are the parameters that describe the interest point detection method
    params = parse_interest_point_params(ips_by_setup[image_id].params)
    ds_xy = float(params.get("downsampleXY", 1))
    ds_z = float(params.get("downsampleZ", 1))

    downsampling_factors = {"z": ds_z, "y": ds_xy, "x": ds_xy}

    if isinstance(store, (str, URL)):
        store_parsed = zarr.N5FSStore(str(store), mode="r")
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

    # points are saved as [num_points, [x, y, z]]
    arrays = dict(interest_points_group.arrays())
    loc = arrays["loc"][:]
    ids = arrays["id"][:]
    ids_list = ids.squeeze().tolist()

    # correct for downsampling
    ds_shift_xyz = np.array(
        [
            (downsampling_factors["x"] - 1) / 2,
            (downsampling_factors["y"] - 1) / 2,
            (downsampling_factors["z"] - 1) / 2,
        ]
    )

    loc_ds_corrected = loc - ds_shift_xyz

    try:
        intensities = arrays["intensities"][:].squeeze().tolist()
    except KeyError:
        intensities = [None] * len(ids_list)

    points_result = pl.DataFrame(
        schema=IP_SCHEMA,
        data={
            "image_id_self": [image_id] * len(ids_list),
            "point_id_self": ids_list,
            "point_loc_xyz": loc_ds_corrected,
            "point_intensity": intensities,
            "image_id_other": [None] * len(ids_list),
            "point_id_other": [None] * len(ids_list),
        },
    )

    matches_exist = "data" in correspondences_group

    if not matches_exist:
        log.info(
            f"No matches found in {get_store_url(store_parsed)} for image {image_id}."
        )
        result = points_result
    else:
        id_map = parse_idmap(correspondences_group.attrs["idMap"])
        matches = np.array(correspondences_group["data"])
        match_result = parse_matches(name=path, data=matches, id_map=id_map)
        result = points_result.drop("image_id_other", "point_id_other").join(
            match_result.drop("image_id_self"),
            on="point_id_self",
            how="left",
            coalesce=True,
        )
    return result


def read_all_interest_points(
    *,
    bs_model: SpimData2,
    store: URL | N5FSStore | str,
    pool: ThreadPoolExecutor,
) -> dict[ViewSetup, pl.DataFrame]:
    """
    Load all the match data from the n5 datasets containing it.
    Takes a URL to an n5 group emitted by bigstitcher for storing interest points, e.g.
    s3://bucket/dataset/interestpoints.n5/.

    This function uses a thread pool to speed things up.
    """
    vips = bs_model.view_interest_points
    result_dict: dict[ViewSetup, pl.DataFrame] = {}
    futures = []
    for ip in vips.elements:
        futures.append(
            pool.submit(
                read_interest_points,
                bs_model=bs_model,
                image_id=ip.setup,
                store=store,
            )
        )
    wait(futures)
    for idx, ip in enumerate(vips.elements):
        result_dict[ip.setup] = futures[idx].result()

    return result_dict


def get_transforms(
    bs_model: SpimData2,
) -> dict[str, tuple[pydantic_bigstitcher.transform.Transform[T_XYZ], ...]]:
    """
    Get transform of view setups referenced in bigstitcher xml data. Note that in bigstitcher metadata,
    the transforms are ordered from last-applied to first-applied, but this function returns them in the reverse order,
    i.e. the first transform is the first one that will be applied in the transform sequence.
    """
    result_tforms: dict[
        str, tuple[pydantic_bigstitcher.transform.Transform[T_XYZ], ...]
    ] = {}
    for vs, vr in zip(
        bs_model.sequence_description.view_setups.elements,
        bs_model.view_registrations.elements,
    ):
        key = vs.ident
        tforms = tuple(t.to_transform() for t in reversed(vr.view_transforms))
        result_tforms[key] = tforms

    return result_tforms


def compose_transforms(
    transforms: Iterable[pydantic_bigstitcher.transform.Transform[T_XYZ]],
):
    """
    Compose transforms from bigstitcher.
    """
    hoaffines = tuple(t.transform for t in transforms)
    if not all(isinstance(t, HoAffine) for t in hoaffines):
        raise ValueError("Expected all transforms to be of type HoAffine")
    return tuple(accumulate(hoaffines, compose_hoaffines))[-1]


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
    *, bs_model: SpimData2, image_id: str, alignment_url: URL | str, dest_url: URL | str
):
    """
    Load points and correspondences (matches) between a single tile an all other tiles, and save as neuroglancer
    precomputed annotations.

        e.g. dataset = 'exaSPIM_3163606_2023-11-17_12-54-51'
        alignment_id = 'alignment_2024-01-09_05-00-44'

    N5 is organized according to the structure defined here: https://github.com/PreibischLab/multiview-reconstruction/blob/a566bf4d6d35a7ab00d976a8bf46f1615b34b2d0/src/main/java/net/preibisch/mvrecon/fiji/spimdata/interestpoints/InterestPointsN5.java#L54

    If matches are not found, then just the interest points will be saved.
    """
    unit = bs_model.sequence_description.view_setups.elements[0].voxel_size.unit
    base_scales = tuple(
        map(
            float,
            bs_model.sequence_description.view_setups.elements[0].voxel_size.size.split(
                " "
            ),
        )
    )
    dimension_names = ["x", "y", "z"]
    base_units = [unit] * len(dimension_names)
    view_setups_by_id = {
        v.ident: v for v in bs_model.sequence_description.view_setups.elements
    }

    image_name = view_setups_by_id[image_id].name

    log = structlog.get_logger(image_name=image_name)
    start = time.time()
    log.info(f"Begin saving annotations for image id {image_id}")
    interest_points_url = parse_url(alignment_url) / "interestpoints.n5"
    dest_url_parsed = parse_url(dest_url)
    points_url = dest_url_parsed.joinpath(f"points/{image_name}.precomputed")
    lines_url = dest_url_parsed.joinpath(f"matches/{image_name}.precomputed")

    log.info(f"Saving points to {points_url}")
    log.info(f"Saving matches to {lines_url}")

    ip_store = zarr.N5FSStore(str(interest_points_url))

    ip_path = bs_model.view_interest_points.elements[int(image_id)].path
    match_group = zarr.open_group(
        store=ip_store, path=f"{ip_path}/correspondences", mode="r"
    )

    to_access: tuple[str, ...] = (image_id,)
    id_map_normalized = {}
    # tuple of view_setup ids to load
    points_map: dict[str, pl.DataFrame] = {}

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
        points_data = read_interest_points(
            bs_model=bs_model,
            image_id=img_id,
            store=ip_store,
        )

        transforms = get_transforms(bs_model=bs_model)[img_id]
        # not clear which transformation to use here, lets take the next-to-last
        affine_composed = compose_transforms(transforms[:-1])
        locs_transformed = apply_hoaffine(
            tx=affine_composed,
            data=points_data["point_loc_xyz"].to_numpy(),
            dimensions=dimension_names,
        )

        points_data = points_data.with_columns(loc_xyz_transformed=locs_transformed)
        points_map[img_id] = points_data

    annotation_space = neuroglancer.CoordinateSpace(
        names=dimension_names, scales=base_scales, units=base_units
    )

    try:
        point_color = tile_coordinate_to_rgba(image_name_to_tile_coord(image_name))
    except IndexError:
        point_color = (125, 125, 125, 125)

    line_starts: list[tuple[float, float, float, float]] = []
    line_stops: list[tuple[float, float, float, float]] = []
    point_map_self = points_map[image_id]

    point_data = points_map[image_id]
    id_data = point_map_self.get_column("point_id_self").to_list()

    write_point_annotations(
        points_url,
        points=point_data["loc_xyz_transformed"].to_numpy(),
        ids=id_data,
        coordinate_space=annotation_space,
        point_color=point_color,
    )

    matches = point_data.filter(pl.col("point_id_other").is_not_null())

    if len(matches) > 0:
        log.info(f"Saving matches to {lines_url}.")
        for image_id_other_tup, match_group in matches.group_by("image_id_other"):
            image_id_other = image_id_other_tup[0]
            joined = points_map[image_id_other].join(
                match_group,
                left_on="point_id_self",
                right_on="point_id_other",
                how="inner",
            )

            line_starts.extend(joined["loc_xyz_transformed"].to_numpy().tolist())
            line_stops.extend(joined["loc_xyz_transformed_right"].to_numpy().tolist())

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

    view_setup_by_ident: dict[str, ViewSetup] = {
        v.ident: v for v in bs_model.sequence_description.view_setups.elements
    }

    ips_by_setup = {v.setup: v for v in bs_model.view_interest_points.elements}

    if image_names is None:
        image_names_parsed = [
            view_setup_by_ident[ip_id].ident for ip_id in ips_by_setup
        ]
    else:
        image_names_parsed = list(image_names)

    if bs_model.view_interest_points is None:
        raise ValueError(
            "No view interest points were found in the bigstitcher xml file."
        )

    for setup_id in image_names_parsed:
        save_annotations(
            bs_model=bs_model,
            image_id=setup_id,
            alignment_url=alignment_url,
            dest_url=dest,
        )


def fetch_summarize_matches(
    *,
    bigstitcher_xml: URL,
    pool: ThreadPoolExecutor,
) -> pl.DataFrame:
    _ = structlog.get_logger()
    bs_model = read_bigstitcher_xml(bigstitcher_xml)
    interest_points_url = bigstitcher_xml.parent.joinpath("interestpoints.n5")
    all_matches = read_all_interest_points(
        bs_model=bs_model,
        store=interest_points_url,
        pool=pool,
    )
    if len(all_matches) == 0:
        raise ValueError("No matches found!")

    summarized = summarize_matches(bs_model=bs_model, matches_dict=all_matches)
    return summarized


def summarize_matches(
    bs_model: SpimData2,
    matches_dict: dict[str, pl.DataFrame],
) -> pl.DataFrame:
    transforms = get_transforms(bs_model=bs_model)
    view_setups = {
        v.ident: v for v in bs_model.sequence_description.view_setups.elements
    }

    matches_tx = {}
    origins = {}
    for k, df in matches_dict.items():
        # the logic here is to assume that every transformation up the the final one is the baseline
        # and the final transform is the one estimated from the detected interest points. This assumption
        # may not hold in general.
        txs = transforms[k]
        point_loc_xyz = df["point_loc_xyz"]

        if len(txs) > 1:
            pre_aff, post_aff = compose_transforms(txs[:-1]), compose_transforms(txs)
        else:
            pre_aff = txs[0].transform
            post_aff = None

        # get the origin of the image after applying the transform
        # center of the image is probably better
        origins[k] = apply_hoaffine(
            tx=pre_aff, data=np.array([[0, 0, 0]]), dimensions=xyz
        )[0]

        point_loc_xyz_baseline = apply_hoaffine(
            tx=pre_aff, data=point_loc_xyz.to_numpy(), dimensions=(xyz)
        )
        if post_aff is not None:
            point_loc_xyz_fitted = apply_hoaffine(
                tx=post_aff, data=point_loc_xyz.to_numpy(), dimensions=(xyz)
            )
            error = np.linalg.norm(
                point_loc_xyz_fitted - point_loc_xyz_baseline, axis=1
            )
        else:
            point_loc_xyz_fitted = [None] * len(point_loc_xyz)
            error = [None] * len(point_loc_xyz)

        matches_tx[k] = df.with_columns(
            point_loc_xyz_baseline=point_loc_xyz_baseline,
            point_loc_xyz_fitted=point_loc_xyz_fitted,
            error=error,
        )

    # concatenate all matches into a single dataframe
    # the only possible nulls should be points that were not matched, so we drop all of those to just
    # get the set of matched points
    # specify diagonal concat because frames with no matches might have a different column order
    all_matches = pl.concat(matches_tx.values(), how="diagonal").filter(
        ~pl.col("image_id_other").is_null()
    )

    out = (
        all_matches.group_by("image_id_self", "image_id_other")
        .agg(
            [
                pl.col("point_id_self").count().name.map(lambda v: "num_matches"),
                pl.col("error").min().name.suffix("_min"),
                pl.col("error").max().name.suffix("_max"),
                pl.col("error").mean().name.suffix("_mean"),
            ]
        )
        .sort("image_id_self", "image_id_other")
    )

    out = out.with_columns(
        image_name_self=pl.Series([view_setups[k].name for k in out["image_id_self"]]),
        image_name_other=pl.Series(
            [view_setups[k].name for k in out["image_id_other"]]
        ),
        image_origin_self=pl.Series(
            [origins[k] for k in out["image_id_self"]],
            dtype=pl.Array(pl.Float64, 3),
        ),
    )

    return out


def get_image_group(bigstitcher_xml: URL, image_id: str) -> dict[str, xarray.DataArray]:
    bs_model = read_bigstitcher_xml(bigstitcher_xml)
    image_loader = bs_model.sequence_description.image_loader
    if isinstance(image_loader, ZarrImageLoader):
        # not sure bigstitcher support local zarr arrays yet
        scheme = "s3://"
        bucket = image_loader.s3bucket
        path = image_loader.zarr.path
        zgroups_by_setup = {s.setup: s for s in image_loader.zgroups.elements}

        if image_id not in zgroups_by_setup:
            raise ValueError(f"image {image_id} not found in zarr groups")

        group_name = zgroups_by_setup[image_id].path
        zgroup = zarr.open_group(f"{scheme}{bucket}/{path}/{group_name}", mode="r")
        msg = read_multiscale_group(
            zgroup, array_wrapper={"name": "dask_array", "config": {"chunks": "auto"}}
        )
        return msg
    else:
        raise NotImplementedError
