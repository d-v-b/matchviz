from __future__ import annotations
import numpy as np
import polars as pl
from pydantic_ome_ngff.v04.multiscale import MultiscaleMetadata
import zarr
from yarl import URL
import pathlib

import os
from typing import Iterable, Literal, Sequence

from matchviz.types import Coords, TileCoordinate


def get_url(node: zarr.Group | zarr.Array) -> URL:
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
        return URL(f"{protocol}://{os.path.join(store_path, node.path)}")
    else:
        msg = (
            f"The store associated with this object has type {type(store)}, which "
            "cannot be resolved to a url"
        )
        raise ValueError(msg)


def ome_ngff_to_coords(url: str | URL) -> Coords:
    multi_meta = MultiscaleMetadata(
        **zarr.open_group(str(url), mode="r").attrs.asdict()["multiscales"][0]
    )
    scale = multi_meta.datasets[0].coordinateTransformations[0].scale
    trans = multi_meta.datasets[0].coordinateTransformations[1].translation
    return {
        axis.name: {"scale": s, "trans": t}
        for axis, s, t in zip(multi_meta.axes, scale, trans)
    }  # type: ignore


def scale_point(point: Sequence[float], params: Sequence[float]):
    return np.multiply(point, params)


def scale_points(points_df: pl.DataFrame, coords: Coords):
    col = "loc_xyz"
    dims = ("x", "y", "z")
    col_index = points_df.columns.index(col)

    scale = [coords[dim]["scale"] for dim in dims]  # type: ignore
    new_col = scale_point(points_df[col].to_list(), scale)

    return points_df.clone().replace_column(
        col_index, pl.Series(name=col, values=new_col.tolist())
    )


def translate_point(point: Sequence[float], params: Sequence[float]):
    return np.add(point, params)


def translate_points(points_df: pl.DataFrame, coords: Coords):
    """
    Apply a translation
    """
    col = "loc_xyz"
    dims: tuple[Literal["x", "y", "z"], ...] = ("x", "y", "z")
    col_index = points_df.columns.index(col)
    local_scale = [coords[dim]["scale"] for dim in dims]
    local_trans = [coords[dim]["trans"] for dim in dims]

    trans = np.divide(local_trans, local_scale)  # type: ignore
    new_col = translate_point(points_df[col].to_list(), trans)

    return points_df.clone().replace_column(
        col_index, pl.Series(name=col, values=new_col.tolist())
    )


def transform_points(coords: Coords, points: np.ndarray):
    translate_points(coords, points)
    scale_points(coords, points)


def get_percentiles(image_url: str) -> tuple[int, int]:
    """
    Get the 5th and 95th percentiles from of the smallest array in a multiscale group
    """
    group = zarr.open_group(image_url, mode="r")
    arrays_sorted = sorted(group.arrays(), key=lambda kv: np.prod(kv[1].shape))
    smallest = arrays_sorted[0][1][:]
    return np.percentile(smallest, (5, 95))


def get_histogram_bounds_batch(group_urls: tuple[str, ...], pool):
    return tuple(pool.map(get_percentiles, group_urls))


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


def tokenize(data: Sequence[float]) -> Sequence[int]:
    uniques = sorted(np.unique(data).tolist())
    return [uniques.index(d) for d in data]

def tokenize_tile_coords(tile_coords: dict[int, Coords]) -> tuple[tuple[int, int], ...]:
    """
    Convert positions in world coordinates to positions in a space where the only coordinates
    allowed are tuples of integers.
    """

    tile_positions = {
        k: (v["x"], v["y"], v["z"]) for k, v in tile_coords.items()
    }
    tile_x, tile_y, tile_z = zip(*tuple(t for t in tile_positions.values()))
    tile_coords_normed = zip(*map(tokenize, (tile_x, tile_y, tile_z)))
    return tile_coords_normed

def tokenize_translations(translations: Iterable[tuple[float, ...]]) -> Iterable[tuple[float, ...]]:
    dims = np.array(tuple(translations))
    return zip(tokenize(dim) for dim in dims.T)    

def parse_url(data: object) -> URL:
    """
    Parse input as a URL
    """
    if isinstance(data, URL):
        return data
    if isinstance(data, str):
        maybe_url = URL(data.rstrip("/"))
        if maybe_url.scheme == "":
            if maybe_url.is_absolute():
                return maybe_url.with_scheme("file")
            else:
                return URL.build(scheme="file", path=maybe_url.path.rstrip("/"))
        else:
            return maybe_url
    if isinstance(data, pathlib.Path):
        return URL.build(scheme="file", path=str(data))
    else:
        raise TypeError(
            f"Invalid input. Expected a str, URL, or pathlib.path. Got {type(data)}"
        )
