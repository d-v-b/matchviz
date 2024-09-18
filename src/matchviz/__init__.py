# SPDX-FileCopyrightText: 2024-present Davis Vann Bennett <davis.v.bennett@gmail.com>
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
from yarl import URL
import zarr
import polars as pl
from neuroglancer import ImageLayer, AnnotationLayer, ViewerState, CoordinateSpace
from matchviz.bigstitcher import image_name_to_tile_coord
from matchviz.core import get_url, parse_url
from matchviz.core import get_histogram_bounds_batch
from matchviz.core import tile_coordinate_to_rgba
from matchviz.neuroglancer_styles import NeuroglancerViewerStyle
from functools import reduce
from concurrent.futures import ThreadPoolExecutor
from matplotlib import colors
import matplotlib.pyplot as plt
from xarray_ome_ngff import read_multiscale_group
import structlog

pool = ThreadPoolExecutor(max_workers=10)


# we assume here that there's no need to parametrize t
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


def create_neuroglancer_state(
    image_url: str | URL,
    points_url: str | URL | None,
    matches_url: str | URL | None,
    style: NeuroglancerViewerStyle,
):
    if points_url is not None:
        points_url_parsed = parse_url(points_url)
    else:
        points_url_parsed = None

    if matches_url is not None:
        matches_url_parsed = parse_url(matches_url)
    else:
        matches_url_parsed = None

    image_url_parsed = parse_url(image_url)
    log = structlog.get_logger()

    image_group = zarr.open_group(store=str(image_url_parsed), path="", mode="r")
    image_sources = {}
    points_sources = {}
    matches_sources = {}
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

    state = ViewerState(
        dimensions=space, cross_section_scale=1000, projection_scale=500_000
    )

    # read the smallest images in the pyramid
    subgroups = tuple(image_group.groups())
    subgroup_urls = tuple(str(get_url(g)) for _, g in subgroups)
    bounds = get_histogram_bounds_batch(subgroup_urls, pool)
    histogram_bounds = {k: v for k, v in zip(image_group.group_keys(), bounds)}

    for fname, sub_group in subgroups:
        subgroup_url = get_url(sub_group)
        image_sources[fname] = f"zarr://{subgroup_url}"

        annotation_dir = fname.removesuffix(".zarr") + ".precomputed"

        if points_url is not None:
            point_url = points_url_parsed.joinpath(annotation_dir)
            points_sources[fname] = f"precomputed://{point_url}"

        if matches_url is not None:
            match_url = matches_url_parsed.joinpath(annotation_dir)
            matches_sources[fname] = f"precomputed://{match_url}"

    # bias the histogram towards the brighter values
    hist_min, hist_max = reduce(
        lambda old, new: (max(old[0], new[0]), max(old[1], new[1])),
        histogram_bounds.values(),
    )
    log.info(f"Using histogram bounded between ({hist_min}, {hist_max})")
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
            if points_url is not None:
                point_source = points_sources[fname]
                state.layers.append(
                    name=f"{name_base}/points",
                    layer=AnnotationLayer(
                        source=point_source, shader=annotation_shader
                    ),
                )
            if matches_url is not None:
                match_source = matches_sources[fname]
                state.layers.append(
                    name=f"{name_base}/matches",
                    layer=AnnotationLayer(
                        source=match_source, shader=annotation_shader
                    ),
                )

    elif style == "images_combined":
        state.layers.append(
            name="images",
            layer=ImageLayer(
                source=list(image_sources.values()),
                shader_controls=image_shader_controls,
            ),
        )
        if points_url is not None:
            state.layers.append(
                name="points",
                layer=AnnotationLayer(
                    source=list(points_sources.values()), shader=annotation_shader
                ),
            )
        if matches_url is not None:
            state.layers.append(
                name="matches",
                layer=AnnotationLayer(
                    source=list(matches_sources.values()), shader=annotation_shader
                ),
            )
    else:
        msg = f"Style {style} not recognized. Style must be one of 'images_combined' or 'images_split'"
        raise ValueError(msg)
    return state
