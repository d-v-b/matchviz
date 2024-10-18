from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import socket
from typing import Iterable, Literal, Sequence
import click
import fsspec
import fsspec.implementations
import fsspec.implementations.local
from yarl import URL
from matchviz import (
    create_neuroglancer_state,
)
from matchviz.bigstitcher import (
    bdv_to_neuroglancer,
    fetch_summarize_matches,
    get_tilegroup_url,
    read_bigstitcher_xml,
    save_interest_points,
)
from matchviz.core import parse_url
from matchviz.neuroglancer_styles import (
    NeuroglancerViewerStyle,
    neuroglancer_view_styles,
)
import structlog
from s3fs import S3FileSystem
import neuroglancer
from matchviz.plot import plot_matches
from pydantic_bigstitcher import SpimData2


@click.group("matchviz")
def cli(): ...


log_level = click.option("--log-level", type=click.STRING, default="info")


@cli.command("plot-matches")
@click.option(
    "--bigstitcher-xml",
    type=click.STRING,
    help="URL pointing to a bigsticher xml document",
    required=True,
)
@click.option(
    "--dest",
    help="Name of a file in which to save the plot.",
    required=True,
    type=click.STRING,
)
@click.option("--invert-y-axis", type=click.BOOL, is_flag=True, default=False)
@click.option("--invert-x-axis", type=click.BOOL, is_flag=True, default=False)
def plot_matches_cli(
    bigstitcher_xml: str,
    dest: str,
    invert_y_axis: bool,
    invert_x_axis: bool,
):
    pool = ThreadPoolExecutor(max_workers=16)
    anon = True

    bigstitcher_xml_normalized = parse_url(bigstitcher_xml)
    data = fetch_summarize_matches(
        bigstitcher_xml=bigstitcher_xml_normalized, pool=pool, anon=anon
    )

    fig = plot_matches(
        data=data,
        dataset_name=bigstitcher_xml_normalized.path,
        invert_x=invert_x_axis,
        invert_y=invert_y_axis,
    )
    fig.savefig(dest)


@cli.command("save-points")
@click.option("--bigstitcher-xml", type=click.STRING, required=True)
@click.option("--dest", type=click.STRING, required=True)
@click.option("--image-names", type=click.STRING)
def save_interest_points_cli(bigstitcher_xml: str, dest: str, image_names: str | None):
    """
    Save bigstitcher interest points from n5 to neuroglancer precomputed annotations.
    """
    # strip trailing '/' from src and dest
    src_parsed = parse_url(bigstitcher_xml)
    dest_parsed = parse_url(dest)

    if image_names is not None:
        image_names_parsed = image_names.replace(" ", "").split(",")
    else:
        image_names_parsed = image_names

    save_points(
        bigstitcher_url=src_parsed, dest=dest_parsed, image_names=image_names_parsed
    )


def save_points(
    bigstitcher_url: URL, dest: URL, image_names: Iterable[str] | None = None
):
    bs_model = read_bigstitcher_xml(bigstitcher_url)
    save_interest_points(
        bs_model=bs_model,
        alignment_url=bigstitcher_url.parent,
        dest=dest,
        image_names=image_names,
    )


@cli.command("ngjson")
@click.option("--bigstitcher-xml", type=click.STRING, required=True)
@click.option("--points-url", type=click.STRING, default=None)
@click.option("--matches-url", type=click.STRING, default=None)
@click.option("--dest-path", type=click.STRING, required=True)
@click.option("--style", type=click.STRING, multiple=True)
def save_neuroglancer_json_cli(
    bigstitcher_xml: str,
    dest_path: str,
    points_url: str | None,
    matches_url: str | None,
    style: Sequence[NeuroglancerViewerStyle] | None = None,
):
    """
    Generate a neuroglancer viewer state as a JSON document.
    """
    log = structlog.get_logger()
    bigstitcher_xml_url = URL(bigstitcher_xml)
    if points_url is not None:
        points_url_parsed = URL(points_url)
    else:
        points_url_parsed = None

    if matches_url is not None:
        matches_url_parsed = URL(matches_url)
    else:
        matches_url_parsed = None

    dest_path_parsed = dest_path.rstrip("/")
    if style is None or len(style) < 1:
        style = neuroglancer_view_styles
    for _style in style:
        out_path = save_neuroglancer_json(
            bigstitcher_xml=bigstitcher_xml_url,
            dest_url=dest_path_parsed,
            points_url=points_url_parsed,
            matches_url=matches_url_parsed,
            style=_style,
        )
        log.info(f"Saved neuroglancer JSON state for style {_style} to {out_path}")


def save_neuroglancer_json(
    *,
    bigstitcher_xml: str | URL,
    points_url: str | URL | None,
    matches_url: str | URL | None,
    dest_url: str | URL,
    style: NeuroglancerViewerStyle,
) -> URL:
    bs_xml_parsed = parse_url(bigstitcher_xml)
    points_url_parsed = parse_url(points_url)
    matches_url_parsed = parse_url(matches_url)
    dest_url_parsed = parse_url(dest_url)
    bs_model = read_bigstitcher_xml(bs_xml_parsed)
    tilegroup_s3_url = get_tilegroup_url(bs_model)
    state = create_neuroglancer_state(
        image_url=tilegroup_s3_url,
        points_url=points_url_parsed,
        matches_url=matches_url_parsed,
        style=style,
    )
    out_fname = f"{style}.json"
    out_path = dest_url_parsed.joinpath(out_fname)

    if dest_url_parsed.scheme == "s3":
        fs = S3FileSystem()
    else:
        fs = fsspec.implementations.local.LocalFileSystem(auto_mkdir=True)

    with fs.open(out_path, mode="w") as fh:
        fh.write(json.dumps(state.to_json(), indent=2))

    return out_path


@cli.command("tabulate-matches")
@click.option("--bigstitcher-xml", type=click.STRING, required=True)
@click.option("--output", type=click.STRING, default="csv")
def tabulate_matches_cli(bigstitcher_xml: str, output: Literal["csv"] | None):
    """
    Generate a tabular representation of the correspondence metadata generated by bigstitcher.
    """
    pool = ThreadPoolExecutor(max_workers=16)
    bigstitcher_xml_url = URL(bigstitcher_xml)
    anon = True
    summarized = fetch_summarize_matches(
        bigstitcher_xml=bigstitcher_xml_url, pool=pool, anon=anon
    )
    
    if output == "csv":
        origin_xyz = summarized['image_origin_self'].to_numpy()
        csv_friendly = summarized.drop('image_origin_self').with_columns(
            image_origin_self_x=origin_xyz[:,0],
            image_origin_self_y=origin_xyz[:,1],
            image_origin_self_z=origin_xyz[:,2],
            )
        click.echo(csv_friendly.write_csv())
    else:
        raise ValueError(f'Format {output} is not recognized. Allowed values: ("csv",)')


@cli.command("view-bdv")
@click.option("--bigstitcher-xml", type=click.STRING, required=True)
@click.option("--transform-index", type=click.INT, default=-1)
@click.option("--interest-points", type=click.STRING, default=None)
@click.option("--host", type=click.STRING, default=None)
@click.option("--view-setups", type=click.STRING, default="all")
@click.option("--channels", type=click.STRING, default="all")
@click.option("--contrast-limits", type=click.STRING, default=None)
@click.option("--bind-address", type=click.STRING, default="localhost")
def view_bdv_cli(
    bigstitcher_xml: str,
    transform_index: int,
    host: str | None,
    view_setups: str,
    interest_points: str | None,
    channels: str,
    contrast_limits: str | None,
    bind_address: str,
):
    if contrast_limits is not None:
        contrast_limits_parsed = tuple(int(x) for x in contrast_limits.split(","))
        if len(contrast_limits_parsed) != 2:
            raise ValueError(
                f"Contrast limits must be two ints separated by a comma. Got {contrast_limits} instead."
            )
    else:
        contrast_limits_parsed = None

    if bind_address == "ip":
        neuroglancer.set_server_bind_address(socket.gethostbyname(socket.gethostname()))
    else:
        neuroglancer.set_server_bind_address("localhost")

    if channels != 'all':
        channels_parsed =  tuple(int(x) for x in channels.split(","))
    else:
        channels_parsed = 'all'

    viewer = view_bdv(
        bs_model=parse_url(bigstitcher_xml),
        host=parse_url(host),
        view_setups=view_setups,
        channels=channels_parsed,
        transform_index=transform_index,
        contrast_limits=contrast_limits_parsed,
        interest_points=interest_points,
    )

    print(f"Viewer link: {viewer}")
    input("Press enter to exit")


def view_bdv(
    *,
    bs_model: SpimData2,
    host: URL | None = None,
    view_setups: str | None = None,
    channels: str | None = None,
    contrast_limits: tuple[int, int] | None,
    interest_points: str | None = None,
    transform_index: int,
):
    display_settings: dict[str, int | None]
    if contrast_limits is not None:
        display_settings = {
            "start": contrast_limits[0],
            "stop": contrast_limits[1],
            "min": contrast_limits[0] - abs(contrast_limits[1] - contrast_limits[1]),
            "max": contrast_limits[1] + abs(contrast_limits[1] - contrast_limits[1]),
        }
    else:
        display_settings = {"start": None, "stop": None, "min": None, "max": None}
    state = bdv_to_neuroglancer(
        bs_model,
        anon=True,
        host=host,
        view_setups=view_setups,
        channels=channels,
        display_settings=display_settings,
        transform_index=transform_index,
        interest_points=interest_points,
    )
    viewer = neuroglancer.Viewer()
    viewer.set_state(state)
    return viewer
