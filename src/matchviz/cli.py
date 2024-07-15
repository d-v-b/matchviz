from __future__ import annotations
import os
import json
from typing import Sequence
import click
import fsspec
import logging
from matchviz import (
    create_neuroglancer_state,
    get_tilegroup_s3_url,
    parse_bigstitcher_xml_from_s3,
    save_interest_points,
)
from matchviz.neuroglancer_styles import (
    NeuroglancerViewerStyle,
    fnames,
    neuroglancer_view_styles,
)


@click.group("matchviz")
def cli(): ...


@cli.command("save-points")
@click.option("--src", type=click.STRING, required=True)
@click.option("--dest", type=click.STRING, required=True)
def save_interest_points_cli(src: str, dest: str):
    logging.basicConfig(level="INFO")
    # strip trailing '/' from src and dest
    src_parsed = src.rstrip("/")
    dest_parsed = dest.rstrip("/")
    save_points(url=src_parsed, dest=dest_parsed)


def save_points(url: str, dest: str):
    bs_model = parse_bigstitcher_xml_from_s3(url)
    save_interest_points(bs_model=bs_model, base_url=url, out_prefix=dest)


@cli.command("ngjson")
@click.option("--alignment-url", type=click.STRING, required=True)
@click.option("--points-url", type=click.STRING, required=True)
@click.option("--dest-path", type=click.STRING, required=True)
@click.option("--style", type=click.STRING, multiple=True)
def save_neuroglancer_json_cli(
    alignment_url: str,
    dest_path: str,
    points_url: str,
    style: Sequence[NeuroglancerViewerStyle] | None = None,
):
    logger = logging.getLogger(__name__)
    # todo: make this sensitive
    logger.setLevel("INFO")
    alignment_url_parsed = alignment_url.rstrip("/")
    dest_path_parsed = dest_path.rstrip("/")
    if style is None or len(style) < 1:
        style = neuroglancer_view_styles
    for _style in style:
        out_path = save_neuroglancer_json(
            alignment_url=alignment_url_parsed,
            dest_path=dest_path_parsed,
            points_url=points_url,
            style=_style,
        )
        logger.info(f"Saved neuroglancer JSON state for style {_style} to {out_path}")


def save_neuroglancer_json(
    *,
    alignment_url: str,
    points_url: str,
    dest_path: str,
    style: NeuroglancerViewerStyle,
) -> str:
    bs_model = parse_bigstitcher_xml_from_s3(alignment_url)
    tilegroup_s3_url = get_tilegroup_s3_url(bs_model)
    state = create_neuroglancer_state(
        image_url=tilegroup_s3_url, points_url=points_url, style=style
    )
    out_fname = f"{style}.json"
    out_path = os.path.join(dest_path, out_fname)
    if dest_path.startswith("s3://"):
        fs, _ = fsspec.url_to_fs(dest_path)
    else:
        fs, _ = fsspec.url_to_fs(dest_path, auto_mkdir=True)

    with fs.open(out_path, mode="w") as fh:
        fh.write(json.dumps(state.to_json()))

    return out_path


@cli.command("html-report")
@click.argument("dest_url", type=click.STRING)
@click.argument("ngjson_url", type=click.STRING)
@click.option("--header", type=click.STRING)
@click.option("--title", type=click.STRING)
def html_report_cl(
    dest_url: str, ngjson_url: str, header: str | None, title: str | None
):
    html_report(dest_url=dest_url, ngjson_url=ngjson_url, header=header, title=title)


def html_report(dest_url: str, ngjson_url: str, header: str | None, title: str | None):
    if title is None:
        title = "Neuroglancer URLs"
    list_items = ()
    for key, value in fnames.items():
        description = value.description
        ng_url = os.path.join(ngjson_url, value.name)
        neuroglancer_url = f"http://neuroglancer-demo.appspot.com/#!{ng_url}"
        list_items += (f"<li><a href={neuroglancer_url}>{description}</a></li>",)
    # obviously jinja is better than this
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
    </head>
    <body>
        <h1>{header}</h1>
        <div>
            <p><ul>
            {list_items[0]}
            {list_items[1]}
            </ul>
            </p>
        </div>
    </body>
    </html>
    """

    fs, path = fsspec.url_to_fs(dest_url)
    with fs.open(path, mode="w") as fh:
        fh.write(html)
    fh.setxattr(content_type="text/html")
