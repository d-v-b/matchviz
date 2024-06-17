from __future__ import annotations
import os
import json
import click
import fsspec
import logging
from matchviz import (
    create_neuroglancer_state,
    get_tilegroup_s3_url,
    parse_bigstitcher_xml_from_s3,
    save_interest_points
)
from matchviz.neuroglancer_styles import NeuroglancerViewerStyle, fnames

@click.group("matchviz")
def cli(): ...


@cli.command("save-points")
@click.argument("url", type=click.STRING)
@click.argument("dest", type=click.STRING)
@click.option("--ngjson", type=click.STRING)
@click.option("--nghost", type=click.STRING)
def save_interest_points_cli(
    url: str, dest: str, ngjson: str | None, nghost: str | None
):
    logging.basicConfig(level="INFO")
    save_points(url=url, dest=dest, ngjson=ngjson, nghost=nghost)


def save_points(url: str, dest: str, ngjson: str | None, nghost: str | None):
    bs_model = parse_bigstitcher_xml_from_s3(url)
    save_interest_points(bs_model=bs_model, base_url=url, out_prefix=dest)

    if ngjson is not None:
        if nghost is not None:
            ng_url = os.path.join(nghost, dest)
        else:
            ng_url = dest
        tilegroup_s3_url = get_tilegroup_s3_url(bs_model)
        state = create_neuroglancer_state(image_url=tilegroup_s3_url, points_url=ng_url)

        fs, _ = fsspec.url_to_fs(ngjson)

        if not ngjson.startswith("s3://"):
            if nghost is None:
                raise ValueError(
                    "You must provide a hostname to generate a neuroglancer viewer state if you are saving to local storage"
                )

        with fs.open(ngjson, mode="w") as fh:
            fh.write(json.dumps(state.to_json()))


@cli.command("ngjson")
@click.argument("alignment_url", type=click.STRING)
@click.argument("points_url", type=click.STRING)
@click.argument("dest_path", type=click.STRING)
@click.option("--style", type=click.STRING, multiple=True)
def save_neuroglancer_json_cli(
    alignment_url: str, 
    dest_path: str, 
    points_url: str, 
    style: list[NeuroglancerViewerStyle] | None = None):
    if style is None or len(style) < 1:
        style = ["images_combined", "images_split"]
    for _style in style:
        save_neuroglancer_json(
            alignment_url=alignment_url, 
            dest_path=dest_path, 
            points_url=points_url, 
            style=_style)


def save_neuroglancer_json(
        *,
        alignment_url: str, 
        points_url: str, 
        dest_path: str, 
        style: NeuroglancerViewerStyle):
    bs_model = parse_bigstitcher_xml_from_s3(alignment_url)
    tilegroup_s3_url = get_tilegroup_s3_url(bs_model)
    state = create_neuroglancer_state(
        image_url=tilegroup_s3_url,
        points_url=points_url,
        style=style
    )
    out_fname = f"{style}.json"
    out_path = os.path.join(dest_path, out_fname)
    fs, _ = fsspec.url_to_fs(dest_path)

    with fs.open(out_path, mode="w") as fh:
        fh.write(json.dumps(state.to_json()))

@cli.command('html-report')
@click.argument('dest_url', type=click.STRING)
@click.argument('ngjson_url', type=click.STRING)
@click.option('--header', type=click.STRING)
@click.option('--title', type=click.STRING)
def html_report_cl(dest_url: str, ngjson_url: str, header: str | None, title: str | None):
    html_report(dest_url=dest_url, ngjson_url=ngjson_url, header=header, title=title)

def html_report(
        dest_url: str, 
        ngjson_url: str, 
        header: str | None, 
        title: str | None):
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
    with fs.open(path, mode='w') as fh:
        fh.write(html)
    fh.setxattr(content_type="text/html")
    
    