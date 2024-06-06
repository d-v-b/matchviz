import json
import click
from matchviz import create_neuroglancer_state, get_tilegroup_s3_url, parse_bigstitcher_xml_from_s3, save_interest_points

@click.group('matchviz')
def cli():
    ...

@cli.command('save-points')
@click.argument('url', type=click.STRING)
@click.argument('dest', type=click.STRING)
@click.option('--ngjson', type=click.STRING)
@click.option('--nghost', type=click.STRING)
def save_interest_points_cli(url: str, dest: str, ngjson: str | None, nghost: str | None):
    
    bs_model = parse_bigstitcher_xml_from_s3(url)
    save_interest_points(bs_model=bs_model, base_url=url, out_prefix=dest)

    if ngjson is not None:
        tilegroup_s3_url = get_tilegroup_s3_url(bs_model)
        state = create_neuroglancer_state(
            image_url=tilegroup_s3_url,
            points_host = nghost,
            points_path=dest)
        
        with open(ngjson, mode='w') as fh:
            fh.write(json.dumps(state.to_json()))


@cli.command('ngjson')
@click.argument('url', type=click.STRING)
@click.argument('dest', type=click.STRING)
@click.argument('points_path', type=click.STRING)
@click.argument('points_host', type=click.STRING)
def neuroglancer_json_cli(url: str, dest: str, points_path: str, points_host: str):
    bs_model = parse_bigstitcher_xml_from_s3(url)
    tilegroup_s3_url = get_tilegroup_s3_url(bs_model)
    state = create_neuroglancer_state(
        image_url = tilegroup_s3_url,
        points_path=points_path, 
        points_host=points_host,)
    
    with open(dest, mode='w') as fh:
        fh.write(json.dumps(state.to_json()))
    
