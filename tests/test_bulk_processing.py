import os
from matchviz.cli import save_points, save_neuroglancer_json

from s3fs import S3FileSystem
import pytest

fs = S3FileSystem()
bucket = "aind-open-data"
alignments = fs.glob(os.path.join(bucket, "exaSPIM_*alignment*"))

OUT_PREFIX = "tile_alignment_visualization"


def test_create_urls(tmpdir):
    dset = "exaSPIM_704523_2024-05-03_11-03-13_alignment_2024-05-23_22-32-43"
    align_url = f"s3://aind-open-data/{dset}/bigstitcher.xml"
    _ = os.path.join(align_url, OUT_PREFIX)
    styles = "images_combined", "images_split"
    for style in styles:
        save_neuroglancer_json(
            bigstitcher_xml=align_url,
            points_url=os.path.join(align_url, OUT_PREFIX, "points"),
            matches_url=os.path.join(align_url, OUT_PREFIX, "matches"),
            dest_url=str(tmpdir),
            style=style,
        )


@pytest.mark.skip
def test_bulk_convert():
    results = {}

    for alignment in alignments:
        print(alignment)
        url = os.path.join(f"s3://{alignment}")
        points_path = os.path.join(url, OUT_PREFIX, "points")
        ng_json_path = os.path.join(url, OUT_PREFIX, "neuroglancer.json")
        try:
            save_points(url=url, dest=points_path, ngjson=ng_json_path, nghost=None)
            ng_url = f"http://neuroglancer-demo.appspot.com/#!{ng_json_path}"
            print(alignment, ng_url)
            results[alignment] = ng_url
        except Exception as e:
            print(alignment, e)
            results[alignment] = e
