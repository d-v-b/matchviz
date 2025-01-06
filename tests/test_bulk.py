import pytest


@pytest.mark.skip
def test_bulk_save():
    import fsspec
    from s3fs import S3FileSystem

    fs = S3FileSystem()
    import os
    from matchviz.cli import save_points

    bucket = "aind-open-data"
    alignments = fs.glob(os.path.join(bucket, "exaSPIM_*alignment*"))
    out_prefix = "tile_alignment_visualization"
    results = {}

    def skipif(url: str) -> bool:
        fs, path = fsspec.url_to_fs(url)
        if not fs.exists(os.path.join(path, out_prefix, "neuroglancer.json")):
            return True
        return False

    alignments_filtered = tuple(filter(skipif, alignments))

    for alignment in alignments_filtered:
        print(alignment)
        url = os.path.join(f"s3://{alignment}")
        points_path = os.path.join(url, out_prefix, "points")
        ng_json_path = os.path.join(url, out_prefix, "neuroglancer.json")
        try:
            save_points(url=url, dest=points_path, ngjson=ng_json_path, nghost=None)
            ng_url = f"http://neuroglancer-demo.appspot.com/#!{ng_json_path}"
            print(alignment, ng_url)
            results[alignment] = ng_url
        except Exception as e:
            print(alignment, e)
            results[alignment] = e
