import pytest
import subprocess
import os


@pytest.mark.parametrize(
    "alignment_url",
    [
        "s3://aind-open-data/exaSPIM_708373_2024-04-02_19-49-38_alignment_2024-05-07_18-15-25/"
    ],
)
def test_save_points(tmpdir, alignment_url):
    out_path = os.path.join(str(tmpdir), "tile_alignment_visualization")
    run_result = subprocess.run(
        [
            "matchviz",
            "save-points",
            "--src",
            alignment_url,
            "--dest",
            out_path,
            "--ngjson",
            "test.json",
            "--nghost",
            "http://localhost:3000",
        ]
    )
    assert run_result.returncode == 0


@pytest.mark.parametrize(
    "alignment_url",
    [
        "s3://aind-open-data/exaSPIM_708373_2024-04-02_19-49-38_alignment_2024-05-07_18-15-25/"
    ],
)
def test_save_neuroglancer(tmpdir, alignment_url):
    points_url = os.path.join(alignment_url, "tile_alignment_visualization", "points")
    out_path = os.path.join(str(tmpdir), "tile_alignment_visualization", "neuroglancer")
    run_result = subprocess.run(
        [
            "matchviz",
            "ngjson",
            "--alignment-url",
            alignment_url,
            "--points-url",
            points_url,
            "--dest-path",
            out_path,
        ]
    )
    assert run_result.returncode == 0
