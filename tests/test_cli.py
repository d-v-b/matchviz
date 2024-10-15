import pytest
import subprocess
import os
from click.testing import CliRunner
from matchviz.cli import tabulate_matches_cli, save_interest_points_cli


@pytest.mark.parametrize('bigstitcher_xml', [0], indirect=True)
def test_save_points(tmpdir, bigstitcher_xml):
    runner = CliRunner()
    dest = str(tmpdir)
    result = runner.invoke(
        save_interest_points_cli, 
        ["--bigstitcher-xml", bigstitcher_xml, '--dest', dest],
        )
    assert result.exit_code == 0


@pytest.mark.skip
@pytest.mark.parametrize('bigstitcher_xml', [0], indirect=True)
def test_save_neuroglancer(tmpdir, bigstitcher_xml):
    points_url = os.path.join(bigstitcher_xml, "tile_alignment_visualization", "points")
    out_path = os.path.join(str(tmpdir), "tile_alignment_visualization", "neuroglancer")
    run_result = subprocess.run(
        [
            "matchviz",
            "ngjson",
            "--alignment-url",
            bigstitcher_xml,
            "--points-url",
            points_url,
            "--dest-path",
            out_path,
        ]
    )
    assert run_result.returncode == 0


@pytest.mark.parametrize('bigstitcher_xml', [0], indirect=True)
def test_summarize_points(bigstitcher_xml):
    runner = CliRunner()
    result = runner.invoke(tabulate_matches_cli, ["--bigstitcher-xml", bigstitcher_xml])
    assert result.exit_code == 0
    head = (
        "image_name,id_self,id_other,num_matches,x_coord_self,y_coord_self,z_coord_self\n"
        "tile_x_0000_y_0000_z_0000_ch_488,0,1,4437,-7096.0,-23408.0003,-28672.0\n"
        )
    assert result.output.startswith(head)
