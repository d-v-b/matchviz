import pytest
import subprocess
import os
from click.testing import CliRunner
from matchviz.cli import tabulate_matches_cli, save_interest_points_cli


@pytest.mark.parametrize("bigstitcher_xml", [0], indirect=True)
def test_save_points(tmpdir, bigstitcher_xml):
    runner = CliRunner()
    dest = str(tmpdir)
    result = runner.invoke(
        save_interest_points_cli,
        ["--bigstitcher-xml", bigstitcher_xml, "--dest", dest],
    )
    assert result.exit_code == 0


@pytest.mark.skip
@pytest.mark.parametrize("bigstitcher_xml", [0], indirect=True)
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


@pytest.mark.parametrize("bigstitcher_xml", [0], indirect=True)
def test_tabulate_matches(bigstitcher_xml):
    runner = CliRunner()
    result = runner.invoke(tabulate_matches_cli, ["--bigstitcher-xml", bigstitcher_xml])
    assert result.exit_code == 0
    head = (
        "image_id_self,image_id_other,num_matches,error_min,error_max,error_mean,image_name_self,image_name_other,image_origin_self_x,image_origin_self_y,image_origin_self_z\n"
        "0,1,4437,68.1842636256799,129.13630390431865,93.92395097473285,tile_x_0000_y_0000_z_0000_ch_488,tile_x_0000_y_0001_z_0000_ch_488,-7203.87535019432,-23429.676463043375,-28695.13399337075\n"
    )
    assert result.output.startswith(head)
