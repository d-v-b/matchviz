from matchviz import (
    create_neuroglancer_state,
    get_tile_coords,
    load_points,
    parse_bigstitcher_xml_from_s3,
    parse_idmap,
    save_points_tile,
    save_interest_points,
    image_name_to_tile_coord,
)
from matchviz.annotation import AnnotationWriterFSSpec
import neuroglancer
import pytest


@pytest.mark.skip
def test_from_bdv_xml():
    base_url = "s3://aind-open-data/exaSPIM_708373_2024-04-02_19-49-38_alignment_2024-05-07_18-15-25/"
    save_interest_points(base_url=base_url)


@pytest.mark.skip
def test_create_neuroglancer_state():
    points_url = "http://localhost:3000/foo"
    image_url = "s3://aind-open-data/exaSPIM_708373_2024-04-02_19-49-38/SPIM.ome.zarr/"
    _ = create_neuroglancer_state(image_url=image_url, points_url=points_url)


@pytest.mark.skip
def test_viz(tmpdir):
    dataset = "exaSPIM_708373_2024-04-02_19-49-38"
    alignment_id = "alignment_2024-05-07_18-15-25"
    _ = save_points_tile(
        dataset=dataset, alignment_id=alignment_id, out_prefix=str(tmpdir)
    )


@pytest.mark.skip
def test_save_points_tile():
    bs_url = "s3://aind-open-data/exaSPIM_708373_2024-04-02_19-49-38_alignment_2024-05-07_18-15-25/"
    bs_model = parse_bigstitcher_xml_from_s3(bs_url)
    tile_name = "tile_x_0000_y_0000_z_0000_ch_488"
    alignment_url = "s3://aind-open-data/exaSPIM_708373_2024-04-02_19-49-38_alignment_2024-05-07_18-15-25/interestpoints.n5/tpId_0_viewSetupId_0/"
    out_prefix = "points_out"
    tile_coords = get_tile_coords(bs_model)
    save_points_tile(
        vs_id=0,
        tile_name=tile_name,
        alignment_url=alignment_url,
        out_prefix=out_prefix,
        tile_coords=tile_coords,
    )


@pytest.mark.parametrize("x", (0, 1))
@pytest.mark.parametrize("y", (0, 1))
@pytest.mark.parametrize("z", (0, 1))
@pytest.mark.parametrize("ch", ("488", "561"))
def test_image_name_to_coordinate(x, y, z, ch):
    image_name = f"tile_x_{x:04}_y_{y:04}_z_{z:04}_ch_{ch}.zarr"
    assert image_name_to_tile_coord(image_name) == {"x": x, "y": y, "z": z, "ch": ch}


def test_parse_idmap():
    data = {"0,1,beads": 0, "0,3,beads": 1}
    assert parse_idmap(data) == {(0, 1, "beads"): 0, (0, 3, "beads"): 1}


@pytest.mark.skip
def test_load_points():
    url = "s3://aind-open-data/exaSPIM_708373_2024-04-02_19-49-38_alignment_2024-05-07_18-15-25/interestpoints.n5/tpId_0_viewSetupId_3/beads/"
    _ = load_points(url)


@pytest.mark.skip
def test_write_annotations():
    coordinate_space = neuroglancer.CoordinateSpace(
        names=["x", "y", "z"], scales=[1, 1, 1], units=["nm", "nm", "nm"]
    )
    writer = AnnotationWriterFSSpec(
        coordinate_space=coordinate_space, annotation_type="point"
    )
    writer.add_point((0, 0, 0))
    writer.add_point((0, 0, 1))
    writer.add_point((0, 1, 1))
    writer.add_point((0, 1, 0))
    # mock s3 here
    writer.write("")
