import json
import os

from yarl import URL
from matchviz import (
    create_neuroglancer_state,
)
from matchviz.annotation import AnnotationWriterFSSpec
import neuroglancer
from matchviz.cli import plot_matches
import pytest

from matchviz.bigstitcher import (
    parse_idmap,
    read_bigstitcher_xml,
    save_annotations,
    save_interest_points,
    read_interest_points,
)


@pytest.mark.skip
def test_from_bdv_xml():
    base_url = "s3://aind-open-data/exaSPIM_708373_2024-04-02_19-49-38_alignment_2024-05-07_18-15-25/"
    save_interest_points(base_url=base_url)


def test_create_neuroglancer_state():
    points_url = "s3://aind-open-data/exaSPIM_715345_2024-06-07_10-03-37_alignment_2024-07-01_19-45-38/tile_alignment_visualization/points/"
    matches_url = "s3://aind-open-data/exaSPIM_715345_2024-06-07_10-03-37_alignment_2024-07-01_19-45-38/tile_alignment_visualization/matches/"
    image_url = "s3://aind-open-data/exaSPIM_715345_2024-06-07_10-03-37/SPIM.ome.zarr/"
    state = create_neuroglancer_state(
        image_url=image_url,
        points_url=points_url,
        matches_url=matches_url,
        style="images_combined",
    )
    print(json.dumps(state.to_json(), indent=2))


@pytest.mark.skip
def test_viz(tmpdir):
    dataset = "exaSPIM_708373_2024-04-02_19-49-38"
    alignment_id = "alignment_2024-05-07_18-15-25"
    _ = save_annotations(
        dataset=dataset, alignment_id=alignment_id, dest_url=str(tmpdir)
    )


def test_save_points_tile(tmpdir):
    bs_url = "s3://aind-open-data/exaSPIM_708373_2024-04-02_19-49-38_alignment_2024-05-07_18-15-25/"
    bs_model = read_bigstitcher_xml(os.path.join(bs_url, "bigstitcher.xml"))
    alignment_url = "s3://aind-open-data/exaSPIM_708373_2024-04-02_19-49-38_alignment_2024-05-07_18-15-25/interestpoints.n5/tpId_0_viewSetupId_0/"
    save_annotations(
        bs_model=bs_model,
        image_id="0",
        alignment_url=alignment_url,
        dest_url=str(tmpdir),
    )


def test_parse_idmap():
    data = {"0,1,beads": 0, "0,3,beads": 1}
    assert parse_idmap(data) == {(0, 1, "beads"): 0, (0, 3, "beads"): 1}


@pytest.mark.skip
def test_load_points():
    url = "s3://aind-open-data/exaSPIM_708373_2024-04-02_19-49-38_alignment_2024-05-07_18-15-25/interestpoints.n5"
    _ = read_interest_points(store=url, path="tpId_0_viewSetupId_3/beads/")


def test_plot_points(tmpdir):
    bs_xml = URL(
        "s3://aind-open-data/exaSPIM_708373_2024-04-02_19-49-38_alignment_2024-05-07_18-15-25/bigstitcher.xml"
    )
    _ = plot_matches(bigstitcher_xml=bs_xml, metric="transform_error_max")


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
