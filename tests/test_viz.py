from correspondence_viz import create_annotation_layer_from_points, visualize_points
import pytest

def test_from_bdv_xml():
    bdv_url = "s3://aind-open-data/exaSPIM_708373_2024-04-02_19-49-38_alignment_2024-05-07_18-15-25/bigstitcher.xml"
    visualize_points(bdv_url=bdv_url)

@pytest.mark.skip
def test_viz():
    dataset = 'exaSPIM_708373_2024-04-02_19-49-38'
    alignment_id = 'alignment_2024-05-07_18-15-25'
    points = create_annotation_layer_from_points(
        dataset=dataset, 
        alignment_id=alignment_id)