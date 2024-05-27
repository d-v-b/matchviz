from correspondence_viz import create_neuroglancer_state, save_points_tile, save_interest_points
import pytest

@pytest.mark.skip
def test_from_bdv_xml():
    base_url = "s3://aind-open-data/exaSPIM_708373_2024-04-02_19-49-38_alignment_2024-05-07_18-15-25/"
    save_interest_points(base_url=base_url)

def test_create_neuroglancer_state():
    points_url = 'http://localhost:3000/foo'
    image_url = 's3://aind-open-data/exaSPIM_708373_2024-04-02_19-49-38/SPIM.ome.zarr/'
    state = create_neuroglancer_state(image_url=image_url, points_url=points_url)
    assert False

@pytest.mark.skip
def test_viz(tmpdir):
    dataset = 'exaSPIM_708373_2024-04-02_19-49-38'
    alignment_id = 'alignment_2024-05-07_18-15-25'
    points = save_points_tile(
        dataset=dataset, 
        alignment_id=alignment_id,
        out_prefix=str(tmpdir))