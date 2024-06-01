from matchviz import create_neuroglancer_state, save_points_tile, save_interest_points, image_name_to_tile_coord
import pytest

@pytest.mark.skip
def test_from_bdv_xml():
    base_url = "s3://aind-open-data/exaSPIM_708373_2024-04-02_19-49-38_alignment_2024-05-07_18-15-25/"
    save_interest_points(base_url=base_url)

@pytest.mark.skip
def test_create_neuroglancer_state():
    points_url = 'http://localhost:3000/foo'
    image_url = 's3://aind-open-data/exaSPIM_708373_2024-04-02_19-49-38/SPIM.ome.zarr/'
    state = create_neuroglancer_state(image_url=image_url, points_url=points_url)

@pytest.mark.skip
def test_viz(tmpdir):
    dataset = 'exaSPIM_708373_2024-04-02_19-49-38'
    alignment_id = 'alignment_2024-05-07_18-15-25'
    points = save_points_tile(
        dataset=dataset, 
        alignment_id=alignment_id,
        out_prefix=str(tmpdir))
    
def test_save_points_tile():
    tile_name = 'tile_x_0000_y_0000_z_0000_ch_488'
    image_url = 's3://aind-open-data/exaSPIM_708373_2024-04-02_19-49-38_flatfield-correction_2024-04-15_18-15-40/SPIM.ome.zarr/tile_x_0000_y_0000_z_0000_ch_488.zarr'
    alignment_url = 's3://aind-open-data/exaSPIM_708373_2024-04-02_19-49-38_alignment_2024-05-07_18-15-25/interestpoints.n5/tpId_0_viewSetupId_0/beads'
    out_prefix = 'points'

    save_points_tile(tile_name=tile_name, image_url=image_url, alignment_url=alignment_url, out_prefix=out_prefix)

@pytest.mark.parametrize('x', (0, 1))
@pytest.mark.parametrize('y', (0, 1))
@pytest.mark.parametrize('z', (0, 1))
@pytest.mark.parametrize('ch', ('488', '561'))
def test_image_name_to_coordinate(x, y, z, ch):
    image_name = f'tile_x_{x:04}_y_{y:04}_z_{z:04}_ch_{ch}.zarr'
    assert image_name_to_tile_coord(image_name) == {'x': x, 'y': y, 'z': z, 'ch' : ch}