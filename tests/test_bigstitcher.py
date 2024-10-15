from concurrent.futures import ThreadPoolExecutor
from pydantic_bigstitcher.transform import VectorMap, MatrixMap, HoAffine
import numpy as np
import pytest
from yarl import URL
from matchviz.bigstitcher import bdv_to_neuroglancer, get_interest_points, read_bigstitcher_xml, read_interest_points
import neuroglancer

from matchviz.transform import affine_to_array, array_to_affine, array_to_translate, compose_hoaffines, hoaffine_to_array, translate_to_array

@pytest.mark.parametrize('bigstitcher_xml', [0], indirect=True)
def test_read_points(bigstitcher_xml: str) -> None:
    bs_url = URL(bigstitcher_xml)
    pool = ThreadPoolExecutor(max_workers=8)
    bs_model = read_bigstitcher_xml(bs_url)

    result = get_interest_points(
        bs_model=bs_model,
        n5_interest_points_url=bs_url.parent / "interestpoints.n5",
        pool=pool,
        anon=True,
        )

def test_translate_to_array() -> None:
    tx_a: VectorMap = {'x': 1, 'y': 2}
    assert np.array_equal(translate_to_array(tx_a, ('x', 'y')), np.array([1,2]))

def test_affine_to_array() -> None:
    tx_a: MatrixMap = {'x': {'x': 1, 'y': 2}, 'y': {'x': 3, 'y': 4}}
    assert np.array_equal(affine_to_array(tx_a, ('x', 'y')), np.array([[1, 2], [3, 4]], dtype='float'))

def test_compose_transforms() -> None:
    dimensions=('x', 'y')
    tx_a: HoAffine = HoAffine(
        affine=array_to_affine(np.eye(2) * 4, dimensions=dimensions), 
        translation={'x': 1, 'y': 2}) 
    tx_b: HoAffine = HoAffine(
        affine=array_to_affine(np.eye(2) * 2, dimensions=dimensions), 
        translation={'x': 3, 'y': 4})
    observed = compose_hoaffines(
                tx_a, 
                tx_b, 
                dimensions=dimensions)
    
    expected = HoAffine(
        affine=array_to_affine(np.eye(2) * 8, dimensions=dimensions), 
        translation=array_to_translate(np.array([4,6]), dimensions=dimensions)
        )
    assert observed == expected

def test_hoaffine_to_array() -> None:
    tx_a: HoAffine = HoAffine(
        affine=array_to_affine(np.eye(2) * 4, dimensions=('x', 'y')), 
        translation={'x': 1, 'y': 2})
    observed = hoaffine_to_array(tx_a, dimensions=('x', 'y'))
    expected = np.array([[4, 0, 1],[0, 4, 2],[0,0,1]], dtype='float')
    assert np.array_equal(observed, expected)


@pytest.mark.parametrize('bigstitcher_xml', [0], indirect=True)
def test_bdv_to_neuroglancer(bigstitcher_xml: str) -> None:    
    host = None
    bs_model = read_bigstitcher_xml(bigstitcher_xml)
    viewer_state = bdv_to_neuroglancer(
        URL(bigstitcher_xml), 
        host=host,
        display_settings = {'start': 100, 'stop': 200, 'min': 0, 'max': 400},
        channels=[0],
        view_setups='all')
    viewer = neuroglancer.Viewer()
    viewer.set_state(viewer_state)