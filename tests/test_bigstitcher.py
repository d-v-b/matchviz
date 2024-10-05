from pydantic_bigstitcher.transform import VectorMap, MatrixMap, HoAffine
import numpy as np
from yarl import URL
from matchviz.bigstitcher import affine_to_array, array_to_affine, array_to_translate, bdv_to_neuroglancer, compose_hoaffines, hoaffine_to_array, read_bigstitcher_xml, translate_to_array
import neuroglancer

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


def test_bdv_to_neuroglancer() -> None:
    bs_xml = "s3://aind-open-data/exaSPIM_708373_2024-04-02_19-49-38_alignment_2024-05-07_18-15-25/bigstitcher.xml"
    bs_model = read_bigstitcher_xml(bs_xml)
    viewer_state = bdv_to_neuroglancer(URL(bs_xml))
    viewer = neuroglancer.Viewer()
    viewer.set_state(viewer_state)
    print(viewer)
    breakpoint()