from pydantic_bigstitcher.transform import VectorMap, MatrixMap, HoAffine
import numpy as np
from matchviz.bigstitcher import affine_to_array, array_to_affine, array_to_translate, compose_transforms, translate_to_array


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
    observed = compose_transforms(
                tx_a, 
                tx_b, 
                dimensions=dimensions)
    
    expected = HoAffine(
        affine=array_to_affine(np.eye(2) * 8, dimensions=dimensions), 
        translation=array_to_translate(np.array([4,6]), dimensions=dimensions)
        )
    assert observed == expected