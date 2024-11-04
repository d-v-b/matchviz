from itertools import product


import numpy as np
import pydantic_bigstitcher.transform


from typing import Any, Iterable


def affine_to_array(
    tx: pydantic_bigstitcher.transform.MatrixMap[Any], dimensions: Iterable[str]
) -> np.ndarray:
    assert set(dimensions) == set(tx.keys())
    out = np.zeros((len(tx),) * 2)

    for idx, (dim_outer, dim_inner) in enumerate(product(dimensions, dimensions)):
        out.ravel()[idx] = tx[dim_outer][dim_inner]

    return out


def translate_to_array(
    tx: pydantic_bigstitcher.transform.VectorMap[Any], dimensions: Iterable[str]
) -> np.ndarray:
    assert set(dimensions) == set(tx.keys())
    out = np.zeros(len(tx))
    for idx, dim in enumerate(dimensions):
        out[idx] = tx[dim]
    return out


def hoaffine_to_array(
    tx: pydantic_bigstitcher.transform.HoAffine[Any], dimensions: Iterable[str]
) -> np.ndarray:
    out = np.eye((len(tuple(dimensions)) + 1))
    affine = affine_to_array(tx.affine, dimensions)
    translation = translate_to_array(tx.translation, dimensions)
    out[:-1, :-1] = affine
    out[:-1, -1] = translation
    return out


def array_to_hoaffine(
    array: np.ndarray, dimensions: tuple[str, ...]
) -> pydantic_bigstitcher.transform.HoAffine[Any]:
    return pydantic_bigstitcher.transform.HoAffine(
        affine=array_to_affine(array[:-1, :-1], dimensions=dimensions),
        translation=array_to_translate(array[:-1, -1], dimensions=dimensions),
    )


def array_to_translate(
    array: np.ndarray, dimensions: Iterable[str]
) -> pydantic_bigstitcher.transform.VectorMap[Any]:
    return dict(zip(dimensions, array))


def array_to_affine(
    array: np.ndarray, dimensions: Iterable[str]
) -> pydantic_bigstitcher.transform.MatrixMap[Any]:
    return dict(zip(dimensions, (array_to_translate(row, dimensions) for row in array)))

def compose_hoaffines(
    tx_a: pydantic_bigstitcher.transform.HoAffine[Any],
    tx_b: pydantic_bigstitcher.transform.HoAffine[Any],
) -> pydantic_bigstitcher.transform.HoAffine[Any]:

    dimensions = tx_a.affine.keys()
    homogeneous_matrix = hoaffine_to_array(
        tx_b, dimensions=dimensions
    ) @ hoaffine_to_array(tx_a, dimensions=dimensions)

    new_affine = homogeneous_matrix[:-1, :-1]
    new_trans = homogeneous_matrix[:-1, -1]

    return pydantic_bigstitcher.transform.HoAffine(
        affine=array_to_affine(new_affine, dimensions=dimensions),
        translation=array_to_translate(new_trans, dimensions=dimensions),
    )


def apply_hoaffine(
    *,
    tx: pydantic_bigstitcher.transform.HoAffine[Any],
    data: np.ndarray,
    dimensions: Iterable[str],
) -> np.ndarray:
    if data.shape[-1] == len(tx.translation):
        # convert input points to homogeneous form
        points_parsed = np.concatenate((data, np.ones((data.shape[0], 1))), axis=1)
        result = (hoaffine_to_array(tx, dimensions=dimensions) @ points_parsed.T).T[
            :, :-1
        ]
    else:
        points_parsed = data
        result = hoaffine_to_array(tx, dimensions=dimensions) @ points_parsed.T
    return result
