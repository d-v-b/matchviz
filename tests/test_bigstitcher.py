from concurrent.futures import ThreadPoolExecutor
from pydantic_bigstitcher.transform import VectorMap, MatrixMap, HoAffine
import numpy as np
import pytest
from yarl import URL
from matchviz.bigstitcher import (
    spimdata_to_neuroglancer,
    read_all_interest_points,
    read_bigstitcher_xml,
)
import neuroglancer

from matchviz.transform import (
    affine_to_array,
    array_to_affine,
    compose_hoaffines,
    hoaffine_to_array,
    translate_to_array,
)


@pytest.mark.parametrize("bigstitcher_xml", [0], indirect=True)
def test_read_points(bigstitcher_xml: str) -> None:
    bs_url = URL(bigstitcher_xml)
    pool = ThreadPoolExecutor(max_workers=8)
    bs_model = read_bigstitcher_xml(bs_url)

    _ = read_all_interest_points(
        bs_model=bs_model,
        store=bs_url.parent / "interestpoints.n5",
        pool=pool,
    )


def test_translate_to_array() -> None:
    tx_a: VectorMap = {"x": 1, "y": 2}
    assert np.array_equal(translate_to_array(tx_a, ("x", "y")), np.array([1, 2]))


def test_affine_to_array() -> None:
    tx_a: MatrixMap = {"x": {"x": 1, "y": 2}, "y": {"x": 3, "y": 4}}
    assert np.array_equal(
        affine_to_array(tx_a, ("x", "y")), np.array([[1, 2], [3, 4]], dtype="float")
    )


def test_compose_transforms() -> None:
    dimensions = ("x", "y")
    scale_a = (2, 3)
    trans_a = (1, 2)

    matrix_a = [[scale_a[0], 0, trans_a[0]], [0, scale_a[1], trans_a[1]], [0, 0, 1]]

    scale_b = (0.5, 0.8)
    trans_b = (3, 4)

    matrix_b = [[scale_b[0], 0, trans_b[0]], [0, scale_b[1], trans_b[1]], [0, 0, 1]]

    tx_a: HoAffine = HoAffine(
        affine=array_to_affine(np.eye(2) * scale_a, dimensions=dimensions),
        translation=dict(zip(dimensions, trans_a)),
    )
    tx_b: HoAffine = HoAffine(
        affine=array_to_affine(np.eye(2) * scale_b, dimensions=dimensions),
        translation=dict(zip(dimensions, trans_b)),
    )

    observed = compose_hoaffines(tx_a, tx_b)

    expected_mat = np.matmul(matrix_a, matrix_b)

    expected = HoAffine(
        affine=array_to_affine(expected_mat[:-1, :-1], dimensions=dimensions),
        translation=dict(zip(dimensions, expected_mat[:, -1])),
    )
    assert observed == expected


def test_hoaffine_to_array() -> None:
    tx_a: HoAffine = HoAffine(
        affine=array_to_affine(np.eye(2) * 4, dimensions=("x", "y")),
        translation={"x": 1, "y": 2},
    )
    observed = hoaffine_to_array(tx_a, dimensions=("x", "y"))
    expected = np.array([[4, 0, 1], [0, 4, 2], [0, 0, 1]], dtype="float")
    assert np.array_equal(observed, expected)


@pytest.mark.parametrize("bigstitcher_xml", [0], indirect=True)
def test_bdv_to_neuroglancer(bigstitcher_xml: str) -> None:
    host = None
    viewer_state = spimdata_to_neuroglancer(
        URL(bigstitcher_xml),
        host=host,
        display_settings={"start": 100, "stop": 200, "min": 0, "max": 400},
        channels=[0],
        view_setups="all",
    )
    viewer = neuroglancer.Viewer()
    viewer.set_state(viewer_state)
