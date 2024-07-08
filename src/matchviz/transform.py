from pydantic_bigstitcher import AffineViewTransform
from typing import Callable, Sequence, TypeVar, TypeAlias, Literal, Generic
import numpy as np
import operator
from pydantic import BaseModel

T = TypeVar("T", bound=str)
Axes: TypeAlias = Literal["z", "y", "x"]
axes: tuple[Axes, ...] = ("z", "y", "x")

VectorMap: TypeAlias = dict[T, float]
MatrixMap: TypeAlias = dict[T, VectorMap[T]]


class HomoAffine(BaseModel, Generic[T]):
    """
    Model a homogeneous affine transformation with named axes. The transform is decomposed into
    a translation transform and an affine transform.
    """

    translation: VectorMap[T]
    affine: MatrixMap[T]


class Transform(BaseModel, Generic[T]):
    name: str
    type: str
    transform: HomoAffine[T]


def destringify_tuple(data: str) -> tuple[str, ...]:
    return tuple(data.split(" "))


def merge_vectors(
    a: VectorMap[T], b: VectorMap[T], op: Callable[[float, float], float]
) -> VectorMap[T]:
    out = {}
    for key in a.keys():
        out[key] = op(a[key], b[key])
    return out


def add_vectors(a: VectorMap[T], b: VectorMap[T]):
    return merge_vectors(a, b, operator.add)


def scale_vectors(a: VectorMap[T], b: VectorMap[T]):
    return merge_vectors(a, b, operator.mul)


def compose_transforms(transforms: Sequence[Transform[Axes]]) -> HomoAffine[Axes]:
    base_tx = {ax: 0.0 for ax in axes}
    base_affine = {
        oax: {iax: 1.0 if iax == oax else 0.0 for iax in axes} for oax in axes
    }
    for transform in transforms:
        base_tx = add_vectors(base_tx, transform.transform.translation)
        for ax in axes:
            base_affine[ax] = scale_vectors(
                base_affine[ax], transform.transform.affine[ax]
            )
    return HomoAffine(translation=base_tx, affine=base_affine)


def parse_transform(tx: AffineViewTransform) -> Transform:
    homo_affine_arrayed = np.array(
        tuple(float(x) for x in destringify_tuple(tx.affine))
    ).reshape(len(axes), len(axes) + 1)
    trans_array = homo_affine_arrayed[:, -1]
    aff_array = homo_affine_arrayed[:, :-1]
    aff_dict: MatrixMap[Axes] = {}
    trans_dict: VectorMap[Axes] = {
        ax: trans_array[idx] for idx, ax in enumerate(reversed(axes))
    }
    for oidx, oax in enumerate(reversed(axes)):
        aff_dict[oax] = {
            iax: aff_array[oidx][iidx] for iidx, iax in enumerate(reversed(axes))
        }

    return Transform(
        name=tx.name,
        type=tx.typ,
        transform=HomoAffine[Axes](translation=trans_dict, affine=aff_dict),
    )
