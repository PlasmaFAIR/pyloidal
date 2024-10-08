from itertools import product
from typing import Any

import numpy as np
import pytest

from pyloidal.cocos import Sigma, Transform, identify_cocos

ALL_COCOS = list(range(1, 9)) + list(range(11, 19))


def _identify_cocos_inputs(
    cocos: int,
    antiparallel_field_and_current: bool,
    use_minor_radii: bool,
    use_clockwise_phi: bool,
) -> tuple[dict[str, Any], tuple[int, ...]]:
    """Generates inputs for ``identify_cocos`` for a given COCOS"""
    # Set up cocos 1 kwargs and modify accordingly
    kwargs: dict[str, Any] = {
        "b_toroidal": 2.5,
        "plasma_current": 1e6,
        "poloidal_flux": np.linspace(0, 2, 3),
        "safety_factor": np.linspace(0.5, 1.5, 3),
    }
    expected: list[int] = [cocos]

    even_cocos = not bool(cocos % 2)
    base_cocos = (cocos % 10) - even_cocos
    if base_cocos in (3, 5):
        kwargs["safety_factor"] *= -1
    if base_cocos in (3, 7):
        kwargs["poloidal_flux"] = kwargs["poloidal_flux"][::-1]
    if antiparallel_field_and_current:
        kwargs["b_toroidal"] *= -1
        kwargs["safety_factor"] *= -1
    if cocos >= 10:
        kwargs["safety_factor"] *= 2 * np.pi
        kwargs["poloidal_flux"] *= 2 * np.pi

    if use_clockwise_phi:
        kwargs["clockwise_phi"] = even_cocos
    else:
        expected.append(expected[0] + (-1 if even_cocos else 1))
    if use_minor_radii:
        kwargs["minor_radii"] = np.linspace(0, 2, 3)
    else:
        expected += [x + (-10 if cocos >= 10 else 10) for x in expected]

    return kwargs, tuple(sorted(expected))


@pytest.mark.parametrize(
    ("cocos", "antiparallel", "use_minor_radii", "use_clockwise_phi"),
    product(ALL_COCOS, *(3 * [[True, False]])),
)
def test_identify_cocos(
    cocos: int,
    antiparallel: bool,
    use_minor_radii: bool,
    use_clockwise_phi: bool,
):
    kwargs, expected = _identify_cocos_inputs(
        cocos, antiparallel, use_minor_radii, use_clockwise_phi
    )
    actual = identify_cocos(**kwargs)
    np.testing.assert_array_equal(actual, expected)


def test_cocos_transform():
    # TODO should be parametrized
    assert Transform(1, 3).poloidal == -1
    for cocos in range(1, 9):
        assert Transform(cocos, cocos + 10).inv_psi != 1
        for cocos_add in (cocos + x * 10 for x in range(2)):
            for key in ("b_toroidal", "toroidal", "poloidal", "q"):
                assert getattr(Transform(cocos_add, cocos_add), key) == 1


def test_sigma_bad_inputs():
    """Test that Sigma raises an excpetion when given an inputs not in ``(-1, 1)``"""
    with pytest.raises(ValueError, match="B_poloidal"):
        Sigma(B_poloidal=2, r_phi_z=1, r_theta_phi=-1)

    with pytest.raises(ValueError, match="r_phi_z"):
        Sigma(B_poloidal=-1, r_phi_z=0, r_theta_phi=-1)

    with pytest.raises(ValueError, match="r_theta_phi"):
        Sigma(B_poloidal=-1, r_phi_z=1, r_theta_phi=-5)
