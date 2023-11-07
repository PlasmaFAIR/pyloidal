import numpy as np
import pytest

from pyloidal.cocos import identify_cocos, Transform

# Create identify_cocos test cases for COCOS 1, 3, 5, 7
# TODO include tests for negative/antiparallel b_toroidal and plasma_current
# TODO include tests for multiple returns

identify_cocos_tests = {}
odds = {
    1: {
        "b_toroidal": 2.5,
        "plasma_current": 1e6,
        "poloidal_flux": np.linspace(0, 2, 3),
        "safety_factor": np.linspace(0.5, 1.5, 3),
        "clockwise_phi": False,
        "minor_radii": np.linspace(0, 2, 3),
    },
    3: {
        "b_toroidal": 2.5,
        "plasma_current": 1e6,
        "poloidal_flux": np.linspace(2, 0, 3),
        "safety_factor": np.linspace(-0.5, -1.5, 3),
        "clockwise_phi": False,
        "minor_radii": np.linspace(0, 2, 3),
    },
    5: {
        "b_toroidal": 2.5,
        "plasma_current": 1e6,
        "poloidal_flux": np.linspace(0, 2, 3),
        "safety_factor": np.linspace(-0.5, -1.5, 3),
        "clockwise_phi": False,
        "minor_radii": np.linspace(0, 2, 3),
    },
    7: {
        "b_toroidal": 2.5,
        "plasma_current": 1e6,
        "poloidal_flux": np.linspace(2, 0, 3),
        "safety_factor": np.linspace(0.5, 1.5, 3),
        "clockwise_phi": False,
        "minor_radii": np.linspace(0, 2, 3),
    },
}
identify_cocos_tests.update(odds)

# Set clockwise_phi to True in these to get COCOS 2, 4, 6, 8
evens = {}
for cocos, kwargs in odds.items():
    even_kwargs = kwargs.copy()
    even_kwargs["clockwise_phi"] = True
    evens[cocos + 1] = even_kwargs
identify_cocos_tests.update(evens)
# Multiply by factor of 2*pi to get COCOS 11 -> 18
tens = {}
for cocos, kwargs in identify_cocos_tests.items():
    tens_kwargs = kwargs.copy()
    # Note: can't use *= here, as some references are shared between tests
    tens_kwargs["poloidal_flux"] = kwargs["poloidal_flux"] * (2 * np.pi)
    tens_kwargs["safety_factor"] = kwargs["safety_factor"] * (2 * np.pi)
    tens[cocos + 10] = tens_kwargs
identify_cocos_tests.update(tens)


@pytest.mark.parametrize("expected_cocos,kwargs", [*identify_cocos_tests.items()])
def test_identify_cocos(expected_cocos, kwargs):
    assert identify_cocos(**kwargs) == (expected_cocos,)


def test_cocos_transform():
    # TODO should be parametrized
    assert Transform(1, 3).poloidal == -1
    for cocos in range(1, 9):
        assert Transform(cocos, cocos + 10).inv_psi != 1
        for cocos_add in (cocos + x * 10 for x in range(2)):
            for key in ('b_toroidal', 'toroidal', 'poloidal', 'q'):
                assert getattr(Transform(cocos_add, cocos_add), key) == 1
