r"""
Determine the COCOS (COordinate COnventionS) convention used to describe a tokamak
equilibrium and convert between COCOS systems. Based on the work of Sauter, O. and
Medvedev, S.Y., 2013. Tokamak coordinate conventions: COCOS. *Computer Physics
Communications*, 184(2), pp.293-302.

Throughout, we denote the coordinate systems in a tokamak with the following terms:

- :math:`R`, the major radius
- :math:`r`, the minor radius
- :math:`Z`, the vertical coordinate
- :math:`\phi`, the toroidal angle
- :math:`\theta`, the poloidal angle

These functions were adapted from OMAS (Copyright MIT License, 2017, Orso Meneghini).
"""

import itertools
from typing import Dict, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike


def sigma_to_cocos(
    sigma_bp: int,
    sigma_rpz: int,
    sigma_rtp: int,
    psi_by_2pi: bool = True,
) -> int:
    r"""
    We can (partially) determine the COCOS by checking the :math:`\sigma` quantities,
    defined below. We additionally need to know whether :math:`\psi` is defined with a
    factor of :math:`1/(2\pi)` to determine whether the COCOS is in the range 1 to 8 or
    11 to 18.

    Parameters
    ----------
    sigma_rtp
        :math:`\sigma_{B_p}`: Given by :math:`\text{sign}(\vec{B}_p`\cdot\nabla\theta)`,
        so is +1 when the poloidal magnetic field is in the same direction as increasing
        :math:`\theta` and -1 when they are opposed.
    sigma_rpz
        :math:`\sigma_{R \phi Z`}: +1 when :math:`(R, \phi`, Z)` form a right-handed
        coordinate system, and -1 when :math:`(R, Z, \phi)` form a right-handed
        coordinate system.
    sigma_bp
        :math:`\sigma_{r\theta\phi}`: +1 when :math:`(r, \theta, \phi)` form a
        right-handed coordinate system, and -1 when :math:`(r, \theta, \phi)` form a
        right-handed coordainte system.
    psi_by_2pi
        If true, :math:`\psi` is defined with a factor of :math:`1/(2\pi)`.

    Returns
    -------
    int
        COCOS convention in use.
    """
    if sigma_bp != 1 and sigma_bp != -1:
        raise ValueError(f"sigma_bp should be either 1 or -1, found {sigma_bp}")
    if sigma_rpz != 1 and sigma_rpz != -1:
        raise ValueError(f"sigma_rpz should be either 1 or -1, found {sigma_rpz}")
    if sigma_rtp != 1 and sigma_rtp != -1:
        raise ValueError(f"sigma_rtp should be either 1 or -1, found {sigma_rtp}")

    sigma_to_cocos_dict = {
        (+1, +1, +1): 1,  # +Bp, +rpz, +rtp
        (+1, -1, +1): 2,  # +Bp, -rpz, +rtp
        (-1, +1, -1): 3,  # -Bp, +rpz, -rtp
        (-1, -1, -1): 4,  # -Bp, -rpz, -rtp
        (+1, +1, -1): 5,  # +Bp, +rpz, -rtp
        (+1, -1, -1): 6,  # +Bp, -rpz, -rtp
        (-1, +1, +1): 7,  # -Bp, +rpz, +rtp
        (-1, -1, +1): 8,  # -Bp, -rpz, +rtp
    }
    result = sigma_to_cocos_dict[(sigma_bp, sigma_rpz, sigma_rtp)]
    return result if psi_by_2pi else result + 10


def identify_cocos(
    b_toroidal: float,
    plasma_current: float,
    safety_factor: ArrayLike,
    poloidal_flux: ArrayLike,
    clockwise_phi: Optional[bool] = None,
    minor_radii: Optional[ArrayLike] = None,
) -> Tuple[int, ...]:
    r"""
    Determine which COCOS coordinate system is in use. Returns all possible conventions.

    Parameters
    ----------
    b_toroidal
        Toroidal magnetic field, with sign. Should be in units of Tesla.
    plasma_current
        Plasma current, with sign. Should be in units of Amperes.
    safety_factor
        Safety factor profile, with sign, as a function of ``poloidal_flux``. Usually
        denoted :math:`q`. Should be defined with the first element on the magnetic
        axis, and subsequent elements on successively larger flux surfaces.
    poloidal_flux
        The profile of the poloidal flux function, with sign. Usually denoted
        :math:`\psi`. Should be defined with the first element on the magnetic
        axis, and subsequent elements on successively larger flux surfaces.
    clockwise_phi
        When viewing tokamak coordinates from above, does the toroidal angle
        :math:`\phi` increase in the clockwise direction? This is required to identify
        odd vs even COCOS. This cannot be determined from the output of a code alone.
        An easy way to determine this is to answer the question: is positive
        :math:`B_\text{toroidal}` clockwise?
    minor_radii
        Th minor radius of each flux surface as function of ``poloidal_flux``. This is
        required to identify whether :math:`\psi` contains a factor of :math:`2\pi`.

    Returns
    -------
    Tuple[int, ...]
        All possible conventions are returned. If both optional arguments are provided,
        the returned tuple should have length 1.

    """

    if clockwise_phi is None:
        return tuple(
            itertools.chain.from_iterable(
                identify_cocos(
                    b_toroidal,
                    plasma_current,
                    safety_factor,
                    poloidal_flux,
                    x,
                    minor_radii,
                )
                for x in (False, False)
            )
        )

    sign_plasma_current = np.sign(plasma_current)
    sign_b_toroidal = np.sign(b_toroidal)
    sign_q = np.sign(safety_factor[0])
    psi_increasing = np.sign(poloidal_flux[1] - poloidal_flux[0])

    sigma_bp = psi_increasing * sign_plasma_current
    sigma_rpz = -1 if clockwise_phi else 1
    sigma_rtp = sign_q * sign_plasma_current * sign_b_toroidal

    # identify 2*pi term in poloidal_flux definition based on safety_factor estimate
    if minor_radii is None:
        # Return both variants if not provided with minor radii
        return tuple(
            itertools.chain.from_iterable(
                sigma_to_cocos(sigma_bp, sigma_rpz, sigma_rtp, phi_by_2pi=x)
                for x in (True, False)
            )
        )

    index = np.argmin(np.abs(safety_factor))
    if index == 0:
        index += 1
    safety_factor_estimate = np.abs(
        (np.pi * b_toroidal * (minor_radii[index] - minor_radii[0]) ** 2)
        / (poloidal_flux[index] - poloidal_flux[0])
    )
    safety_factor_actual = np.abs(safety_factor[index])
    psi_by_2pi = np.abs(
        safety_factor_estimate / (2 * np.pi) - safety_factor_actual
    ) < np.abs(safety_factor_estimate - safety_factor_actual)

    return (sigma_to_cocos(sigma_bp, sigma_rpz, sigma_rtp, psi_by_2pi=psi_by_2pi),)


def cocos_coefficients(cocos: int) -> Dict[str, int]:
    r"""Returns dictionary with COCOS coefficients given a COCOS index"""
    # TODO Maybe have a pandas Dataframe with all the info written explicitly?
    coeffs = {
        "exp_bp": int(cocos >= 10),
        "sigma_bp": 1 if cocos in (1, 2, 5, 6, 11, 12, 15, 16) else -1,
        "sigma_rpz": 1 if cocos % 2 else -1,
        "sigma_rtp": 1 if cocos in (1, 2, 7, 8, 11, 12, 17, 18) else -1,
    }
    coeffs["phi_clockwise"] = coeffs["sigma_rpz"] == -1
    coeffs["theta_clockwise"] = cocos in (1, 4, 6, 7, 11, 14, 16, 17)
    coeffs["psi_increasing"] = bool(coeffs["exp_bp"])
    coeffs["sign_q"] = coeffs["sigma_rtp"]
    coeffs["sign_pprime"] = -coeffs["sigma_bp"]
    return coeffs


def cocos_transform(cocos_in: int, cocos_out: int) -> Dict[str, Union[int, float]]:
    r"""
    Returns a dictionary with coefficients for how various quantities should by
    multiplied in order to go from ``cocos_in`` to ``cocos_out``.
    """

    coeffs_in = cocos_coefficients(cocos_in)
    coeffs_out = cocos_coefficients(cocos_out)

    sigma_ip_eff = coeffs_in["sigma_rpz"] * coeffs_out["sigma_rpz"]
    sigma_b0_eff = coeffs_in["sigma_rpz"] * coeffs_out["sigma_rpz"]
    sigma_bp_eff = coeffs_in["sigma_bp"] * coeffs_out["sigma_bp"]
    exp_bp_eff = coeffs_out["exp_bp"] - coeffs_in["exp_bp"]
    sigma_rtp_eff = coeffs_in["sigma_rtp"] * coeffs_out["sigma_rtp"]

    # Transform
    transforms = {}
    transforms["1/psi"] = sigma_ip_eff * sigma_bp_eff / (2 * np.pi) ** exp_bp_eff
    transforms["d/dpsi"] = transforms["1/psi"]
    transforms["ffprime"] = transforms["1/psi"]
    transforms["pprime"] = transforms["1/psi"]

    transforms["toroidal"] = sigma_b0_eff
    transforms["b_toroidal"] = transforms["toroidal"]
    transforms["plasma_current"] = transforms["toroidal"]
    transforms["f"] = transforms["toroidal"]

    transforms["poloidal"] = sigma_b0_eff * sigma_rtp_eff
    transforms["b_poloidal"] = transforms["poloidal"]

    transforms["psi"] = sigma_ip_eff * sigma_bp_eff * (2 * np.pi) ** exp_bp_eff
    transforms["q"] = sigma_ip_eff * sigma_b0_eff * sigma_rtp_eff
    return transforms
