"""Parameter algebra: ordering, frame conventions, pseudo-inertia, priors."""

from __future__ import annotations

import numpy as np
import pinocchio as pin
import pytest

from franka_payload_id.model import params as P


def test_ordering_matches_pinocchio():
    """phi layout must be [m, m*c, Ixx, Ixy, Iyy, Ixz, Iyz, Izz], inertia about ORIGIN.

    This is the trap the whole pipeline is built around: Pinocchio uses the
    column-major lower-triangular order xx, xy, yy, xz, yz, zz -- not the common
    xx, xy, xz, yy, yz, zz. Swapping them silently exchanges Iyy with Ixz.
    """
    mass = 2.0
    com = np.array([0.1, 0.2, 0.3])
    inertia_com = np.array([[0.11, 0.012, 0.013],
                            [0.012, 0.22, 0.023],
                            [0.013, 0.023, 0.33]])

    ours = P.phi_from_mci(mass, com, inertia_com)
    theirs = pin.Inertia(mass, com, inertia_com).toDynamicParameters()
    np.testing.assert_allclose(ours, theirs, rtol=0, atol=1e-14)

    ibar = P.inertia_about_origin(inertia_com, mass, com)
    expected = [ibar[0, 0], ibar[0, 1], ibar[1, 1], ibar[0, 2], ibar[1, 2], ibar[2, 2]]
    np.testing.assert_allclose(ours[4:], expected, atol=1e-14)

    # And the six entries are NOT the CoM inertia.
    assert not np.allclose(ours[4:], P.vec6_from_inertia_matrix(inertia_com))


def test_phi_roundtrip(tool_phi):
    mass, com, inertia_com = P.phi_to_mci(tool_phi)
    np.testing.assert_allclose(P.phi_from_mci(mass, com, inertia_com), tool_phi, atol=1e-15)

    ip = P.InertialParams.from_phi(tool_phi)
    np.testing.assert_allclose(ip.to_phi(), tool_phi, atol=1e-15)
    # InertialParams.inertia_com is what Desk wants; it must round-trip too.
    np.testing.assert_allclose(
        P.inertia_about_com(ip.inertia_origin, ip.mass, ip.com), ip.inertia_com, atol=1e-15)


def test_phi_to_mci_rejects_nonpositive_mass():
    bad = np.zeros(10)
    with pytest.raises(ValueError, match="non-positive mass"):
        P.phi_to_mci(bad)


def test_parallel_axis_inverse_pair(rng):
    for _ in range(20):
        mass = float(rng.uniform(0.1, 3.0))
        com = rng.uniform(-0.2, 0.2, 3)
        ic = np.diag(rng.uniform(1e-4, 1e-2, 3))
        back = P.inertia_about_com(P.inertia_about_origin(ic, mass, com), mass, com)
        np.testing.assert_allclose(back, ic, atol=1e-15)


def test_translate_phi_matches_pinocchio_se3_action(rng, tool_phi):
    """A pure-translation frame change must agree with Pinocchio's SE3 action."""
    t = np.array([0.0, 0.0, 0.107])
    moved = P.translate_phi(tool_phi, t)

    inertia = pin.Inertia.FromDynamicParameters(tool_phi)
    placement = pin.SE3(np.eye(3), t)
    np.testing.assert_allclose(moved, placement.act(inertia).toDynamicParameters(), atol=1e-14)

    # Translating there and back is the identity.
    np.testing.assert_allclose(P.translate_phi(moved, -t), tool_phi, atol=1e-14)


def test_translation_map_is_linear_and_consistent(rng, tool_phi):
    t = rng.uniform(-0.3, 0.3, 3)
    mat = P.translation_map(t)
    np.testing.assert_allclose(mat @ tool_phi, P.translate_phi(tool_phi, t), atol=1e-14)
    # Linearity: valid even for a zero-mass basis vector, which is why the map exists.
    for k in range(P.N_PARAMS):
        e = np.eye(P.N_PARAMS)[k]
        np.testing.assert_allclose(mat @ e, P.translate_phi(e, t), atol=1e-14)


def test_pseudo_inertia_definition(tool_phi):
    """J(phi) must equal the second-moment matrix of the mass distribution."""
    j = P.pseudo_inertia(tool_phi)
    mass, com, _ = P.phi_to_mci(tool_phi)
    ibar = P.inertia_matrix_from_vec6(tool_phi[4:])
    sigma = 0.5 * np.trace(ibar) * np.eye(3) - ibar
    np.testing.assert_allclose(j[:3, :3], sigma, atol=1e-15)
    np.testing.assert_allclose(j[:3, 3], mass * com, atol=1e-15)
    assert j[3, 3] == pytest.approx(mass)
    np.testing.assert_allclose(j, j.T, atol=1e-15)


def test_pseudo_inertia_basis_reconstructs(tool_phi):
    basis = P.pseudo_inertia_basis()
    assert basis.shape == (10, 4, 4)
    np.testing.assert_allclose(np.einsum("k,kij->ij", tool_phi, basis),
                               P.pseudo_inertia(tool_phi), atol=1e-15)


def test_physical_consistency_detects_violations(tool_phi):
    assert P.is_physically_consistent(tool_phi)

    negative_mass = tool_phi.copy()
    negative_mass[0] = -1.0
    assert not P.is_physically_consistent(negative_mass)

    # Violate the triangle inequality: make Izz far larger than Ixx + Iyy.
    mass, com, ic = P.phi_to_mci(tool_phi)
    bad = P.phi_from_mci(mass, com, np.diag([1e-4, 1e-4, 1.0]))
    assert not P.is_physically_consistent(bad)
    report = P.consistency_report(bad)
    assert report["inertia_com_pd"] is True         # still PD ...
    assert report["triangle_inequality"] is False   # ... but not realizable
    assert report["physically_consistent"] is False


def test_bounding_box_prior_is_consistent():
    prior = P.bounding_box_prior(0.5, np.array([-0.05, -0.05, 0.0]),
                                 np.array([0.05, 0.05, 0.16]))
    assert prior.mass == pytest.approx(0.5)
    np.testing.assert_allclose(prior.com, [0.0, 0.0, 0.08])
    # Solid box about its own centre: Ixx = m/12 (b^2 + c^2)
    assert prior.inertia_com[0, 0] == pytest.approx(0.5 / 12.0 * (0.1**2 + 0.16**2))
    assert P.is_physically_consistent(prior.to_phi())


def test_bounding_ellipsoid_constraint_holds_for_contained_mass():
    """tr(J Q) >= 0 for any body whose mass sits inside the bounding ellipsoid."""
    lo, hi = np.array([-0.05, -0.05, 0.0]), np.array([0.05, 0.05, 0.16])
    q_mat = P.bounding_ellipsoid_matrix(lo, hi)
    prior = P.bounding_box_prior(0.5, lo, hi)
    assert np.trace(P.pseudo_inertia(prior.to_phi()) @ q_mat) >= -1e-12

    # A point mass far outside the box must violate it.
    outside = P.phi_from_mci(0.5, np.array([0.0, 0.0, 1.0]), np.eye(3) * 1e-9)
    assert np.trace(P.pseudo_inertia(outside) @ q_mat) < 0.0


def test_scaling_matrix_non_dimensionalises():
    d = P.scaling_matrix(0.1)
    np.testing.assert_allclose(np.diag(d), [1, .1, .1, .1, .01, .01, .01, .01, .01, .01])
    with pytest.raises(ValueError):
        P.scaling_matrix(0.0)


def test_desk_fields_are_column_major_about_com(tool_phi):
    fields = P.InertialParams.from_phi(tool_phi).as_desk_fields()
    mat = np.asarray(fields["inertia_matrix"])
    np.testing.assert_allclose(np.asarray(fields["inertia_column_major"]),
                               mat.flatten(order="F"), atol=1e-15)
    _, _, ic = P.phi_to_mci(tool_phi)
    np.testing.assert_allclose(mat, ic, atol=1e-15)
