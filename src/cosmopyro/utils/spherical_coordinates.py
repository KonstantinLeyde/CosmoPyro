import jax
import jax.numpy as jnp

from ..utils.jax_utils import (
    hist_v,
)
from . import healjax
from .class_utils import get_init_vars
from .jax_utils import compute_centers_and_delta_from_array

__all__ = [
    "HealPixDiscretization",
    "HealPixDiscretization3D",
    "compute_spherical_coordinates",
    "extract_valid_positions",
    "get_cartesian_coords_from_spherical_coords",
    "get_centered_array_for_ratio2",
    "get_normalized_coords",
    "get_radial_centers_and_boundaries",
    "get_spherical_coords_from_cartesian_coords",
    "repeat_array_in_3d",
]


def get_cartesian_coords_from_spherical_coords(r, theta, phi):
    x = r * jnp.sin(theta) * jnp.cos(phi)
    y = r * jnp.sin(theta) * jnp.sin(phi)
    z = r * jnp.cos(theta)

    return x, y, z


def get_spherical_coords_from_cartesian_coords(x, y, z):
    r = jnp.sqrt(x**2 + y**2 + z**2)
    theta = jnp.arccos(z / r)
    phi = jnp.sign(y) * jnp.arccos(x / jnp.sqrt(x**2 + y**2))

    return r, theta, phi


def compute_spherical_coordinates(r_array, theta_array, phi_array):
    """
    Compute Cartesian coordinates from spherical coordinates.

    Parameters:
        r_array (jnp.ndarray): Array of radial distances with shape (n_r,).
        theta_array (jnp.ndarray): Array of polar angles (0 <= theta <= pi) with shape (n_theta,).
        phi_array (jnp.ndarray): Array of azimuthal angles (0 <= phi < 2*pi) with shape (n_phi,).

    Returns:
        jnp.ndarray: Cartesian coordinates with shape (n_r, n_theta * n_phi, 3).
    """

    # Create meshgrid for theta and phi
    theta_grid, phi_grid = jnp.meshgrid(theta_array, phi_array, indexing="ij")

    # Flatten theta and phi grids
    theta_flat = theta_grid.ravel()
    phi_flat = phi_grid.ravel()

    # Compute Cartesian coordinates for unit sphere
    x_unit, y_unit, z_unit = get_cartesian_coords_from_spherical_coords(
        1.0, theta_flat, phi_flat
    )

    # Stack unit Cartesian coordinates
    unit_coords = jnp.stack(
        (x_unit, y_unit, z_unit), axis=-1
    )  # Shape: (n_theta * n_phi, 3)

    # Scale by r_array for all radial distances
    cartesian_coords = (
        r_array[:, None, None] * unit_coords[None, :, :]
    )  # Shape: (n_r, n_theta * n_phi, 3)

    return cartesian_coords


def get_radial_centers_and_boundaries(n_r, r_min, r_max):

    boundaries = jnp.linspace(r_min, r_max, n_r + 1)
    centers, deltas = compute_centers_and_delta_from_array(boundaries)

    return centers, boundaries, deltas


def get_normalized_coords(coords, r_max, n_x):
    return (
        (coords + jnp.array([r_max, r_max, r_max]))
        / jnp.array([r_max, r_max, r_max])
        * (n_x - 1)
        / 2
    )


class HealPixDiscretization:
    def __init__(self, nside, scheme="ring"):
        self.nside = nside
        self.scheme = scheme

        self.boundaries = {}
        self.boundaries["healpix_idx"] = jnp.arange(-1 / 2, self.npix + 1 / 2)
        self.centers, self.deltas = {}, {}
        self.centers["healpix_idx"], self.deltas["healpix_idx"] = (
            compute_centers_and_delta_from_array(self.boundaries["healpix_idx"])
        )

        # TODO this will take a while and maybe not always needed, add keyword initialize_jax
        # print('Warning healpix jax vec. disabled. ')
        def fun(theta, phi):
            return healjax.ang2pix(self.scheme, self.nside, theta, phi)

        self.ang2pixv = jax.vmap(fun, in_axes=(0, 0))
        self.ang2pixvv = jax.vmap(self.ang2pixv, in_axes=(0, 0))

    @property
    def kwargs_healpix(self):
        return {"nside": self.nside, "scheme": self.scheme}

    @property
    def npix(self):
        return self._npix

    @property
    def nside(self):
        return self._nside

    @nside.setter
    def nside(self, value):
        self._nside = value
        self._npix = healjax.nside2npix(value)

    @property
    def lmax(self):
        return 3 * self.nside - 1

    @property
    def degree_alm_real(self):
        l = self.lmax
        return (l + 1) * (l + 2) / 2

    @property
    def degree_alm_imag(self):
        l = self.lmax
        return (l + 1) * l / 2


class HealPixDiscretization3D(HealPixDiscretization):
    def __init__(self, n_r, r_min, r_max, nside, scheme="ring"):
        self.n_r = n_r
        self.r_min = r_min
        self.r_max = r_max

        super().__init__(nside, scheme)
        self.centers["r"], self.boundaries["r"], self.deltas["r"] = (
            get_radial_centers_and_boundaries(n_r, r_min, r_max)
        )

    def to_init_dict(self):
        return get_init_vars(self)

    @property
    def shape(self):
        return (self.n_r, self.npix)

    def histogram_healpix_3d_from_r_dec_ra(
        self, r, dec, ra, radial_key, weights_xyz=None
    ):

        theta = jnp.pi / 2 - dec
        phi = ra

        positions_in_hpidx = self.ang2pixv(theta, phi)

        positions_in_comoving_distance_hpidx = jnp.stack(
            [r, positions_in_hpidx], axis=-1
        )

        return self.histogram_healpix_3d(
            positions_in_comoving_distance_hpidx,
            radial_key=radial_key,
            weights_xyz=weights_xyz,
        )

    def histogram_healpix_3d(
        self, positions_in_comoving_distance_hpidx, radial_key, weights_xyz=None
    ):

        if weights_xyz is not None:
            weights = weights_xyz.flatten()
        else:
            weights = None

        counts, _ = jnp.histogramdd(
            positions_in_comoving_distance_hpidx,
            [self.boundaries[radial_key], self.boundaries["healpix_idx"]],
            weights=weights,
        )
        return counts

    def histogram_healpix_3d_v(self, positions_in_E_radial_coord_hpidx, radial_key):

        counts, _ = hist_v(
            positions_in_E_radial_coord_hpidx,
            [self.boundaries[radial_key], self.boundaries["healpix_idx"]],
        )
        return counts

    def ang2pixv(self, d, ra):
        return self.healpix_discretization.ang2pixv(d, ra)

    def add_comoving_coordinates(self, cosmological_model):
        self.boundaries["comoving_distance"] = (
            cosmological_model.get_comoving_distance_from_z(self.boundaries["r"])
        )
        self.centers["comoving_distance"], self.deltas["comoving_distance"] = (
            compute_centers_and_delta_from_array(self.boundaries["comoving_distance"])
        )

    def add_mock_comoving_distance(self):
        self.boundaries["comoving_distance"] = self.boundaries["r"]
        self.centers["comoving_distance"], self.deltas["comoving_distance"] = (
            compute_centers_and_delta_from_array(self.boundaries["comoving_distance"])
        )

    def get_volume_element(self, radial_key):
        """
        Compute the volume element for spherical coordinates.

        Returns:
            float: Volume element.
        """

        r = self.centers[radial_key]
        dr = self.deltas[radial_key]

        sky_portion = 4 * jnp.pi / self.npix

        return r[:, None] ** 2 * dr[:, None] * sky_portion

    def get_array_values_in_bin_idx(self, array, idx_r, radial_key):

        if len(array.shape) != 2 and array.shape[-1] != 2:
            # last axis should be (r, healpix)
            raise ValueError("Array must have shape (n_points, 2).")

        lower_bound = self.boundaries[radial_key][idx_r]
        upper_bound = self.boundaries[radial_key][idx_r + 1]

        idx = jnp.logical_and(array[:, 0] >= lower_bound, array[:, 0] < upper_bound)

        return array[idx], idx


def extract_valid_positions(coords, hpidx, mask):
    return jnp.stack([coords[mask, 0], hpidx[mask]], axis=1)


def get_centered_array_for_ratio2(array):
    slices = tuple(slice(s // 4, 3 * s // 4) for s in array.shape)
    return array[slices]


def repeat_array_in_3d(array, n):
    return jnp.repeat(jnp.repeat(jnp.repeat(array, n, axis=0), n, axis=1), n, axis=2)
