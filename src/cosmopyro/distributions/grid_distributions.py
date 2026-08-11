import jax.numpy as jnp
import numpy as np

__all__ = [
    "InterpolatedConditional1D",
    "SeparableConditional1D",
    "normalize_cond_interpolated_1d",
    "normalize_probs_over_axes",
    "piecewise_linear_quadrature_weights",
]


def _val_edges(name: str, arr):
    arr = jnp.asarray(arr)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError(f"'{name}' boundaries must be 1-D with ≥2 endpoints.")
    return arr


def _uniform_spacing(points) -> float | None:
    """Concrete spacing of a uniformly spaced 1-D array, else None.

    Returns None for traced arrays, so uniformity can only be exploited
    when the grid is known at construction time.
    """
    try:
        pts = np.asarray(points)
    except Exception:
        return None
    if pts.ndim != 1 or pts.size < 2:
        return None
    d = np.diff(pts)
    if not np.all(d > 0):
        return None
    if np.allclose(d, d[0], rtol=1e-10, atol=0.0):
        return float(d[0])
    return None


def _locate_on_centers(centers, vals, spacing):
    """Bracketing indices (i0, i1) and weight t for linear interpolation.

    Uses direct arithmetic when ``spacing`` is known (uniform grid),
    otherwise a binary search. t is clipped to [0, 1] so values beyond the
    outermost centers are held constant.
    """
    n = centers.shape[0]
    if n == 1:
        i0 = jnp.zeros(vals.shape, dtype=jnp.int32)
        return i0, i0, jnp.zeros(vals.shape, dtype=centers.dtype)
    if spacing is not None:
        u = (vals - centers[0]) / spacing
        i0 = jnp.clip(jnp.floor(u).astype(jnp.int32), 0, n - 2)
        t = jnp.clip(u - i0, 0.0, 1.0)
    else:
        i0 = jnp.clip(jnp.searchsorted(centers, vals, side="right") - 1, 0, n - 2)
        c0 = centers[i0]
        c1 = centers[i0 + 1]
        denom = jnp.where(c1 > c0, c1 - c0, jnp.ones_like(c1))
        t = jnp.clip((vals - c0) / denom, 0.0, 1.0)
    return i0, i0 + 1, t


def _bin_index(edges, vals, spacing):
    """Bin index of ``vals`` in ``edges``, clipped to valid bins."""
    if spacing is not None:
        idx = jnp.floor((vals - edges[0]) / spacing).astype(jnp.int32)
    else:
        idx = jnp.searchsorted(edges, vals, side="right") - 1
    return jnp.clip(idx, 0, edges.shape[0] - 2)


def _safe_log_density(dens, oob, epsilon):
    """Log-density with gradient-safe OOB handling."""
    dens_clean = jnp.where(oob, jnp.zeros_like(dens), dens)
    dens_safe = jnp.where(oob, jnp.asarray(epsilon, dtype=dens.dtype), dens_clean)
    return jnp.log(dens_safe)


def normalize_probs_over_axes(base, vol, axes):
    Z = jnp.sum(base * vol, axis=axes)
    cond = base / Z
    return cond


class InterpolatedConditional1D:
    """1D approximation of P(x | y) with interpolation over x.

    x_bins must have exactly one key. y_bins keys are sorted alphabetically.
    ``cond`` shape must be ``(n_x, *y_counts_sorted)``.
    """

    def __init__(
        self,
        x_bins: dict[str, jnp.ndarray],
        y_bins: dict[str, jnp.ndarray] | None,
        cond: jnp.ndarray,
        continuous_y_names: list[str] | None = None,
        tol: float = 1e-6,
        epsilon: float = 1e-12,
        interpolation: str = "linear_density",
    ):
        self.epsilon = float(epsilon)
        valid_interpolations = {"linear", "linear_density", "smooth_log"}
        if interpolation not in valid_interpolations:
            raise ValueError(
                f"Unknown interpolation='{interpolation}'. "
                f"Expected one of {sorted(valid_interpolations)}."
            )
        self.interpolation = (
            "linear_density" if interpolation == "linear" else interpolation
        )

        # --- X (single variable) ---
        self.x_names = list(x_bins.keys())
        if len(self.x_names) != 1:
            raise ValueError(
                "InterpolatedConditional1D requires exactly one x variable."
            )
        self.x_var = self.x_names[0]

        self.x_edges = _val_edges(self.x_var, x_bins[self.x_var])
        self.x_widths = jnp.diff(self.x_edges)
        self.x_centers = 0.5 * (self.x_edges[:-1] + self.x_edges[1:])
        self.x_edge_spacing = _uniform_spacing(self.x_edges)
        self.x_center_spacing = _uniform_spacing(self.x_centers)

        # --- Y (conditioning, sorted alphabetically) ---
        self.y_names: list[str] = sorted(y_bins.keys()) if y_bins else []
        self.y_edges: dict[str, jnp.ndarray] = {}
        self.y_centers: dict[str, jnp.ndarray] = {}
        self.y_edge_spacing: dict[str, float | None] = {}
        self.y_center_spacing: dict[str, float | None] = {}
        self.continuous_y_names = set(continuous_y_names or [])
        invalid_continuous_y = self.continuous_y_names - set(self.y_names)
        if invalid_continuous_y:
            raise ValueError(
                f"continuous_y_names contains unknown variables: {sorted(invalid_continuous_y)}. "
                f"Known y variables: {self.y_names}"
            )
        if y_bins:
            for k in self.y_names:
                self.y_edges[k] = _val_edges(k, y_bins[k])
                self.y_centers[k] = 0.5 * (self.y_edges[k][:-1] + self.y_edges[k][1:])
                self.y_edge_spacing[k] = _uniform_spacing(self.y_edges[k])
                self.y_center_spacing[k] = _uniform_spacing(self.y_centers[k])

        # --- Shape check ---
        nx = self.x_centers.shape[0]
        ny_counts = [self.y_edges[k].shape[0] - 1 for k in self.y_names]
        expected_shape = tuple([nx] + ny_counts)
        self.cond = jnp.asarray(cond)
        if self.cond.shape != expected_shape:
            raise ValueError(
                f"cond shape mismatch. Expected {expected_shape} "
                f"(x='{self.x_var}', y keys sorted: {self.y_names}), got {self.cond.shape}"
            )
        # Only smooth_log interpolates in log space; skip the full-grid log
        # otherwise (the grid can be large and is rebuilt per likelihood call).
        if self.interpolation == "smooth_log":
            log_floor = jnp.asarray(
                jnp.finfo(self.cond.dtype).tiny, dtype=self.cond.dtype
            )
            self.log_cond = jnp.log(jnp.maximum(self.cond, log_floor))
        else:
            self.log_cond = None

    def _interpolation_t(self, t):
        if self.interpolation == "smooth_log":
            return t * t * (3.0 - 2.0 * t)
        return t

    def _resolve_y(self, y_vals):
        """Resolve y conditioning interpolation info and OOB masks."""
        y_vals = y_vals or {}
        y_infos = []
        for k in self.y_names:
            val = jnp.asarray(y_vals[k])
            edges = self.y_edges[k]
            oob = (val < edges[0]) | (val >= edges[-1])

            if k in self.continuous_y_names:
                lower, upper, t = _locate_on_centers(
                    self.y_centers[k], val, self.y_center_spacing[k]
                )
                y_infos.append(
                    dict(
                        continuous=True,
                        lower=lower,
                        upper=upper,
                        t=t,
                        oob=oob,
                    )
                )
            else:
                idx = _bin_index(edges, val, self.y_edge_spacing[k])
                y_infos.append(
                    dict(
                        continuous=False,
                        index=idx,
                        oob=oob,
                    )
                )
        return y_infos

    def _interpolate(self, x_vals, y_vals=None, log_space=False):
        """Linear interpolation over x and selected continuous y dimensions."""
        grid = self.log_cond if log_space else self.cond
        y_infos = self._resolve_y(y_vals)

        x_arr = jnp.asarray(x_vals[self.x_var])
        x_oob = (x_arr < self.x_edges[0]) | (x_arr >= self.x_edges[-1])

        shapes = [x_arr.shape]
        for info in y_infos:
            shapes.append(
                (info.get("lower") if info["continuous"] else info["index"]).shape
            )
        bshape = jnp.broadcast_shapes(*shapes) if shapes else ()

        x_flat = jnp.broadcast_to(x_arr, bshape).reshape(-1)
        i0, i1, tx = _locate_on_centers(self.x_centers, x_flat, self.x_center_spacing)
        wx = self._interpolation_t(tx)

        y_infos_flat = []
        for info in y_infos:
            if info["continuous"]:
                y_infos_flat.append(
                    dict(
                        continuous=True,
                        lower=jnp.broadcast_to(info["lower"], bshape).reshape(-1),
                        upper=jnp.broadcast_to(info["upper"], bshape).reshape(-1),
                        t=self._interpolation_t(
                            jnp.broadcast_to(info["t"], bshape).reshape(-1)
                        ),
                        oob=jnp.broadcast_to(info["oob"], bshape).reshape(-1),
                    )
                )
            else:
                y_infos_flat.append(
                    dict(
                        continuous=False,
                        index=jnp.broadcast_to(info["index"], bshape).reshape(-1),
                        oob=jnp.broadcast_to(info["oob"], bshape).reshape(-1),
                    )
                )

        corners = [
            ([i0], 1.0 - wx),
            ([i1], wx),
        ]
        for info in y_infos_flat:
            next_corners = []
            if info["continuous"]:
                for idxs, weight in corners:
                    next_corners.append(
                        (idxs + [info["lower"]], weight * (1.0 - info["t"]))
                    )
                    next_corners.append((idxs + [info["upper"]], weight * info["t"]))
            else:
                for idxs, weight in corners:
                    next_corners.append((idxs + [info["index"]], weight))
            corners = next_corners

        dens_flat = jnp.zeros_like(x_flat, dtype=grid.dtype)
        for idxs, weight in corners:
            dens_flat = dens_flat + weight * grid[tuple(idxs)]

        dens = dens_flat.reshape(bshape)

        # Combined OOB mask
        any_oob = jnp.broadcast_to(x_oob, bshape) if bshape else x_oob
        for info in y_infos:
            yo = info["oob"]
            any_oob = any_oob | (jnp.broadcast_to(yo, bshape) if bshape else yo)

        return dens, any_oob, bshape

    def _safe_log_density(self, dens, oob):
        return _safe_log_density(dens, oob, self.epsilon)

    def log_prob(self, x_vals, y_vals=None):
        if self.interpolation == "smooth_log":
            log_dens, oob, _ = self._interpolate(
                x_vals,
                y_vals,
                log_space=True,
            )
            log_epsilon = jnp.log(jnp.asarray(self.epsilon, dtype=log_dens.dtype))
            return jnp.where(oob, log_epsilon, log_dens)
        dens, oob, _ = self._interpolate(x_vals, y_vals)
        return self._safe_log_density(dens, oob)

    def log_mass(self, x_vals, y_vals=None):
        x_arr = jnp.asarray(x_vals[self.x_var])
        x_idx = _bin_index(self.x_edges, x_arr, self.x_edge_spacing)
        if self.interpolation == "smooth_log":
            log_dens, oob, bshape = self._interpolate(
                x_vals,
                y_vals,
                log_space=True,
            )
            width_vals = self.x_widths[x_idx]
            if bshape:
                width_vals = jnp.broadcast_to(width_vals, bshape)
            log_mass = log_dens + jnp.log(width_vals)
            log_epsilon = jnp.log(jnp.asarray(self.epsilon, dtype=log_dens.dtype))
            return jnp.where(oob, log_epsilon, log_mass)
        dens, oob, bshape = self._interpolate(x_vals, y_vals)
        width_vals = self.x_widths[x_idx]
        if bshape:
            width_vals = jnp.broadcast_to(width_vals, bshape)
        mass = dens * width_vals
        mass_safe = jnp.where(oob, self.epsilon, mass)
        return jnp.log(mass_safe)


class SeparableConditional1D:
    """P(x | y) ∝ prob_x(x) · grid(x, y) for a single discrete y variable.

    ``grid`` is a fixed ``(nx, ny)`` array of values at the x-bin centers
    (e.g. a skymap that does not depend on the sampled parameters). The full
    product grid is never materialized: the per-y normalization reduces to a
    single matrix-vector product ``Z = (q * prob_x) @ grid`` with ``q`` the
    quadrature weights of ``normalize_cond_interpolated_1d``.

    At the x-bin centers ``log_prob`` matches ``InterpolatedConditional1D``
    built on the normalized product grid. Between centers the interpolant is
    the product of the two linear interpolants (piecewise quadratic in x)
    rather than the linear interpolant of the product.

    ``grid=None`` means grid ≡ 1 (y-independent), in which case
    P(x | y) = P(x) and no ``(nx, ny)`` array is ever allocated.
    """

    def __init__(
        self,
        x_bins: dict[str, jnp.ndarray],
        y_bins: dict[str, jnp.ndarray],
        prob_x: jnp.ndarray,
        grid: jnp.ndarray | None = None,
        epsilon: float = 1e-12,
    ):
        self.epsilon = float(epsilon)

        self.x_names = list(x_bins.keys())
        if len(self.x_names) != 1:
            raise ValueError("SeparableConditional1D requires exactly one x variable.")
        self.x_var = self.x_names[0]
        self.x_edges = _val_edges(self.x_var, x_bins[self.x_var])
        self.x_centers = 0.5 * (self.x_edges[:-1] + self.x_edges[1:])
        self.x_center_spacing = _uniform_spacing(self.x_centers)

        self.y_names = list(y_bins.keys())
        if len(self.y_names) != 1:
            raise ValueError("SeparableConditional1D requires exactly one y variable.")
        self.y_var = self.y_names[0]
        self.y_edges = _val_edges(self.y_var, y_bins[self.y_var])
        self.y_edge_spacing = _uniform_spacing(self.y_edges)

        nx = self.x_centers.shape[0]
        ny = self.y_edges.shape[0] - 1
        self.prob_x = jnp.asarray(prob_x)
        if self.prob_x.shape != (nx,):
            raise ValueError(
                f"prob_x shape mismatch. Expected ({nx},), got {self.prob_x.shape}"
            )

        weighted = piecewise_linear_quadrature_weights(self.x_edges) * self.prob_x
        if grid is None:
            self.grid = None
            Z = jnp.sum(weighted)
        else:
            self.grid = jnp.asarray(grid)
            if self.grid.shape != (nx, ny):
                raise ValueError(
                    f"grid shape mismatch. Expected ({nx}, {ny}), got {self.grid.shape}"
                )
            Z = weighted @ self.grid
        floor = jnp.asarray(1e-30, dtype=Z.dtype)
        self.log_Z = jnp.log(jnp.maximum(Z, floor))

    def log_prob(self, x_vals, y_vals):
        x_arr = jnp.asarray(x_vals[self.x_var])
        y_arr = jnp.asarray(y_vals[self.y_var])
        bshape = jnp.broadcast_shapes(x_arr.shape, y_arr.shape)
        x = jnp.broadcast_to(x_arr, bshape)
        y = jnp.broadcast_to(y_arr, bshape)

        oob = (
            (x < self.x_edges[0])
            | (x >= self.x_edges[-1])
            | (y < self.y_edges[0])
            | (y >= self.y_edges[-1])
        )

        i0, i1, t = _locate_on_centers(self.x_centers, x, self.x_center_spacing)
        dens = (1.0 - t) * self.prob_x[i0] + t * self.prob_x[i1]

        if self.grid is None:
            log_Z = self.log_Z
        else:
            y_idx = _bin_index(self.y_edges, y, self.y_edge_spacing)
            s = (1.0 - t) * self.grid[i0, y_idx] + t * self.grid[i1, y_idx]
            dens = dens * s
            log_Z = self.log_Z[y_idx]

        log_dens = _safe_log_density(dens, oob, self.epsilon)
        return log_dens - jnp.where(oob, jnp.zeros_like(log_Z), log_Z)


def piecewise_linear_quadrature_weights(x_edges: jnp.ndarray) -> jnp.ndarray:
    """Quadrature weights q such that ``q @ cond`` is the integral over x.

    Uses the piecewise model of ``InterpolatedConditional1D`` (constant
    beyond the outermost centers, linear between centers), i.e. the same
    mass computed by ``normalize_cond_interpolated_1d``.
    """
    widths = jnp.diff(jnp.asarray(x_edges))
    nx = widths.shape[0]
    if nx == 0:
        raise ValueError("x_edges must have at least two points.")
    q = jnp.zeros(nx, dtype=widths.dtype)
    if nx >= 2:
        L = 0.5 * (widths[:-1] + widths[1:])
        q = q.at[:-1].add(0.5 * L)
        q = q.at[1:].add(0.5 * L)
    q = q.at[0].add(0.5 * widths[0])
    q = q.at[-1].add(0.5 * widths[-1])
    return q


def normalize_cond_interpolated_1d(
    x_edges: jnp.ndarray, cond: jnp.ndarray
) -> jnp.ndarray:
    """
    Normalize ``cond`` so the integral over x is exactly 1 for each y-slice.

    Assumes the piecewise model used by ``InterpolatedConditional1D``:

    - constant from ``x_edges[0]`` to the first center
    - linear between consecutive centers
    - constant from the last center to ``x_edges[-1]``

    Parameters
    ----------
    x_edges : (nx+1,) array
        Bin boundaries for x (strictly increasing).
    cond : ``(nx, *y_counts)``
        Density values at x-bin centers (one per bin). May include
        additional y axes.

    Returns
    -------
    cond_norm : same shape as cond
        Renormalized densities.
    """
    cond = jnp.asarray(cond)
    q = piecewise_linear_quadrature_weights(x_edges)
    nx = q.shape[0]
    if cond.shape[0] != nx:
        raise ValueError(
            f"cond.shape[0] ({cond.shape[0]}) must equal len(x_edges)-1 ({nx})."
        )

    mass = jnp.tensordot(q, cond, axes=(0, 0))

    floor = jnp.asarray(1e-30, dtype=mass.dtype)
    safe_mass = jnp.maximum(mass, floor)
    cond_norm = cond / safe_mass.reshape((1,) + mass.shape)

    return cond_norm
