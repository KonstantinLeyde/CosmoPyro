# Changelog

All notable changes to cosmopyro are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Development before 0.0.0 is not summarised here.

## [Unreleased]

### Added

- `check_skymap_values` (`cosmopyro.data`), called from `run_data_checks`: sky maps must be
  finite and non-negative (errors); exact zeros give a warning.

### Changed

- Faster likelihood gradient (~1.75x per NUTS step for the gwcats LOS runs, 115 -> 65 ms on an
  RTX A6000; value and gradient unchanged to 3e-13):
  - `apply_batched_operation_1d` applies `jax.checkpoint` only if the data is split into more
    than one piece. With a single piece (the default `*_batch_size` of 5e7 exceeds the sample
    count) it saved no memory but recomputed the whole forward pass for the gradient.
  - The mass-redshift Jacobian gets dD_L/dz from the sample's own luminosity distance
    (D_C = D_L / (1 + z)) instead of a second spline lookup of D_C(z):
    `get_dluminosity_distance_gw_over_dz_from_redshift(..., luminosity_distance_gw=None)`
    (used by `FlatLambdaCDM`; accepted and ignored by the modified-GW-distance cosmologies).

### Fixed

- Grid log-densities (`SeparableConditional1D`, `InterpolatedConditional1D` with linear
  interpolation) are now gradient-safe at zero densities. A posterior sample in an empty
  sky-map cell gave log(0) = -inf; the `repair` policy fixed the value but the gradient was
  NaN, so NUTS rejected every trajectory where any sample hit such a cell, which silently
  biased the posterior (e.g. h ~ 1.08 with LOS sky maps at nside 64). Zero densities are now
  replaced by `epsilon` before the log, as out-of-bounds points already were.

## [0.0.0] - 2026-XYZ

First tagged version, marking the start of explicit version tracking.

### Added


### Changed
