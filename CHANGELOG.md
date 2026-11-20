# Changelog

All notable changes to cosmopyro are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Development before 0.0.0 is not summarised here.

## [Unreleased]

### Added

- `check_skymap_values` (`cosmopyro.data`), called from `run_data_checks`: sky maps must be
  finite and non-negative (errors); exact zeros give a warning.
- `bins: {redshift: from_skymap}`: the redshift bins are the sky map's `z_edges`, which can
  be non-uniform (e.g. fine at low z, broad at high z). Requires a sky map; results read the
  edges back from `run_kwargs.path_skymap`. `HealPixDiscretization3D` accepts explicit
  `r_edges`.
- LVK low-mass turn-on (Talbot & Thrane 2018) for the parametric mass models,
  `log_smoothing_low_mass_lvk`, selected with the `_smoothing_lvk` model names (see Changed).
  S is floored at `LVK_SMOOTHING_FLOOR` = 1e-4 instead of being zero below `mmin`, so grid
  log-densities and gradients stay finite. (A floor of 1e-30 made the likelihood jagged in
  `mmin` for any mass-grid spacing, as the grid interpolation of log-densities cannot follow
  the cliff of log S just above `mmin`.)

### Changed

- **Breaking:** parametric mass models name their low-mass turn-on,
  `<model>_smoothing_gaussian` (the previous, approximate turn-on) or `<model>_smoothing_lvk`:
  `power_law_peak`, `power_law_peak2`, `power_law_peak2_partial_windowed`,
  `mass_ratio_running_power_law_in_log`, `mass_ratio_truncated_gaussian`. The `mass_1_s` and
  `mass_ratio` suffixes must match. Configs and results with the old names are read as
  `_smoothing_gaussian` with a warning (`rename_legacy_distribution_names`, applied when an
  `Analysis` / `Result` loads its settings). The model functions take a required keyword
  `smoothing="gaussian" | "lvk"`, and `log_lvk_smoothing_low_mass_approximation` is renamed to
  `log_smoothing_low_mass_gaussian`.
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
