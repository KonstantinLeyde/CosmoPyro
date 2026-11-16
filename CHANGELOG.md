# Changelog

All notable changes to cosmopyro are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Development before 0.0.0 is not summarised here.

## [Unreleased]

### Added

- `check_skymap_values` (`cosmopyro.data`), called from `run_data_checks`: sky maps must be
  finite and non-negative (errors); exact zeros give a warning.

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
