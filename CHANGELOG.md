# Changelog

All notable changes to cosmopyro are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Development before 0.0.0 is not summarised here.

## [Unreleased]

### Added

- `check_skymap_values` (`cosmopyro.data`): sky maps must be finite, non-negative and
  free of exact zeros. Called from `run_data_checks`, so a run aborts before sampling if
  the sky map has empty cells: a posterior sample in such a cell gives log(0) = -inf and,
  despite the `repair` policy for the value, a NaN gradient that biases NUTS.

## [0.0.0] - 2026-XYZ

First tagged version, marking the start of explicit version tracking.

### Added


### Changed
