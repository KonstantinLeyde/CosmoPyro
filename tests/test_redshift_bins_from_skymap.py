"""Non-uniform redshift bins taken from the sky map: ``bins: {redshift: from_skymap}``."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from cosmopyro.data.build_skymap import validate_skymap
from cosmopyro.data.data_utils import save_namespace_to_hdf5
from cosmopyro.models import models
from cosmopyro.utils import analyses
from cosmopyro.utils.spherical_coordinates import HealPixDiscretization3D

EXAMPLE_CONFIG = Path(__file__).parents[1] / "examples" / "configs" / "powerlaw_2peaks_gwtc5.yaml"
NSIDE = 2
# piecewise uniform: fine at low z, coarse above 0.5
Z_EDGES = np.concatenate([np.linspace(0.0, 0.5, 51), np.linspace(0.5, 5.0, 46)[1:]])


def _write_skymap(path, z_edges=Z_EDGES, nside=NSIDE, with_z_edges=True):
    npix = 12 * nside**2
    rng = np.random.default_rng(0)
    prob = rng.uniform(0.5, 1.5, size=(z_edges.size - 1, npix))
    prob /= prob.sum(axis=1, keepdims=True)
    ns = SimpleNamespace(prob_skyposition_zhp=prob, nside=nside)
    if with_z_edges:
        ns.z_edges = z_edges
    save_namespace_to_hdf5(ns, str(path))
    return ns


def _config(tmp_path, redshift_bins="from_skymap"):
    cfg = yaml.safe_load(open(EXAMPLE_CONFIG))
    cfg["catalog_metadata"] = {"healpix_discretization_3d": {"nside": NSIDE}}
    cfg["bins"]["redshift"] = redshift_bins
    cfg["cosmology_numerics"]["z_max"] = 6.0
    cfg["results_path"] = str(tmp_path / "results") + "/"
    path = tmp_path / "config.yaml"
    yaml.safe_dump(cfg, open(path, "w"))
    return path


def test_discretization_with_explicit_edges():
    d = HealPixDiscretization3D(nside=NSIDE, r_edges=Z_EDGES)
    assert d.n_r == Z_EDGES.size - 1 and d.r_min == 0.0 and d.r_max == 5.0
    np.testing.assert_array_equal(np.asarray(d.boundaries["r"]), Z_EDGES)
    np.testing.assert_allclose(np.asarray(d.deltas["r"]), np.diff(Z_EDGES))
    np.testing.assert_allclose(np.asarray(d.centers["r"]), 0.5 * (Z_EDGES[1:] + Z_EDGES[:-1]))
    assert d.shape == (Z_EDGES.size - 1, 12 * NSIDE**2)


def test_discretization_rejects_bad_edges():
    with pytest.raises(ValueError, match="strictly increasing"):
        HealPixDiscretization3D(nside=NSIDE, r_edges=[0.0, 0.2, 0.1])
    with pytest.raises(ValueError, match="either"):
        HealPixDiscretization3D(nside=NSIDE)


def test_redshift_discretization_kwargs(tmp_path):
    path = tmp_path / "skymap.hdf5"
    _write_skymap(path)
    uniform = {"bins": {"redshift": {"min": 0.0, "max": 5.0, "num": 10}}}
    assert analyses.get_redshift_discretization_kwargs(uniform) == dict(r_min=0.0, r_max=5.0, n_r=10)

    from_skymap = {"bins": {"redshift": "from_skymap"}}
    assert analyses.get_redshift_discretization_kwargs(from_skymap) is None  # resolved when the map is loaded
    with pytest.raises(ValueError, match="needs a sky map"):
        analyses.pick_out_add_healpix_discretization_3d_kwargs(from_skymap)

    kwargs = analyses.get_redshift_discretization_kwargs({**from_skymap, "run_kwargs": {"path_skymap": str(path)}})
    np.testing.assert_array_equal(kwargs["r_edges"], Z_EDGES)


def test_read_skymap_z_edges_missing(tmp_path):
    path = tmp_path / "skymap_no_edges.hdf5"
    _write_skymap(path, with_z_edges=False)
    with pytest.raises(ValueError, match="no z_edges"):
        analyses.read_skymap_z_edges(str(path))


def test_analysis_takes_redshift_bins_from_skymap(tmp_path):
    path = tmp_path / "skymap.hdf5"
    skymap_written = _write_skymap(path)
    a = analyses.Analysis(model=models.model_evaluate_p_theta, settings_path=str(_config(tmp_path)), job_id="test")
    assert a.discretization_3d is None  # no sky map known yet

    skymap = a._load_skymap(str(path))
    np.testing.assert_array_equal(np.asarray(a.binning["boundaries"]["redshift"]), Z_EDGES)
    np.testing.assert_allclose(np.asarray(a.binning["deltas"]["redshift"]), np.diff(Z_EDGES))
    validate_skymap(skymap, a.binning, NSIDE)
    np.testing.assert_allclose(np.asarray(skymap.prob_skyposition_zhp), skymap_written.prob_skyposition_zhp)

    # a later Analysis / Result for the same config reads the edges from run_kwargs.path_skymap
    a.update_kwargs_analysis(["run_kwargs"], {"path_skymap": str(path)})
    cfg_with_path = tmp_path / "config_with_path.yaml"
    yaml.safe_dump(a.kwargs_analysis, open(cfg_with_path, "w"))
    b = analyses.Analysis(model=models.model_evaluate_p_theta, settings_path=str(cfg_with_path), job_id="test2")
    np.testing.assert_array_equal(np.asarray(b.binning["boundaries"]["redshift"]), Z_EDGES)


def test_from_skymap_without_skymap_raises(tmp_path):
    a = analyses.Analysis(model=models.model_evaluate_p_theta, settings_path=str(_config(tmp_path)), job_id="test")
    with pytest.raises(ValueError, match="needs a sky map"):
        a._load_skymap(None)
