import warnings

import corner
import healpy as hp
import jax.numpy as jnp
import matplotlib.pyplot as plt

from .utils import construct_array_from_df

__all__ = [
    "CORNER_KWARGS_DEFAULT",
    "PARAMETER_LABELS",
    "clean_log_pdfs",
    "draw_2d_posterior_with_marginals",
    "draw_frame",
    "extent_from_centers",
    "find_shared_params_from_dfs",
    "get_mollweide_ra_from_ra",
    "make_2d_mass_predictive_plot",
    "make_luminosity_distance_ratio_plot",
    "make_mass_plot",
    "make_mass_plot_row",
    "make_mass_plot_s_delta",
    "make_percentile_plot",
    "plot_marginal_m1",
    "plot_mollview_with_healpix",
    "plot_mollweide_healpix",
    "plot_posterior_comparison",
    "plot_trace_scalar_variables",
    "setup_layout",
]


CORNER_KWARGS_DEFAULT = dict(
    plot_datapoints=False,
    hist_kwargs=dict(density=True, linewidth=2),
    plot_contours=True,
    no_fill_contours=True,
    plot_density=False,
    smooth=0.05,
    levels=[0.5, 0.9],
    contour_kwargs={"linewidths": 2.5},
)

M_unit = r" $[M_\odot]$"
PARAMETER_LABELS = {
    "h": r"$h$",
    "H0": r"$H_0$",
    "Omega_m": r"$\Omega_{\rm m}$",
    "mmax": r"$m_{\rm max}$" + M_unit,
    "mmin": r"$m_{\rm min}$" + M_unit,
    "alpha": r"$\alpha$",
    "beta": r"$\beta$",
    "sigma_g_low": r"$\sigma_{\rm g, low}$" + M_unit,
    "sigma_g_high": r"$\sigma_{\rm g, high}$" + M_unit,
    "mu_g_low": r"$\mu_{\rm g, low}$" + M_unit,
    "mu_g_high": r"$\mu_{\rm g, high}$" + M_unit,
    "delta_m": r"$\delta_{\rm m}$" + M_unit,
    "lambda_g_low": r"$\lambda_{\rm g, low}$",
    "lambda_g": r"$\lambda_{\rm g}$",
    "zp": r"$z_{\rm p}$",
    "gamma": r"$\gamma$",
    "kappa": r"$\kappa$",
    "beta_0": r"$\beta_0$",
    "beta_1": r"$\beta_1$",
    "mass_max": r"$m_{\rm max}$" + M_unit,
    "mass_min": r"$m_{\rm min}$" + M_unit,
    "sigma_high_fractional": r"$\sigma_{\rm f, high}$",
    "sigma_low_fractional": r"$\sigma_{\rm f, low}$",
    "alpha_0": r"$\alpha_{\mathrm{ref}}$",
    "sigma_mass_cutoff_mass_2": r"$\delta_{m,2}$" + M_unit,
    "power_law_reference_mass_1_s": r"$\alpha_{\rm ref}$",
    "power_law_reference_mass_ratio": r"$\beta_{\rm ref}$",
    "power_spectrum_amplitude": r"$A_{\rm PS}$",
    "power_spectrum_cutoff": r"$k_{\rm cut}$",
    "mass_1_d": r"$m_{1,{\rm d}}$" + M_unit,
    "mass_ratio": r"$q$",
    "luminosity_distance": r"$d_{\rm L}$ [Mpc]",
    "chirp_mass_d": r"$\mathcal{M}_{\rm c, d}$" + M_unit,
}


def plot_mollweide_healpix(
    map_data,
    scale=None,
    cmap="viridis",
    title="",
    vmin=None,
    vmax=None,
    ax=None,
    **kwargs,
):

    # mollview can only take these scale args
    if scale in ["log", None]:
        pass
    else:
        raise RuntimeError(f"scale must be 'log' or None. You passed '{scale}'. ")

    if ax is None:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111, projection="mollweide")

    plot_kwargs = dict(bgcolor="white", hold=True, badcolor="white")
    plot_kwargs.update(dict(min=vmin, max=vmax))
    plot_kwargs.update(dict(norm=scale, cmap=cmap))
    plot_kwargs.update(kwargs)

    plt.sca(ax)
    hp.mollview(map_data, title=title, **plot_kwargs)

    # hp.graticule()
    return ax


def plot_mollview_with_healpix(
    map_data, scale=None, cmap="viridis", title="", vmin=None, vmax=None, **kwargs
):

    # mollview can only take these scale args
    if scale in ["log", None]:
        pass
    else:
        raise RuntimeError(f"scale must be 'log' or None. You passed '{scale}'. ")

    plot_kwargs = dict(bgcolor="white", hold=True, badcolor="white")
    plot_kwargs.update(dict(min=vmin, max=vmax))
    plot_kwargs.update(dict(norm=scale, cmap=cmap))
    plot_kwargs.update(kwargs)

    hp.projview(map_data, title=title, **plot_kwargs)


def plot_trace_scalar_variables(inf_data, ncols=7):
    """
    Plot MCMC trace lines for all scalar (chain, draw) variables in an ArviZ
    InferenceData object. Each chain is drawn as a separate line.

    Parameters
    ----------
    inf_data : arviz.InferenceData
    ncols : int
        Number of columns in the subplot grid.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    scalar_vars = [
        v
        for v in inf_data.posterior.data_vars
        if len(inf_data.posterior[v].dims) == 2 and not v.endswith("_base")
    ]
    if not scalar_vars:
        print("No scalar variables found in posterior.")
        return None

    nrows = (len(scalar_vars) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 5 * nrows / 3))
    axes = axes.flatten()

    posterior = inf_data.posterior
    for i, param in enumerate(scalar_vars):
        da = posterior[param]
        for chain_idx in range(da.sizes["chain"]):
            axes[i].plot(da.isel(chain=chain_idx).values, lw=0.7, alpha=0.8)
        axes[i].set_title(param, fontsize=8)
        axes[i].tick_params(labelsize=6)

    for j in range(len(scalar_vars), len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    return fig


def get_mollweide_ra_from_ra(ra):
    ra_wrapped = (-ra + jnp.pi) % (2 * jnp.pi) - jnp.pi
    return ra_wrapped


def _plot_marginal_on_ax(ax, vals, perc, color=None, label=None):
    """Plot median + credible bands on an existing axis."""
    ax.plot(vals, perc[2], color=color, label=label)
    ax.fill_between(vals, perc[1], perc[-2], alpha=0.4, color=color)
    # ax.fill_between(vals, perc[0], perc[-1], alpha=0.2, color=color)


def clean_log_pdfs(log_pdfs):
    v_min = jnp.nanmin(log_pdfs[~jnp.isnan(log_pdfs) & ~jnp.isinf(log_pdfs)])

    return jnp.where(jnp.isnan(log_pdfs) | jnp.isinf(log_pdfs), v_min, log_pdfs)


def extent_from_centers(x_vals, y_vals, rtol=1e-3):
    """imshow's ``extent`` is the outer edge of the corner pixels, not the bin centers.

    Also warns if the grid is non-uniform, in which case imshow is the wrong
    primitive altogether (use pcolormesh).
    """
    x_vals = jnp.asarray(x_vals)
    y_vals = jnp.asarray(y_vals)

    for name, v in (("x", x_vals), ("y", y_vals)):
        d = jnp.diff(v)
        if float(jnp.max(jnp.abs(d - d[0]))) > rtol * float(jnp.abs(d[0])):
            warnings.warn(
                f"{name} grid is non-uniform; imshow assumes uniform spacing and "
                "will misplace the data. Use pcolormesh instead.",
                RuntimeWarning,
            )

    dx = float(x_vals[1] - x_vals[0]) / 2
    dy = float(y_vals[1] - y_vals[0]) / 2

    return [
        float(x_vals[0]) - dx,
        float(x_vals[-1]) + dx,
        float(y_vals[0]) - dy,
        float(y_vals[-1]) + dy,
    ]


def draw_2d_posterior_with_marginals(
    axes,
    x_vals,
    y_vals,
    log_pdf_on_grid,
    x_label="x",
    y_label="y",
    idx="median",
    vmin=None,
    vmax=None,
    extent=None,
    marginal_x_log=True,
    marginal_x_ylim=None,
    marginal_y_log=True,
    marginal_y_xlim=None,
    plot_marginals=True,
    cmap=None,
    fontsize=20,
):
    """
    Generic 2D density frame with optional marginals.

    Parameters
    ----------
    axes :
        If plot_marginals=True:
            [ax_main, ax_top, ax_right, ax_cbar]
        If plot_marginals=False:
            ax_main
    """
    if plot_marginals:
        ax_main, ax_top, ax_right, ax_cbar = axes
    else:
        if isinstance(axes, (list, tuple)):
            ax_main = axes[0]
            ax_cbar = axes[1] if len(axes) > 1 else None
        else:
            ax_main = axes
            ax_cbar = None

    log_pdf_on_grid = clean_log_pdfs(log_pdf_on_grid)

    if idx == "median":
        data_2D = jnp.median(log_pdf_on_grid, axis=0)
    elif type(idx) is not int:
        raise ValueError("idx must be an integer or 'median'. You passed: {idx}")
    else:
        data_2D = log_pdf_on_grid[idx]

    extent_data = extent_from_centers(x_vals, y_vals)

    if vmin is None or vmax is None:
        if vmax is None:
            vmax = float(jnp.percentile(data_2D, 99))
        if vmin is None:
            vmin = vmax - 6

    # --- Main image plot ---
    im = ax_main.imshow(
        data_2D.T,
        origin="lower",
        aspect="auto",
        extent=extent_data,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    ax_main.set_xlabel(x_label, fontsize=fontsize)
    ax_main.set_ylabel(y_label, fontsize=fontsize)

    if extent is None:
        extent = extent_data
    ax_main.set_xlim(extent[0], extent[1])
    ax_main.set_ylim(extent[2], extent[3])

    if plot_marginals:
        # Marginals (integrate over the other axis)
        pdf_x = jnp.trapezoid(jnp.exp(log_pdf_on_grid), x=y_vals, axis=-1)
        perc_x = jnp.percentile(pdf_x, jnp.array([5, 16, 50, 84, 95]), axis=0)

        pdf_y = jnp.trapezoid(jnp.exp(log_pdf_on_grid), x=x_vals, axis=-2)
        perc_y = jnp.percentile(pdf_y, jnp.array([5, 16, 50, 84, 95]), axis=0)

        # Derive marginal limits from 2D vmin/vmax, scaled by integration extent
        if marginal_x_ylim is None:
            delta_y = extent[3] - extent[2]
            marginal_x_ylim = (jnp.exp(vmin) * delta_y, jnp.exp(vmax) * delta_y)
        if marginal_y_xlim is None:
            delta_x = extent[1] - extent[0]
            marginal_y_xlim = (jnp.exp(vmin) * delta_x, jnp.exp(vmax) * delta_x)

        # --- Top marginal ---
        _plot_marginal_on_ax(ax_top, x_vals, perc_x)
        if marginal_x_log:
            ax_top.set_yscale("log")
        ax_top.set_ylim(*marginal_x_ylim)
        ax_top.set_xlim(extent[0], extent[1])
        ax_top.tick_params(axis="x", labelbottom=False)

        # --- Right marginal (rotated: y_vals on y-axis, pdf on x-axis) ---
        ax_right.plot(perc_y[2], y_vals)
        ax_right.fill_betweenx(y_vals, perc_y[1], perc_y[-2], alpha=0.4)
        if marginal_y_log:
            ax_right.set_xscale("log")
        ax_right.set_xlim(*marginal_y_xlim)
        ax_right.set_ylim(extent[2], extent[3])
        ax_right.tick_params(axis="y", labelleft=False)

    # --- Colorbar ---
    if ax_cbar is not None and not ax_cbar.collections and not ax_cbar.images:
        fig = ax_main.get_figure()
        cbar = fig.colorbar(im, cax=ax_cbar)
        cbar.set_label("log probability")

    return im


def make_mass_plot_row(
    plot_inputs,
    idx="median",
    filename=None,
    x_label=r"$m_{1,{\rm s}}$ $[M_\odot]$",
    y_label=r"$q$",
    fontsize=20,
    vmin=None,
    vmax=None,
    extent=None,
    figsize_per_panel=(4.5, 4.0),
    share_colorbar=True,  # Defaulted to True as requested
    add_uncertainty=False,
    cmap="viridis",
    cmap_uncertainty="magma",
    include_color_boxes=True,
):
    """
    Plot one or more mass distributions in a row.

    Parameters
    ----------
    plot_inputs : sequence
        Sequence of SimpleNamespace-like objects with ``grid_ref``, ``log_pdfs``,
        and optional ``label`` attributes.
    """

    def get_plot_input(plot_input):
        if hasattr(plot_input, "grid_ref") and hasattr(plot_input, "log_pdfs"):
            return (
                plot_input.grid_ref,
                plot_input.log_pdfs,
                getattr(plot_input, "label", None),
            )

        raise TypeError(
            "plot_inputs must contain SimpleNamespace-like objects with "
            "grid_ref and log_pdfs attributes."
        )

    def add_label_with_color_box(ax, label, color, titlesize):
        "If color is None, no color box is drawn. If label is None, nothing is drawn."

        if label is None:
            return

        if color is None:
            ax.set_title(label, fontsize=titlesize, pad=5)
            return

        from matplotlib.offsetbox import (
            AnchoredOffsetbox,
            DrawingArea,
            HPacker,
            TextArea,
        )
        from matplotlib.patches import Rectangle

        swatch_w, swatch_h = 24, 16

        offset = swatch_h * 0.12
        swatch = DrawingArea(swatch_w, swatch_h + offset, 0, 0)
        swatch.add_artist(
            Rectangle(
                (0, offset),
                swatch_w,
                swatch_h,
                facecolor=color,
                edgecolor="black",
                linewidth=0.5,
            )
        )

        text = TextArea(label, textprops={"size": titlesize})
        title_box = HPacker(children=[swatch, text], align="center", pad=0, sep=5)
        anchored_title = AnchoredOffsetbox(
            loc="lower center",
            child=title_box,
            pad=0,
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )
        ax.add_artist(anchored_title)

    def has_posterior_samples(log_pdfs):
        """A single 'sample' (e.g. an injected truth) carries no uncertainty."""
        return log_pdfs["log_prob_mass_1_s_mass_ratio"].shape[0] > 1

    def uncertainty_metric(log_pdfs):
        """log[(p_84 - p_16) / (2 p_50)] on the (mass_1_s, mass_ratio) grid.

        The log is what the colorbar label and the paper definition specify, and
        what the unc_vmin/unc_vmax logic below assumes.

        Returns the metric together with the low-probability mask, so callers can
        apply the fill value without recomputing the percentiles: the colour
        scale is derived from the unmasked metric, and the fill value it yields
        is only known afterwards.
        """
        log_pdf = clean_log_pdfs(log_pdfs["log_prob_mass_1_s_mass_ratio"])
        perc = jnp.percentile(jnp.exp(log_pdf), jnp.array([16.0, 50.0, 84.0]), axis=0)

        # Guards must be relative: p is a density in 1/M_sun, so an absolute
        # epsilon silently rescales the tails (where p can be many orders of
        # magnitude below the peak) and makes them look well measured.
        p_scale = jnp.max(perc[1])
        p_50 = jnp.maximum(perc[1], 1e-30 * p_scale)
        ratio = (perc[2] - perc[0]) / (2 * p_50)
        uncertainty_measure = jnp.log(jnp.maximum(ratio, 1e-30))

        # Regions where the distribution carries essentially no probability; the
        # relative uncertainty there is meaningless.
        low_prob_mask = perc[1] < 1e-7 * p_scale

        return uncertainty_measure, low_prob_mask

    nplots = len(plot_inputs)
    if nplots == 0:
        raise ValueError("plot_inputs must contain at least one element.")
    plot_inputs = [get_plot_input(plot_input) for plot_input in plot_inputs]
    if idx != "median" and not isinstance(idx, int):
        raise ValueError(f"idx must be an integer or 'median'. You passed: {idx}")

    # --- Calculate Percentiles for Dynamic Scaling ---
    if vmin is None or vmax is None:
        all_data_to_plot = []
        for _, log_pdfs, _ in plot_inputs:
            data = log_pdfs["log_prob_mass_1_s_mass_ratio"]
            if idx == "median":
                all_data_to_plot.append(jnp.median(data, axis=0))
            else:
                all_data_to_plot.append(data[idx])

        combined_data = jnp.concatenate([d.flatten() for d in all_data_to_plot])

        if vmax is None:
            vmax = float(jnp.percentile(combined_data, 99.99))
        if vmin is None:
            vmin = vmax - 6

        print(f"Calculated shared vmin: {vmin:.2f}, vmax: {vmax:.2f}")

    # --- Uncertainty (1-sigma width of log prob across posterior samples) ---
    if add_uncertainty:
        # Panels with a single sample (an injected truth) have zero spread by
        # construction and must not enter the shared colour scale.
        # Computed once per panel and reused by the uncertainty row below;
        # None marks a panel that has no uncertainty to show.
        uncertainties = [
            uncertainty_metric(log_pdfs) if has_posterior_samples(log_pdfs) else None
            for _, log_pdfs, _ in plot_inputs
        ]
        us = [u for u in uncertainties if u is not None]
        if not us:
            raise ValueError(
                "add_uncertainty=True but no plot_input has more than one posterior sample."
            )
        combined_unc = jnp.concatenate([u.flatten() for u, _ in us])
        combined_unc = combined_unc[jnp.isfinite(combined_unc)]
        unc_vmax = min(float(jnp.percentile(combined_unc, 99.5)), float(jnp.log(10.0)))
        unc_vmin = unc_vmax - 4.0

    # --- Layout with GridSpec ---
    from matplotlib.gridspec import GridSpec

    cbar_width_ratio = 0.06 if share_colorbar else 0.0
    ncols = nplots + (1 if share_colorbar else 0)
    width_ratios = [1] * nplots + ([cbar_width_ratio] if share_colorbar else [])

    nrows = 2 if add_uncertainty else 1
    fig = plt.figure(
        figsize=(
            figsize_per_panel[0] * nplots + (0.8 if share_colorbar else 0),
            figsize_per_panel[1] * nrows,
        )
    )
    gs = GridSpec(
        nrows, ncols, figure=fig, width_ratios=width_ratios, wspace=0.05, hspace=0.12
    )

    axes_main = [fig.add_subplot(gs[0, i]) for i in range(nplots)]
    if share_colorbar:
        cbar_ax_main = fig.add_subplot(gs[0, nplots])

    if add_uncertainty:
        axes_unc = [fig.add_subplot(gs[1, i]) for i in range(nplots)]
        if share_colorbar:
            cbar_ax_unc = fig.add_subplot(gs[1, nplots])

    # --- Main row ---
    ims = []
    for i, ((grid_ref, log_pdfs, label), ax) in enumerate(zip(plot_inputs, axes_main)):
        if idx == "median":
            log_prob = jnp.median(log_pdfs["log_prob_mass_1_s_mass_ratio"], axis=0)[
                None, ...
            ]
            idx_use = 0
        elif isinstance(idx, int):
            log_prob = log_pdfs["log_prob_mass_1_s_mass_ratio"]
            idx_use = idx

        # Hide x labels on the main row when uncertainty row sits below
        x_label_main = "" if add_uncertainty else x_label

        im = draw_2d_posterior_with_marginals(
            ax,
            x_vals=grid_ref["mass_1_s"],
            y_vals=grid_ref["mass_ratio"],
            log_pdf_on_grid=log_prob,
            x_label=x_label_main,
            y_label=y_label if i == 0 else "",
            idx=idx_use,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            plot_marginals=False,
            cmap=cmap,
            fontsize=fontsize,
        )
        ims.append(im)

        add_label_with_color_box(
            ax, label, f"C{i}" if include_color_boxes else None, titlesize=fontsize + 2
        )

        if i > 0:
            ax.tick_params(axis="y", labelleft=False)

        if add_uncertainty:
            ax.tick_params(axis="x", labelbottom=False)

    # --- Uncertainty row ---
    ims_unc = []
    if add_uncertainty:
        for i, ((grid_ref, _, _), ax) in enumerate(zip(plot_inputs, axes_unc)):
            x_vals = grid_ref["mass_1_s"]
            y_vals = grid_ref["mass_ratio"]
            extent_data = extent_from_centers(x_vals, y_vals)

            if uncertainties[i] is not None:
                uncertainty_measure, low_prob_mask = uncertainties[i]
                uncertainty_indicator = jnp.where(
                    low_prob_mask, unc_vmax, uncertainty_measure
                )

                im_u = ax.imshow(
                    uncertainty_indicator.T,
                    origin="lower",
                    aspect="auto",
                    extent=extent_data,
                    cmap=cmap_uncertainty,
                    vmin=unc_vmin,
                    vmax=unc_vmax,
                )
                ims_unc.append(im_u)
            else:
                # Single sample: no uncertainty is defined. Leave the panel blank
                # rather than rendering a constant that reads as a measurement.
                ax.text(
                    0.5,
                    0.5,
                    "n/a",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    color="0.5",
                )

            ax.set_xlabel(x_label, fontsize=fontsize)
            if i == 0:
                ax.set_ylabel(y_label, fontsize=fontsize)
            else:
                ax.tick_params(axis="y", labelleft=False)

            plot_extent = extent if extent is not None else extent_data
            ax.set_xlim(plot_extent[0], plot_extent[1])
            ax.set_ylim(plot_extent[2], plot_extent[3])

    # --- Shared Colorbars (aligned via GridSpec) ---
    if share_colorbar:
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=ims[0].cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax_main, label="log probability")

        if add_uncertainty:
            norm_u = plt.Normalize(vmin=unc_vmin, vmax=unc_vmax)
            sm_u = plt.cm.ScalarMappable(cmap=ims_unc[0].cmap, norm=norm_u)
            sm_u.set_array([])
            fig.colorbar(
                sm_u,
                cax=cbar_ax_unc,
                label=r"$\log [(p_{84} - p_{16}) /  p_{50} / 2] $",
            )

    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")

    plt.show()
    plt.close(fig)

    return fig


# TODO: can remove?
def draw_frame(axes, grid_ref_book_keeping, log_pdfs, idx=0):
    """Legacy wrapper — draws in (mass_1_s, mass_ratio) space."""
    print(
        f"draw_frame is deprecated. Please call draw_2d_posterior_with_marginals directly with your desired x/y values and labels. You passed idx={idx}."
    )

    return draw_2d_posterior_with_marginals(
        axes,
        x_vals=grid_ref_book_keeping["mass_1_s"],
        y_vals=grid_ref_book_keeping["mass_ratio"],
        log_pdf_on_grid=log_pdfs["log_prob_mass_1_s_mass_ratio"],
        x_label=r"$m_{1,{\rm s}}$",
        y_label=r"$q$",
        idx=idx,
    )


def make_2d_mass_predictive_plot(
    x_vals,
    y_vals,
    log_pdf_on_grid,
    x_label,
    y_label,
    idx="median",
    extent=None,
    filename=None,
    plot_marginals=True,
    **draw_kwargs,
):
    """Shared logic for all 2D mass-plot entry points."""
    fig, axes = setup_layout(plot_marginals=plot_marginals)
    draw_2d_posterior_with_marginals(
        axes,
        x_vals,
        y_vals,
        log_pdf_on_grid,
        x_label=x_label,
        y_label=y_label,
        idx=idx,
        extent=extent,
        plot_marginals=plot_marginals,
        **draw_kwargs,
    )
    if filename:
        plt.savefig(filename)
        plt.close(fig)
    else:
        plt.show()
    return fig


def make_mass_plot(
    grid_ref_book_keeping,
    log_pdfs,
    idx="median",
    extent=None,
    filename=None,
    plot_marginals=True,
    coordinates="m1s_q",
):
    """Plot p(m1_s, q) with marginals."""

    if coordinates == "m1s_q":
        log_pdf_on_grid = log_pdfs["log_prob_mass_1_s_mass_ratio"]
        x_vals = grid_ref_book_keeping["mass_1_s"]
        y_vals = grid_ref_book_keeping["mass_ratio"]
        x_label = r"$m_{1,{\rm s}}\,[M_\odot]$"
        y_label = r"$q$"

    elif coordinates == "m1s_m2s":
        raise NotImplementedError(
            "This code is not fully tested yet. Please verify the coordinate transformation and jacobian before using."
        )

    else:
        raise ValueError(
            f"Invalid coordinates: {coordinates}. Must be 'm1s_q' or 'm1s_m2s'."
        )

    return make_2d_mass_predictive_plot(
        x_vals=x_vals,
        y_vals=y_vals,
        log_pdf_on_grid=log_pdf_on_grid,
        x_label=x_label,
        y_label=y_label,
        idx=idx,
        extent=extent,
        filename=filename,
        plot_marginals=plot_marginals,
    )


def make_mass_plot_s_delta(
    grid_ref_book_keeping, log_pdfs, idx=0, filename=None, **kwargs
):
    """Plot p(log_mass_total_s, minus_log_mass_ratio) with marginals."""
    return make_2d_mass_predictive_plot(
        x_vals=grid_ref_book_keeping["log_mass_total_s"],
        y_vals=grid_ref_book_keeping["minus_log_mass_ratio"],
        log_pdf_on_grid=log_pdfs["log_prob_mass_1_s_mass_ratio"],
        x_label=r"$\log(m_{1,{\rm s}} + m_{2,{\rm s}})$",
        y_label=r"$\log(m_{1,{\rm s}} / m_{2,{\rm s}})$",
        idx=idx,
        filename=filename,
        marginal_x_ylim=None,
        marginal_y_xlim=None,
        **kwargs,
    )


def setup_layout(plot_marginals=True):
    if not plot_marginals:
        fig = plt.figure(figsize=(7, 5.5))
        gs = fig.add_gridspec(
            1,
            2,
            width_ratios=[1, 0.06],
            wspace=0.08,
        )
        ax_main = fig.add_subplot(gs[0, 0])
        ax_cbar = fig.add_subplot(gs[0, 1])
        return fig, [ax_main, ax_cbar]

    fig = plt.figure(figsize=(9, 6.5))

    # 1. Create an outer grid with a large gap for the right column
    # The width_ratios [4, 1] creates the "some space" effect
    outer = fig.add_gridspec(
        2,
        2,
        width_ratios=[4, 1.2],
        height_ratios=[1, 3.5],
        wspace=0.22,  # This controls the "big gap" before the right marginal
        hspace=0.05,
    )

    # 2. Nest the Main Plot and Colorbar tightly in the left column
    # width_ratios [1, 0.05] keeps the colorbar close to the imshow
    left_content = outer[1, 0].subgridspec(
        1,
        2,
        width_ratios=[1, 0.06],
        wspace=0.08,  # This keeps the colorbar close to the left plot
    )

    # 3. Define the Axes
    ax_main = fig.add_subplot(left_content[0, 0])
    ax_cbar = fig.add_subplot(left_content[0, 1])

    # Match the top marginal to the width of ax_main only
    ax_top = fig.add_subplot(
        outer[0, 0].subgridspec(1, 2, width_ratios=[1, 0.06])[0, 0], sharex=ax_main
    )

    # Place the right marginal in the outer right column
    ax_right = fig.add_subplot(outer[1, 1], sharey=ax_main)

    # Formatting to match your image
    ax_top.tick_params(labelbottom=False)
    ax_right.tick_params(labelleft=False)

    return fig, [ax_main, ax_top, ax_right, ax_cbar]


def make_percentile_plot(
    grid_vals,
    f_samples,
    color,
    label,
    include_90_percentiles=False,
    include_first_N_lines=None,
    fig=None,
):
    if fig is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        ax = fig.gca()  # get current axis from figure

    # Compute percentiles
    perc = jnp.percentile(f_samples, q=jnp.array([5, 16, 50, 84, 95]), axis=0)

    # Plot median and credible intervals
    ax.plot(grid_vals, perc[2], label=label, color=color)
    ax.fill_between(grid_vals, perc[1], perc[3], alpha=0.5, color=color)
    if include_90_percentiles:
        ax.fill_between(grid_vals, perc[0], perc[4], alpha=0.3, color=color)

    if include_first_N_lines is not None:
        for i in range(min(include_first_N_lines, f_samples.shape[0])):
            ax.plot(
                grid_vals, f_samples[i], color=color, alpha=1.0 / include_first_N_lines
            )

    return ax


def plot_marginal_m1(
    samples_log_prob,
    grid_ref,
    color=None,
    label=None,
    fig=None,
    correction_factor=None,
    xlim=None,
    ylim=(2e-4, 1),
    perc=None,
):
    """Plot the marginal distribution over m1 (x-axis marginal).

    Parameters
    ----------
    samples_log_prob : array (N_samples, N_x, N_y)
        Log-probability samples on the 2D grid.
    grid_ref : dict
        Must contain 'mass_1_s' (x grid) and 'mass_ratio' (y grid).
    color : str, optional
        Line / fill color.
    label : str, optional
        Legend label for the median line.
    fig : Figure, optional
        If None a new figure is created; otherwise plots on fig's current axis.
    correction_factor : float
        Multiplicative correction applied to the pdf before plotting.
    """
    x_vals = grid_ref["mass_1_s"]
    y_vals = grid_ref["mass_ratio"]

    if perc is None:
        perc = jnp.array([16, 50, 84])

    if len(perc) % 2 == 0 or perc[len(perc) // 2] != 50:
        raise ValueError(
            "perc must have an odd number of elements to be symmetric around the median."
        )

    mid_idx = len(perc) // 2

    pdf_x = jnp.trapezoid(jnp.exp(samples_log_prob), x=y_vals, axis=-1)
    perc_x = jnp.percentile(pdf_x, perc, axis=0)

    if correction_factor is None:
        # Normalize the pdf so that the area under the median curve is 1
        median = perc_x[mid_idx]
        correction_factor = 1 / jnp.trapezoid(median, x=x_vals)

    perc_x *= correction_factor

    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = fig.gca()

    ax.plot(x_vals, perc_x[mid_idx], color=color, label=label, alpha=0.6)
    for i in range(len(perc_x) // 2):
        ax.fill_between(
            x_vals, perc_x[i], perc_x[-i - 1], alpha=0.2 / (i + 1), color=color
        )
    ax.set_xlabel(r"$m_{1,{\rm s}}$")
    ax.set_ylabel(r"$p(m_{1,{\rm s}})$")
    ax.set_yscale("log")
    ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    return ax


def make_luminosity_distance_ratio_plot(
    grid_ref_book_keeping,
    samples_postp,
    samples_prior=None,
    xlim=None,
    include_first_N_lines=None,
    filepath=None,
    fig=None,
    **kwargs,
):

    if "ratio_luminosity_distance_gw_em" not in samples_postp:
        print("No luminosity distance ratio in samples_postp. Skipping plot.")
        return

    if fig is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        ax = fig.gca()

    make_percentile_plot(
        grid_ref_book_keeping["redshift"],
        samples_postp["ratio_luminosity_distance_gw_em"].squeeze(),
        color="blue",
        label="Posterior",
        fig=fig,
        **kwargs,
    )

    if samples_prior is not None:
        make_percentile_plot(
            grid_ref_book_keeping["redshift"],
            samples_prior["ratio_luminosity_distance_gw_em"].squeeze(),
            color="grey",
            label="Prior",
            fig=fig,
            include_first_N_lines=include_first_N_lines,
            **kwargs,
        )

    ax.legend()
    ax.set_xlabel("Redshift")
    ax.set_ylabel("Luminosity Distance Ratio")
    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(
            grid_ref_book_keeping["redshift"][0],
            min(1, grid_ref_book_keeping["redshift"][-1]),
        )
    if filepath is not None:
        plt.savefig(filepath)
        plt.close()

    return fig, ax


def find_shared_params_from_dfs(dfs, exclude_base_params=True):
    shared_params = set(dfs[0].columns)
    for df in dfs[1:]:
        shared_params = shared_params.intersection(set(df.columns))
    if exclude_base_params:
        shared_params = {p for p in shared_params if not p.endswith("_base")}
    return list(shared_params)


def plot_posterior_comparison(
    posteriors, truth_dict=None, filepath=None, **corner_kwargs
):
    """
    Plot one corner plot from a list of panda dataframes.
    Only the shared parameters will be plotted.
    """
    if "fig" not in corner_kwargs:
        corner_kwargs["fig"] = None

    color_dict = {
        k: getattr(posteriors[k], "color", f"C{i}")
        for i, k in enumerate(posteriors.keys())
    }

    corner_kwargs = CORNER_KWARGS_DEFAULT | corner_kwargs

    if corner_kwargs.get("params_list", None) is None:
        shared_parameters = find_shared_params_from_dfs(
            [posteriors[k].posterior for k in posteriors.keys()]
        )
    else:
        shared_parameters = corner_kwargs["params_list"]

    labels = []
    for i, k in enumerate(posteriors.keys()):
        arr, sorted_keys = construct_array_from_df(
            posteriors[k].posterior, keys=shared_parameters, sort_keys=False
        )
        latex_labels = [PARAMETER_LABELS.get(p, p) for p in sorted_keys]

        corner_kwargs["hist_kwargs"]["color"] = color_dict[k]
        corner_kwargs["contour_kwargs"]["colors"] = color_dict[k]
        corner_kwargs["color"] = color_dict[k]

        if truth_dict is not None:
            if i == len(posteriors) - 1:
                truth_array = [truth_dict.get(p, None) for p in sorted_keys]
                corner_kwargs["truths"] = truth_array

        fig = corner.corner(arr, labels=latex_labels, **corner_kwargs)
        corner_kwargs["fig"] = fig

        if hasattr(posteriors[k], "label"):
            labels.append(posteriors[k].label)
        else:
            labels.append(k)

    # add legend
    handles = [
        plt.Line2D([0], [0], color=color_dict[k], lw=2) for k in posteriors.keys()
    ]

    fig.legend(
        handles,
        labels,
        loc="upper right",
        fontsize=corner_kwargs.get("legend_fontsize", 30),
    )
    if filepath is not None:
        plt.savefig(filepath)
        plt.close()
    return fig, shared_parameters
