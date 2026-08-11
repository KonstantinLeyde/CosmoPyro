import os
from collections import Counter
from pathlib import Path

import yaml

from .jupyter_formatting import display_differences

__all__ = [
    "find_and_display_differences",
    "find_differences",
    "find_results",
    "load_settings",
    "print_diffs_as_latex",
]


def load_settings(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _flatten_dict(d, parent_key="", sep="."):
    items = {}
    for k, v in (d or {}).items():
        key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, key, sep))
        else:
            items[key] = v
    return items


def _matches_filter(settings, filter_dict):
    flat = _flatten_dict(settings)
    for k, v in _flatten_dict(filter_dict).items():
        if flat.get(k) != v:
            return False
    return True


def _has_samples(result_folder):
    samples_dir = os.path.join(result_folder, "samples")
    if not os.path.isdir(samples_dir):
        return False
    return any(f.endswith(".av") for f in os.listdir(samples_dir))


def _unique_labels(settings_paths):
    """Label each result by its folder name, disambiguated when names repeat.

    Nested results can share the same ``id_*`` folder name, e.g.
    ``v13/a/id_1`` and ``v13/b/id_1``. Colliding labels get parent folders
    prepended (``a/id_1``, ``b/id_1``) until they are unique.
    """
    dirs = [Path(p).parent for p in settings_paths]
    depths = [1] * len(dirs)
    while True:
        labels = ["/".join(d.parts[-k:]) for d, k in zip(dirs, depths)]
        counts = Counter(labels)
        # only grow labels that still collide and have a parent left to add
        growable = [
            i
            for i, label in enumerate(labels)
            if counts[label] > 1 and depths[i] < len(dirs[i].parts)
        ]
        if not growable:
            return labels
        for i in growable:
            depths[i] += 1


def find_results(results_dir, filter_dict=None, only_finished=False):
    """Find result folders whose kwargs_analysis matches a filter.

    Parameters
    ----------
    results_dir : str
        Root directory containing result folders. Searched recursively, so
        results may live in arbitrarily nested subfolders.
    filter_dict : dict, optional
        Nested dict of required settings. Uses dot-flattened key
        matching, e.g. ``{"distribution_names": {"source_frame_masses": "fourier_gp_2D_logMdelta"}}``
        matches any result whose ``distribution_names.source_frame_masses``
        equals that value.
    only_finished : bool
        If True, only return folders that contain ``.av`` sample files.

    Returns
    -------
    list of str
        Paths to matching ``kwargs_analysis.yaml`` files.
    """
    matches = []
    for yaml_path in sorted(Path(results_dir).rglob("kwargs_analysis.yaml")):
        entry = yaml_path.parent
        if only_finished and not _has_samples(str(entry)):
            continue
        if filter_dict is not None:
            settings = load_settings(yaml_path)
            if not _matches_filter(settings, filter_dict):
                continue
        matches.append(str(yaml_path))
    return matches


def find_differences(settings_paths):
    """Compare kwargs_analysis across multiple results and return differences.

    Parameters
    ----------
    settings_paths : list of str
        Paths to ``kwargs_analysis.yaml`` files.

    Returns
    -------
    list of dict
        One dict per setting, with keys ``parameter``, ``values``
        (list of per-result values), ``labels`` (folder names, extended
        with parent folders where names would otherwise collide), and
        ``majority_value``. Ties are resolved by first occurrence in
        ``settings_paths``.
    """
    if len(settings_paths) < 2:
        print("Need at least two settings paths to compare.")
        return []

    all_settings = [load_settings(p) for p in settings_paths]
    labels = _unique_labels(settings_paths)
    all_flat = [_flatten_dict(s) for s in all_settings]

    all_keys = sorted(set().union(*(d.keys() for d in all_flat)))

    diffs = []
    for key in all_keys:
        vals = [d.get(key) for d in all_flat]
        # skip keys that are identical across all results
        if all(v == vals[0] for v in vals):
            continue
        value_strings = [str(v) if v is not None else "<em>missing</em>" for v in vals]
        counts = Counter(value_strings)
        majority_value = max(value_strings, key=lambda v: counts[v])
        diffs.append(
            dict(
                parameter=key,
                values=vals,
                labels=labels,
                majority_value=majority_value,
            )
        )
    return diffs


def find_and_display_differences(setting_dirs):
    settings_paths = [os.path.join(dir, "kwargs_analysis.yaml") for dir in setting_dirs]
    diffs = find_differences(settings_paths)
    display_differences(diffs)

    return diffs


def print_diffs_as_latex(
    diffs, param_mapping=None, enumerate_labels=False, color_list=None
):
    """Parses the output of find_differences and prints it as a custom LaTeX table.

    Parameters
    ----------
    diffs : list of dict
        Output from the find_differences function.
    param_mapping : dict, optional
        A dictionary mapping original parameter names to their LaTeX representations.
    enumerate_labels : bool, default False
        If True, replaces string labels with numbers accompanied by a colored box.
    color_list : list of str, optional
        A list of hex strings (without '#') or standard LaTeX color names.
        Defaults to the exact Matplotlib C0-C9 palette.
    """
    if not diffs:
        print("% No differences found to display.")
        return

    # Default to Matplotlib's exact C0 through C9 hex color cycle
    if color_list is None:
        color_list = [
            "1F77B4",  # C0 (Blue)
            "FF7F0E",  # C1 (Orange)
            "2CA02C",  # C2 (Green)
            "D62728",  # C3 (Red)
            "9467BD",  # C4 (Purple)
            "8C564B",  # C5 (Brown)
            "E377C2",  # C6 (Pink)
            "7F7F7F",  # C7 (Gray)
            "BCBD22",  # C8 (Olive)
            "17BECF",  # C9 (Cyan)
        ]

    def latex_escape(text):
        if text is None:
            return ""
        text = str(text)
        if text == "<em>missing</em>":
            return r"\textit{missing}"
        replacements = {
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\}",
            "}": r"\}",
        }
        return "".join(replacements.get(c, c) for c in text)

    filtered_diffs = []
    for d in diffs:
        orig_param = d["parameter"]
        if param_mapping is not None and orig_param not in param_mapping:
            continue
        latex_param = (
            param_mapping[orig_param] if param_mapping else latex_escape(orig_param)
        )
        filtered_diffs.append((latex_param, d["values"], d["labels"]))

    if not filtered_diffs:
        print("% No matching parameters found after filtering.")
        return

    _, sample_vals, original_labels = filtered_diffs[0]
    num_files = len(sample_vals)

    header_cols = []
    if enumerate_labels:
        for i in range(num_files):
            color = color_list[i % len(color_list)]
            # If it looks like a hex string, use the [HTML] macro variant
            if len(color) == 6 and all(c in "0123456789ABCDEFabcdef" for c in color):
                box_str = f"\\textcolor[HTML]{{{color}}}{{\\rule{{1.5ex}}{{1.5ex}}}}"
            else:
                box_str = f"\\textcolor{{{color}}}{{\\rule{{1.5ex}}{{1.5ex}}}}"
            header_cols.append(f"{box_str}~[{i + 1}]")
    else:
        header_cols = [latex_escape(lbl) for lbl in original_labels]

    col_alignment = "l" + "c" * num_files

    print(r"\begin{table}[htbp]")
    print(r"  \centering")
    print(f"  \\begin{{tabular}}{{{col_alignment}}}")
    print(r"    \toprule")
    print("    Parameter & " + " & ".join(header_cols) + r" \\")
    print(r"    \midrule")

    for latex_param, values, _ in filtered_diffs:
        escaped_vals = [
            latex_escape(v if v is not None else "<em>missing</em>") for v in values
        ]
        print(f"    {latex_param} & " + " & ".join(escaped_vals) + r" \\")

    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"  \caption{Comparison of configuration differences.}")
    print(r"  \label{tab:settings_diffs}")
    print(r"\end{table}")
