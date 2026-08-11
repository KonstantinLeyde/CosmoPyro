from IPython.display import HTML, Markdown, display
from rich import print
from rich.table import Table

__all__ = [
    "add_rows_recursively",
    "display_differences",
    "print_summary_in_notebook",
]


def print_summary_in_notebook(data):

    priors = data["kwargs_priors"]
    prior_ranges = []
    fixed_parameters = []
    for cat, params in priors.items():
        if params is None:
            continue
        if isinstance(params, dict):
            for param, v in params.items():
                if isinstance(v, dict) and "min" in v and "max" in v:
                    dist_type = v.get("dist_type", "Uniform")
                    prior_ranges.append(
                        (f"{cat}.{param}", v["min"], v["max"], dist_type)
                    )
                elif isinstance(v, dict) and v.get("dist_type") == "Delta":
                    fixed_parameters.append((f"{cat}.{param}", v["value"]))

    field_parameters = data.get("kwargs_field", {})
    catalog_metadata = data.get("catalog_metadata", {})

    display(Markdown("## Summary of Key Parameters"))

    table = Table(title="Prior Ranges", show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Min", justify="right", style="red")
    table.add_column("Max", justify="right", style="green")
    table.add_column("Type", style="yellow")

    for param, min_val, max_val, dist_type in prior_ranges:
        table.add_row(param, str(min_val), str(max_val), dist_type)

    print(table)

    if fixed_parameters:
        fixed_table = Table(
            title="Fixed Parameters", show_header=True, header_style="bold magenta"
        )
        fixed_table.add_column("Parameter", style="cyan", no_wrap=True)
        fixed_table.add_column("Value", justify="right", style="red")

        for param, value in fixed_parameters:
            fixed_table.add_row(param, str(value))

        print(fixed_table)

    if field_parameters:
        field_table = Table(
            title="Field Parameters", show_header=True, header_style="bold magenta"
        )
        field_table.add_column("Parameter", style="cyan", no_wrap=True)
        field_table.add_column("Value", justify="right", style="red")

        add_rows_recursively(field_table, field_parameters)

        print(field_table)

    if catalog_metadata:
        catalog_table = Table(
            title="Catalog Metadata", show_header=True, header_style="bold magenta"
        )
        catalog_table.add_column("Key", style="cyan", no_wrap=True)
        catalog_table.add_column(
            "Value", justify="left", style="red"
        )  # Changed justify to 'left' for better readability of potentially longer values

        add_rows_recursively(catalog_table, catalog_metadata)
        print(catalog_table)


def add_rows_recursively(table, d, parent_key=""):
    for k, v in d.items():
        full_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            add_rows_recursively(table, v, full_key)
        else:
            table.add_row(full_key, str(v))  # Ensure value is string for table


def display_differences(diffs):
    """Render a list of setting differences as a styled HTML table in Jupyter.

    Parameters
    ----------
    diffs : list of dict
        Output of :func:`~cosmopyro.utils.analysis_utils.find_differences`.
        Each entry has keys ``parameter``, ``values``, and ``labels``.
    """
    if not diffs:
        display(Markdown("**No differences found.**"))
        return

    labels = diffs[0]["labels"]

    # Build colour palette: one hue per unique value within a row.
    # Same values get the same colour so differences pop out.
    palette = [
        "#e8f5e9",  # green
        "#fff3e0",  # orange
        "#e3f2fd",  # blue
        "#fce4ec",  # pink
        "#f3e5f5",  # purple
        "#e0f7fa",  # teal
        "#fff9c4",  # yellow
        "#efebe9",  # brown
    ]

    header = "".join(
        f'<th style="padding:6px 12px;text-align:center;">{label}</th>'
        for label in labels
    )

    rows = []
    for d in diffs:
        if d["parameter"] in ["run_kwargs.job_id", "run_kwargs.path_kwargs", "job_id"]:
            continue

        vals = d["values"]
        strs = [str(v) if v is not None else "<em>missing</em>" for v in vals]

        # Assign colours: the majority value always gets the first palette
        # colour, independent of where it appears in the row.
        unique = list(dict.fromkeys(strs))
        majority_value = d.get("majority_value", unique[0])
        ordered_unique = [majority_value] + [
            value for value in unique if value != majority_value
        ]
        colour_map = {
            s: palette[i % len(palette)] for i, s in enumerate(ordered_unique)
        }

        cells = "".join(
            f'<td style="padding:6px 12px;text-align:center;background:{colour_map[s]};color:#000;">{s}</td>'
            for s in strs
        )
        rows.append(
            f'<tr><td style="padding:6px 12px;font-family:monospace;white-space:nowrap;">'
            f"{d['parameter']}</td>{cells}</tr>"
        )

    html = f"""
    <table style="border-collapse:collapse;font-size:0.9em;margin:0.5em 0;">
    <thead><tr style="border-bottom:2px solid #333;">
      <th style="padding:6px 12px;text-align:left;">Parameter</th>{header}
    </tr></thead>
    <tbody>{"".join(rows)}</tbody>
    </table>"""

    display(HTML(html))
