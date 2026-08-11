import yaml

from cosmopyro.utils.analysis_utils import find_differences


def _write_settings(path, value):
    path.mkdir()
    settings_path = path / "kwargs_analysis.yaml"
    with open(settings_path, "w") as f:
        yaml.safe_dump({"nested": {"setting": value}}, f)
    return settings_path


def test_find_differences_reports_majority_value(tmp_path):
    paths = [
        _write_settings(tmp_path / "id_1", "minority"),
        _write_settings(tmp_path / "id_2", "majority"),
        _write_settings(tmp_path / "id_3", "majority"),
    ]

    diffs = find_differences([str(path) for path in paths])

    assert diffs == [
        {
            "parameter": "nested.setting",
            "values": ["minority", "majority", "majority"],
            "labels": ["id_1", "id_2", "id_3"],
            "majority_value": "majority",
        }
    ]
