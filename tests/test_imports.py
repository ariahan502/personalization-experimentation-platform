from personalization_platform.pipeline.show_blueprint import load_config


def test_load_config_reads_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("run_name: smoke\n", encoding="utf-8")

    loaded = load_config(config_path)

    assert loaded["run_name"] == "smoke"
