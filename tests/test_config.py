import pytest

from amods.config import load_config, resolve_config, user_config_dir


def test_loads_packaged_default_stream_config(isolated_env):
    config = load_config("stream", "default")
    assert config["sr"] == 48000
    assert config["channels_in"] == 1


def test_loads_packaged_default_vad_config(isolated_env):
    config = load_config("vad", "none_source")
    assert config["vad_type"] == "none"


def test_name_without_yaml_extension_is_normalized(isolated_env):
    with_ext = load_config("vad", "none_source.yaml")
    without_ext = load_config("vad", "none_source")
    assert with_ext == without_ext


def test_invalid_extension_raises(isolated_env):
    with pytest.raises(ValueError):
        load_config("vad", "none_source.json")


def test_unknown_config_raises_file_not_found(isolated_env):
    with pytest.raises(FileNotFoundError):
        load_config("vad", "does_not_exist")


def test_explicit_path_is_used_as_is(isolated_env):
    custom_dir = isolated_env / "somewhere"
    (custom_dir / "vad").mkdir(parents=True)
    custom_path = custom_dir / "vad" / "custom.yaml"
    custom_path.write_text("vad_type: rms\ndb_threshold: -40.0\n")

    config = load_config("vad", str(custom_path))
    assert config == {"vad_type": "rms", "db_threshold": -40.0}


def test_local_configs_directory_takes_priority_over_packaged(isolated_env):
    local_dir = isolated_env / "configs" / "vad"
    local_dir.mkdir(parents=True)
    (local_dir / "none_source.yaml").write_text("vad_type: rms\ndb_threshold: -30.0\n")

    config = load_config("vad", "none_source")
    assert config == {"vad_type": "rms", "db_threshold": -30.0}


def test_user_config_dir_used_when_no_local_configs_directory(isolated_env):
    user_dir = user_config_dir() / "vad"
    user_dir.mkdir(parents=True)
    (user_dir / "none_source.yaml").write_text("vad_type: rms\ndb_threshold: -20.0\n")

    config = load_config("vad", "none_source")
    assert config == {"vad_type": "rms", "db_threshold": -20.0}


def test_search_dir_is_checked_before_cwd_and_user_dir(isolated_env):
    search_dir = isolated_env / "myconfigs"
    (search_dir / "vad").mkdir(parents=True)
    (search_dir / "vad" / "custom.yaml").write_text("vad_type: rms\ndb_threshold: -10.0\n")

    config = load_config("vad", "custom", search_dir=str(search_dir))
    assert config == {"vad_type": "rms", "db_threshold": -10.0}


def test_resolve_config_passes_through_a_dict_unchanged(isolated_env):
    inline = {"vad_type": "none"}
    resolved = resolve_config("vad", inline)
    assert resolved == inline


def test_resolve_config_returns_a_copy_not_the_same_dict(isolated_env):
    inline = {"vad_type": "none"}
    resolved = resolve_config("vad", inline)
    resolved["vad_type"] = "rms"
    assert inline["vad_type"] == "none"


def test_resolve_config_still_resolves_names_like_load_config(isolated_env):
    assert resolve_config("vad", "none_source") == load_config("vad", "none_source")
