"""Regression tests for #3098: FileAgentStore.get() must not raise a spurious
"Agent config not found" when a concurrent update() commits config.yaml via an
atomic temp-file + os.replace (which unlinks the old path for an instant)."""

from __future__ import annotations

import builtins
import os
import threading
from pathlib import Path

import pytest

from deerflow.persistence.agents.file import FileAgentStore


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """Root the file store at a temp DEER_FLOW_HOME with one seeded agent."""
    monkeypatch.setenv("DEER_FLOW_HOME", str(tmp_path))
    from deerflow.config import paths as paths_module

    monkeypatch.setattr(paths_module, "_paths", None)
    fs = FileAgentStore()
    fs.create(
        "efficiency-analysis",
        {"name": "efficiency-analysis", "description": "analyses efficiency"},
        "efficiency soul",
        user_id="u1",
    )
    return fs


def test_get_returns_config_when_present(store):
    config = store.get("efficiency-analysis", user_id="u1")
    assert config.name == "efficiency-analysis"
    assert config.description == "analyses efficiency"


def test_get_raises_directory_not_found_for_unknown_agent(store):
    with pytest.raises(FileNotFoundError, match="Agent directory not found"):
        store.get("does-not-exist", user_id="u1")


def test_get_raises_config_not_found_when_dir_lacks_config(store):
    # A directory that exists but holds no config.yaml must still raise exactly
    # as before the fix (no concurrent write in flight).
    store.create("orphan", {"name": "orphan"}, "soul", user_id="u1")
    from deerflow.config import agents_config as _ac

    cfg_dir = _ac.get_paths().user_agent_dir("u1", "orphan")
    (cfg_dir / "config.yaml").unlink()
    with pytest.raises(FileNotFoundError, match="Agent config not found"):
        store.get("orphan", user_id="u1")


def test_get_tolerates_open_losing_race_to_rename(store, monkeypatch):
    """When open() loses the race to an atomic replace, get() must retry and
    succeed rather than raising "Agent config not found"."""
    real_open = builtins.open
    calls = {"n": 0}

    def fake_open(path, *args, **kwargs):
        if Path(path).name == "config.yaml" and calls["n"] == 0:
            calls["n"] += 1
            raise FileNotFoundError(path)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)
    config = store.get("efficiency-analysis", user_id="u1")
    assert config.name == "efficiency-analysis"
    assert calls["n"] == 1


def test_get_reprobes_layouts_when_resolver_fell_back(store, monkeypatch):
    """If resolve_agent_dir temporarily fell back to the legacy layout (e.g.
    because the per-user config.yaml was momentarily absent during a replace),
    get() must still find the per-user config via re-probe."""
    import deerflow.persistence.agents.file as file_mod
    from deerflow.config import agents_config as _ac

    legacy = _ac.get_paths().agent_dir("efficiency-analysis")
    monkeypatch.setattr(
        file_mod,
        "resolve_agent_dir",
        lambda name, *, user_id=None: legacy,
    )
    # Per-user config still exists; resolver points at legacy.
    config = store.get("efficiency-analysis", user_id="u1")
    assert config.name == "efficiency-analysis"


def test_get_survives_concurrent_atomic_replace(store):
    """Reproduce #3098: a reader calling get() while another thread commits
    config.yaml via atomic os.replace must not see a spurious not-found."""
    from deerflow.config import agents_config as _ac

    cfg_dir = _ac.get_paths().user_agent_dir("u1", "efficiency-analysis")
    config_file = cfg_dir / "config.yaml"

    stop = threading.Event()
    errors: list[str] = []

    def writer() -> None:
        i = 0
        while not stop.is_set():
            tmp = cfg_dir / f"config.yaml.tmp{i % 2}"
            tmp.write_text(
                "name: efficiency-analysis\ndescription: analyses efficiency\n",
                encoding="utf-8",
            )
            os.replace(tmp, config_file)
            i += 1

    def reader() -> None:
        while not stop.is_set():
            try:
                store.get("efficiency-analysis", user_id="u1")
            except FileNotFoundError as e:
                errors.append(str(e))

    w = threading.Thread(target=writer)
    r = threading.Thread(target=reader)
    w.start()
    r.start()
    r.join(timeout=2.0)
    stop.set()
    w.join(timeout=2.0)
    assert not errors, (
        f"get() raised {len(errors)} spurious not-found error(s): {errors[:3]}"
    )
