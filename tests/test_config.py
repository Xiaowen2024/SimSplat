from pathlib import Path

from sim_splat.config import load_scene_config, load_train_config


# tests/ lives directly under the project root.
ROOT = Path(__file__).resolve().parents[1]


def test_church_configs_load() -> None:
    scene = load_scene_config(ROOT / "configs/scenes/church.yaml")
    train = load_train_config(ROOT / "configs/methods/baseline.yaml")
    assert scene.name == "church"
    assert train.max_steps == 30_000
