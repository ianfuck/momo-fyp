import os

from backend.env_loader import _load_env_file


def test_load_env_file_sets_missing_values_only(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("HF_TOKEN=abc123\nexport MOMO_MODE=test\n", encoding="utf-8")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("MOMO_MODE", "existing")

    _load_env_file(env_file)

    assert os.environ["HF_TOKEN"] == "abc123"
    assert os.environ["MOMO_MODE"] == "existing"
