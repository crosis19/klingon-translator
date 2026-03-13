"""Tests for training.colab_utils module."""

import json

import pytest

from klingon_translator.training.colab_utils import copy_data_to_local_ssd, load_jsonl


class TestLoadJsonl:
    """Test JSONL file loading."""

    def test_loads_valid_jsonl(self, tmp_path):
        path = tmp_path / "test.jsonl"
        data = [{"en": "hello", "tlh": "nuqneH"}, {"en": "yes", "tlh": "HIja"}]
        text = chr(10).join(json.dumps(d) for d in data)
        path.write_text(text, encoding="utf-8")
        result = load_jsonl(path)
        assert len(result) == 2
        assert result[0]["en"] == "hello"

    def test_skips_empty_lines(self, tmp_path):
        path = tmp_path / "test.jsonl"
        lines = [
            json.dumps({"en": "hello", "tlh": "nuqneH"}),
            "",
            json.dumps({"en": "yes", "tlh": "HIja"}),
            "",
        ]
        path.write_text(chr(10).join(lines), encoding="utf-8")
        result = load_jsonl(path)
        assert len(result) == 2

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_jsonl("/nonexistent/path.jsonl")


class TestCopyDataToLocalSSD:
    """Test data copy to local SSD."""

    def test_copies_jsonl_files(self, tmp_path):
        project = tmp_path / "project"
        processed = project / "data" / "processed"
        processed.mkdir(parents=True)
        (processed / "train.jsonl").write_text(
            json.dumps({"en": "hi", "tlh": "nuqneH"}), encoding="utf-8"
        )
        (processed / "val.jsonl").write_text(
            json.dumps({"en": "yes", "tlh": "HIja"}), encoding="utf-8"
        )
        raw = project / "data" / "raw"
        raw.mkdir(parents=True)
        local = tmp_path / "local"
        data_dir, raw_dir = copy_data_to_local_ssd(project, local)
        assert (data_dir / "train.jsonl").exists()
        assert (data_dir / "val.jsonl").exists()

    def test_creates_directories(self, tmp_path):
        project = tmp_path / "project"
        (project / "data" / "processed").mkdir(parents=True)
        (project / "data" / "raw").mkdir(parents=True)
        local = tmp_path / "local"
        data_dir, raw_dir = copy_data_to_local_ssd(project, local)
        assert data_dir.exists()
        assert raw_dir.exists()

    def test_handles_missing_raw_files(self, tmp_path):
        """Should not crash when optional raw files are absent."""
        project = tmp_path / "project"
        (project / "data" / "processed").mkdir(parents=True)
        (project / "data" / "raw").mkdir(parents=True)
        local = tmp_path / "local"
        data_dir, raw_dir = copy_data_to_local_ssd(project, local)
        assert data_dir.exists()
