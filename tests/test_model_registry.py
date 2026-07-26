import pytest

from chess_accuracy.maia3.model_registry import (
    ModelResolutionError,
    format_model_list,
    parse_huggingface_url,
    resolve_model_spec,
)


class TestResolveModelSpec:
    def test_alias_5m(self):
        spec = resolve_model_spec("5m")
        assert spec.name == "maia3-5m"

    def test_alias_23m(self):
        spec = resolve_model_spec("23m")
        assert spec.name == "maia3-23m"

    def test_alias_79m(self):
        spec = resolve_model_spec("79m")
        assert spec.name == "maia3-79m"

    def test_full_name(self):
        spec = resolve_model_spec("maia3-5m")
        assert spec.name == "maia3-5m"

    def test_empty_raises(self):
        with pytest.raises(ModelResolutionError):
            resolve_model_spec("")

    def test_unknown_raises(self):
        with pytest.raises(ModelResolutionError):
            resolve_model_spec("nonexistent-model-xyz")

    def test_hf_repo_id(self):
        spec = resolve_model_spec("UofTCSSLab/Maia3-5M")
        assert spec.repo_id == "UofTCSSLab/Maia3-5M"

    def test_url_parsing(self):
        spec = resolve_model_spec("https://huggingface.co/UofTCSSLab/Maia3-5M")
        assert spec.repo_id == "UofTCSSLab/Maia3-5M"

    def test_spec_has_config(self):
        spec = resolve_model_spec("maia3-5m")
        assert "dim_vit" in spec.config
        assert "num_heads" in spec.config


class TestParseHuggingfaceUrl:
    def test_blob_url(self):
        result = parse_huggingface_url("https://huggingface.co/UofTCSSLab/Maia3-5M/blob/main/model.pt")
        assert result is not None
        repo_id, revision, filename = result
        assert repo_id == "UofTCSSLab/Maia3-5M"
        assert revision == "main"
        assert filename == "model.pt"

    def test_tree_url(self):
        result = parse_huggingface_url("https://huggingface.co/UofTCSSLab/Maia3-5M/tree/main")
        assert result is not None
        repo_id, revision, filename = result
        assert repo_id == "UofTCSSLab/Maia3-5M"
        assert revision == "main"
        assert filename is None

    def test_non_hf_url(self):
        result = parse_huggingface_url("https://example.com/something")
        assert result is None

    def test_minimal_url(self):
        result = parse_huggingface_url("https://huggingface.co/UofTCSSLab/Maia3-5M")
        assert result is not None
        repo_id, revision, filename = result
        assert repo_id == "UofTCSSLab/Maia3-5M"
        assert revision is None
        assert filename is None


class TestFormatModelList:
    def test_returns_string(self):
        result = format_model_list()
        assert isinstance(result, str)

    def test_contains_model_names(self):
        result = format_model_list()
        assert "maia3-5m" in result
        assert "maia3-23m" in result
        assert "maia3-79m" in result
