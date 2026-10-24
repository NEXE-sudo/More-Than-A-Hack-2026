import pytest

from possession.config import load_config


def test_configuration_keeps_baseline_default_and_accepts_revised_strategies(tmp_path):
    default = load_config("configs/default.toml")
    assert default.model.evidence_model == "baseline"
    assert default.decoder.strategy == "baseline"

    path = tmp_path / "revised.toml"
    path.write_text(
        '[model]\nevidence_model = "revised"\ntagging_team_weight = 0.1\n'
        '[decoder]\nstrategy = "revised"\ncontested_tolerance_frames = 2\n',
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.model.evidence_model == "revised"
    assert config.model.tagging_team_weight == 0.1
    assert config.decoder.strategy == "revised"
    assert config.decoder.contested_tolerance_frames == 2


def test_configuration_rejects_unknown_model_or_decoder_strategy(tmp_path):
    path = tmp_path / "invalid.toml"
    path.write_text('[model]\nevidence_model = "other"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="evidence_model"):
        load_config(path)

    path.write_text('[decoder]\nstrategy = "other"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="decoder.strategy"):
        load_config(path)
