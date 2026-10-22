import pytest

from possession.evaluation.official_adapter import run_official_evaluator


def test_official_evaluator_adapter_refuses_private_ground_truth():
    with pytest.raises(ValueError, match="private ground truth"):
        run_official_evaluator(".", ".", ".", ground_truth="PRIVATE")
