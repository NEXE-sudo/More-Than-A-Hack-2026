import numpy as np
import pytest

from possession.data.validation import validate_predictions, validate_submission_file


def test_prediction_contract_requires_order_and_permitted_labels():
    validate_predictions(np.arange(3), np.array(["H", "D", "X"]), expected_frames=3)
    with pytest.raises(ValueError, match="ordered"):
        validate_predictions(np.array([0, 2, 1]), np.array(["H", "A", "D"]), 3)
    with pytest.raises(ValueError, match="invalid labels"):
        validate_predictions(np.arange(2), np.array(["H", "Z"]), 2)


def test_submission_file_requires_exact_header_rows_and_no_duplicate_frames(tmp_path):
    path = tmp_path / "submission.csv"
    path.write_text("frame,label\n0,H\n1,A\n", encoding="utf-8")
    validate_submission_file(path, expected_frames=2)
    path.write_text("frame,label\n0,H\n0,A\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        validate_submission_file(path, expected_frames=2)
    path.write_text("frame,prediction\n0,H\n1,A\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header"):
        validate_submission_file(path, expected_frames=2)
