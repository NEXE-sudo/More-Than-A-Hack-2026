import numpy as np

from possession.features.geometry import project_to_image


def test_homography_projection_honors_validity_and_missing_points():
    matrices = np.repeat(np.eye(3)[None, :, :], 3, axis=0)
    positions = np.array([[0.2, 0.4], [0.5, 0.5], [np.nan, np.nan]])
    projected, valid = project_to_image(matrices, positions, np.array([True, False, True]))
    assert projected[0].tolist() == [0.2, 0.4]
    assert valid.tolist() == [True, False, False]
    assert np.isnan(projected[1:]).all()
