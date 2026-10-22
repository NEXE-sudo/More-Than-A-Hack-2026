"""Homography-based image projection for diagnostics and optional appearance fusion."""

import numpy as np


def project_to_image(
    matrices: np.ndarray, positions: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project normalized pitch points to pixels only where homography rows are valid."""
    transforms = np.asarray(matrices, dtype=np.float64)
    points = np.asarray(positions, dtype=np.float64)
    usable = np.asarray(valid, dtype=bool)
    if transforms.ndim != 3 or transforms.shape[1:] != (3, 3):
        raise ValueError("matrices must have shape (frames, 3, 3)")
    if points.shape != (len(transforms), 2) or usable.shape != (len(transforms),):
        raise ValueError("positions and valid mask must align with homography frames")
    output = np.full((len(transforms), 2), np.nan, dtype=np.float64)
    finite = usable & np.isfinite(points).all(axis=1) & np.isfinite(transforms).all(axis=(1, 2))
    if finite.any():
        homogeneous = np.column_stack((points[finite], np.ones(finite.sum())))
        projected = np.einsum("nij,nj->ni", transforms[finite], homogeneous)
        denominators = projected[:, 2]
        stable = np.abs(denominators) > 1e-12
        selected = np.flatnonzero(finite)[stable]
        output[selected] = projected[stable, :2] / denominators[stable, None]
        finite[selected[~stable]] = False
    return output, finite
