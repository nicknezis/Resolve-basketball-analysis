"""Two-team clustering on embeddings (SigLIP itself is not loaded in tests)."""

from __future__ import annotations

import numpy as np

from src.analysis.team_classifier import cluster_two_teams


def _blob(center, n, rng, spread=0.05, dim=16):
    return center + spread * rng.standard_normal((n, dim))


def test_two_clusters_and_outlier():
    rng = np.random.default_rng(0)
    a = np.zeros(16); a[0] = 1.0
    b = np.zeros(16); b[1] = 1.0
    ref = np.zeros(16); ref[2] = 3.0  # a referee far from both
    feats = np.vstack([_blob(a, 6, rng), _blob(b, 6, rng), ref[None, :]])
    labels, info = cluster_two_teams(feats, outlier_std=1.5)
    assert set(labels[:6]) == {labels[0]} and set(labels[6:12]) == {labels[6]}
    assert labels[0] != labels[6] and labels[0] >= 0 and labels[6] >= 0
    assert labels[12] == -1
    assert info["separation_ratio"] > 3


def test_too_few_tracks():
    labels, info = cluster_two_teams(np.zeros((1, 8)))
    assert list(labels) == [-1]
