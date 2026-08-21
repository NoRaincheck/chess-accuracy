import numpy as np
import pytest

from chess_accuracy.batch_inference import (
    bicubic_upsample_surface,
    elo_log_prior,
    joint_posterior_2d,
)


def quadratic_surface(w_vals, b_vals):
    """Smooth synthetic log-likelihood surface with a known peak."""
    w = np.asarray(w_vals, dtype=np.float64)[:, np.newaxis]
    b = np.asarray(b_vals, dtype=np.float64)[np.newaxis, :]
    return -(
        ((w - 1900.0) / 700.0) ** 2
        + ((b - 2400.0) / 600.0) ** 2
        + 0.3 * ((w - 1500.0) / 700.0) * ((b - 2000.0) / 600.0)
    )


class TestBicubicUpsampleSurface:
    def test_output_shape(self):
        src = np.zeros((6, 6))
        out = bicubic_upsample_surface(src, (55, 55))
        assert out.shape == (55, 55)

    def test_corners_preserved(self):
        src = np.arange(36, dtype=np.float64).reshape(6, 6)
        out = bicubic_upsample_surface(src, (21, 21))
        np.testing.assert_allclose(out[0, 0], src[0, 0], atol=1e-5)
        np.testing.assert_allclose(out[0, -1], src[0, -1], atol=1e-5)
        np.testing.assert_allclose(out[-1, 0], src[-1, 0], atol=1e-5)
        np.testing.assert_allclose(out[-1, -1], src[-1, -1], atol=1e-5)

    def test_quadratic_peak_location_recovered(self):
        """Anchor-grid + upsample must localize the true peak of a smooth surface.

        The contract is localization (stage B re-centers and re-evaluates the
        real model around these modes), so sub-anchor-cell accuracy is enough.
        """
        anchor_vals = np.linspace(300.0, 3000.0, 6)
        dense_vals = np.linspace(300.0, 3000.0, 55)
        anchor_step = anchor_vals[1] - anchor_vals[0]

        # Ground truth on the dense grid
        truth = quadratic_surface(dense_vals, dense_vals)
        ti, tj = np.unravel_index(np.argmax(truth), truth.shape)

        # Anchors -> upsample -> argmax
        anchors = quadratic_surface(anchor_vals, anchor_vals)
        dense = bicubic_upsample_surface(anchors, (len(dense_vals), len(dense_vals)))
        di, dj = np.unravel_index(np.argmax(dense), dense.shape)

        assert abs(dense_vals[di] - dense_vals[ti]) <= anchor_step / 2
        assert abs(dense_vals[dj] - dense_vals[tj]) <= anchor_step / 2

    def test_bilinear_surface_reproduced_exactly_at_anchors(self):
        src = np.array([[0.0, 2.0], [4.0, 6.0]])
        out = bicubic_upsample_surface(src, (3, 3))
        # align_corners=True maps source anchors to output corners
        np.testing.assert_allclose(out[::2, ::2], src, atol=1e-6)

    def test_finite_output(self):
        rng = np.random.default_rng(0)
        src = rng.standard_normal((5, 7))
        out = bicubic_upsample_surface(src, (31, 41))
        assert np.all(np.isfinite(out))


class TestJointPosterior2d:
    def test_normalized_and_peaked_near_prior_weighted_mode(self):
        vals = np.linspace(300.0, 3000.0, 28)
        surface = quadratic_surface(vals, vals)
        post = joint_posterior_2d(surface, vals, vals, prior_mean=1500.0, prior_std=350.0)
        assert post.sum() == pytest.approx(1.0)
        assert post.min() >= 0.0
        # Prior pulls the marginal mode below the raw surface peak (1900/2400)
        wi = int(np.argmax(post.sum(axis=1)))
        bi = int(np.argmax(post.sum(axis=0)))
        assert vals[wi] < 1900
        assert vals[bi] < 2400


class TestEloLogPrior:
    def test_zero_at_mean(self):
        assert elo_log_prior(np.array([1500.0]), 1500.0, 350.0)[0] == pytest.approx(0.0)

    def test_symmetric(self):
        p = elo_log_prior(np.array([1000.0, 2000.0]), 1500.0, 350.0)
        assert p[0] == pytest.approx(p[1])
