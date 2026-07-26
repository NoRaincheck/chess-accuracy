import numpy as np

from chess_accuracy.batch_inference import _compute_score


class TestComputeScore:
    def test_all_correct(self):
        n_pos, n_elo, vocab = 10, 3, 100
        logits = np.random.randn(n_pos, n_elo, vocab)
        human_moves = np.array([5] * n_pos)
        # Set human move logits very high
        for e in range(n_elo):
            logits[:, e, human_moves] = 100.0
        scores = _compute_score(logits, human_moves, n_pos, alpha=0.6)
        assert scores.shape == (n_elo,)
        # The rank function uses >= which counts the human move itself,
        # so perfect predictions get rank=2, MRR=0.5.
        # score = 0.6 * 1.0 + 0.4 * 0.5 = 0.8
        np.testing.assert_allclose(scores, 0.8, atol=1e-6)

    def test_all_wrong(self):
        n_pos, n_elo, vocab = 10, 3, 100
        logits = np.random.randn(n_pos, n_elo, vocab)
        human_moves = np.array([0] * n_pos)
        # Set human move logits very low, all others high
        for e in range(n_elo):
            logits[:, e, 0] = -100.0
            logits[:, e, 1:] = 100.0
        scores = _compute_score(logits, human_moves, n_pos, alpha=0.6)
        assert scores.shape == (n_elo,)
        assert np.all(scores < 0.1)

    def test_zero_positions(self):
        logits = np.zeros((0, 3, 100))
        human_moves = np.array([], dtype=np.int64)
        scores = _compute_score(logits, human_moves, 0, alpha=0.6)
        assert scores.shape == (3,)
        np.testing.assert_allclose(scores, 0.0)

    def test_alpha_one_pure_top1(self):
        n_pos, n_elo, vocab = 5, 2, 50
        logits = np.random.randn(n_pos, n_elo, vocab)
        human_moves = np.random.randint(0, vocab, size=n_pos)
        scores = _compute_score(logits, human_moves, n_pos, alpha=1.0)
        # alpha=1.0 means pure top-1 accuracy, no MRR
        top1_moves = logits.argmax(axis=2)
        top1_acc = (top1_moves == human_moves[:, None]).mean(axis=0)
        np.testing.assert_allclose(scores, top1_acc, atol=1e-10)

    def test_alpha_zero_pure_mrr(self):
        n_pos, n_elo, vocab = 5, 2, 50
        logits = np.random.randn(n_pos, n_elo, vocab)
        human_moves = np.random.randint(0, vocab, size=n_pos)
        scores = _compute_score(logits, human_moves, n_pos, alpha=0.0)
        # alpha=0.0 means pure MRR
        pos_idx = np.arange(n_pos)[:, None]
        elo_idx = np.arange(n_elo)[None, :]
        human_logits = logits[pos_idx, elo_idx, human_moves[:, None]]
        rank = (logits >= human_logits[:, :, None]).sum(axis=2) + 1
        mrr = (1.0 / rank).mean(axis=0)
        np.testing.assert_allclose(scores, mrr, atol=1e-10)

    def test_single_position(self):
        n_elo, vocab = 4, 200
        logits = np.random.randn(1, n_elo, vocab)
        human_moves = np.array([42])
        scores = _compute_score(logits, human_moves, 1, alpha=0.6)
        assert scores.shape == (n_elo,)
        assert np.all((scores >= 0) & (scores <= 1))
