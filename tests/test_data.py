from __future__ import annotations

import numpy as np

from keyed_gram.data import TokenEpochSampler, TokenStreamPool, iter_eval_batches


def test_fraction_uses_a_fixed_seeded_token_subset(tmp_path):
    path = tmp_path / "topic_train.bin"
    np.arange(100, dtype=np.uint16).tofile(path)

    pool = TokenStreamPool((path,), context_length=4, fraction=0.25, seed=7)
    duplicate = TokenStreamPool((path,), context_length=4, fraction=0.25, seed=7)
    assert pool.subset_lengths.tolist() == [25]
    assert pool.subset_starts.tolist() == duplicate.subset_starts.tolist()
    assert pool.num_sequences == 6

    x, y = pool.sample_batch(64, np.random.default_rng(9))
    lower = int(pool.subset_starts[0])
    upper = lower + int(pool.subset_lengths[0])
    assert int(x.min()) >= lower
    assert int(y.max()) < upper

    epoch_x, _ = TokenEpochSampler(pool, np.random.default_rng(11)).sample_batch(
        pool.num_sequences
    )
    assert sorted(epoch_x[:, 0].tolist()) == [lower + 4 * index for index in range(6)]


def test_evaluation_uses_every_non_overlapping_window_once(tmp_path):
    path = tmp_path / "topic_test.bin"
    np.arange(17, dtype=np.uint16).tofile(path)
    batches = list(iter_eval_batches(path, 4, 100, 3, seed=1))
    starts = [int(row[0]) for x, _ in batches for row in x]
    assert starts == [0, 4, 8, 12]
