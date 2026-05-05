"""Deterministic 11/4/4 partition over the benchmark corpus.

The partition function is the seed→corpus contract that every multi-seed
artifact relies on: ``--seeds N`` runs, the empirical-holdout gate, the
probe-grounded sensing prompt, and the cross-domain script all consume
its output. Determinism, disjointness, and the security-corpus overlap
flag are pinned here so changes to the hashing strategy can't silently
re-shape any of them.
"""

from __future__ import annotations

from stem_agent.evaluation.fixtures.code_samples import (
    CorpusPartition,
    get_benchmark_corpus,
    partition,
)
from stem_agent.evaluation.fixtures.security_audit_samples import (
    get_security_audit_corpus,
)


def _ids(samples: list) -> list[str]:
    return [s.sample_id for s in samples]


class TestPartitionDeterminism:
    def test_same_seed_same_corpus_returns_same_slices(self) -> None:
        corpus = get_benchmark_corpus()
        a = partition(corpus, seed=0)
        b = partition(corpus, seed=0)
        assert _ids(a.validation) == _ids(b.validation)
        assert _ids(a.holdout) == _ids(b.holdout)
        assert _ids(a.probe) == _ids(b.probe)

    def test_different_seeds_differ(self) -> None:
        corpus = get_benchmark_corpus()
        a = partition(corpus, seed=0)
        b = partition(corpus, seed=1)
        assert _ids(a.validation) != _ids(b.validation)

    def test_input_ordering_does_not_change_partition(self) -> None:
        """Sorting by sample_id internally → order-independent input."""
        corpus = get_benchmark_corpus()
        reversed_corpus = list(reversed(corpus))
        a = partition(corpus, seed=0)
        b = partition(reversed_corpus, seed=0)
        assert _ids(a.validation) == _ids(b.validation)
        assert _ids(a.holdout) == _ids(b.holdout)
        assert _ids(a.probe) == _ids(b.probe)


class TestPartitionShape:
    def test_default_corpus_partition_sizes(self) -> None:
        corpus = get_benchmark_corpus()
        p = partition(corpus, seed=0)
        assert len(p.validation) == 11
        assert len(p.holdout) == 4
        assert len(p.probe) == 4

    def test_default_corpus_returns_dataclass(self) -> None:
        p = partition(get_benchmark_corpus(), seed=0)
        assert isinstance(p, CorpusPartition)
        assert p.overlapping is False


class TestPartitionDisjointness:
    def test_pairwise_disjoint_for_default_corpus(self) -> None:
        p = partition(get_benchmark_corpus(), seed=0)
        v = set(_ids(p.validation))
        h = set(_ids(p.holdout))
        pr = set(_ids(p.probe))
        assert v.isdisjoint(h)
        assert v.isdisjoint(pr)
        assert h.isdisjoint(pr)

    def test_every_sample_lands_in_exactly_one_slice(self) -> None:
        corpus = get_benchmark_corpus()
        p = partition(corpus, seed=0)
        all_ids = set(_ids(p.validation + p.holdout + p.probe))
        assert all_ids == {s.sample_id for s in corpus}

    def test_probe_disjoint_across_seeds(self) -> None:
        """Per-seed probes don't have to differ; but they shouldn't be identical
        for adjacent seeds, or we're shipping the same probe sample to every run."""
        a = partition(get_benchmark_corpus(), seed=0)
        b = partition(get_benchmark_corpus(), seed=1)
        assert set(_ids(a.probe)) != set(_ids(b.probe))


class TestSecurityCorpusOverlap:
    def test_security_corpus_partition_flags_overlap(self) -> None:
        p = partition(get_security_audit_corpus(), seed=0)
        assert p.overlapping is True

    def test_security_corpus_yields_non_empty_slices(self) -> None:
        p = partition(get_security_audit_corpus(), seed=0)
        assert p.validation
        assert p.holdout
        assert p.probe

    def test_security_corpus_overlap_is_deterministic(self) -> None:
        a = partition(get_security_audit_corpus(), seed=2)
        b = partition(get_security_audit_corpus(), seed=2)
        assert _ids(a.validation) == _ids(b.validation)
        assert _ids(a.holdout) == _ids(b.holdout)
        assert _ids(a.probe) == _ids(b.probe)
