"""
Tests for creep_model.eda.stage_classification.

Every test here uses `strain_from_plateau_lengths` to construct a strain
series with an EXACTLY known plateau structure, so expected primary_end_idx
/ secondary_end_idx values can be computed by hand and asserted precisely --
see the worked derivations in each test's docstring.
"""
import pytest

from creep_model.eda.stage_classification import _find_plateaus, classify_stages


# ---------------------------------------------------------------------------
# _find_plateaus
# ---------------------------------------------------------------------------

class TestFindPlateaus:
    def test_single_plateau(self, strain_from_plateau_lengths):
        strain = strain_from_plateau_lengths([5])
        plateaus = _find_plateaus(strain)
        assert len(plateaus) == 1
        assert plateaus[0].n_points == 5
        assert plateaus[0].start_idx == 0
        assert plateaus[0].end_idx == 4

    def test_no_repeats(self, strain_from_plateau_lengths):
        strain = strain_from_plateau_lengths([1, 1, 1, 1])
        plateaus = _find_plateaus(strain)
        assert len(plateaus) == 4
        assert all(p.n_points == 1 for p in plateaus)

    def test_mixed_lengths_exact_indices(self, strain_from_plateau_lengths):
        strain = strain_from_plateau_lengths([1, 2, 3, 4])
        plateaus = _find_plateaus(strain)
        assert [p.n_points for p in plateaus] == [1, 2, 3, 4]
        assert [p.start_idx for p in plateaus] == [0, 1, 3, 6]
        assert [p.end_idx for p in plateaus] == [0, 2, 5, 9]


# ---------------------------------------------------------------------------
# classify_stages -- primary creep only (no secondary/tertiary detected)
# ---------------------------------------------------------------------------

class TestClassifyStagesPrimaryOnly:
    def test_strictly_increasing_never_triggers_secondary(
        self, make_test, strain_from_plateau_lengths
    ):
        """
        Plateau lengths [1,2,3,4,5]: even after excluding the final plateau,
        [1,2,3,4] is strictly increasing, so no k1-run of non-increasing
        lengths ever appears. The test ends while still (apparently) in
        primary creep -- both boundaries should be None.
        """
        strain = strain_from_plateau_lengths([1, 2, 3, 4, 5])
        test = make_test(strain_series=strain)

        result = classify_stages(test, k1=2, k2=2)

        assert result.primary_end_idx is None
        assert result.secondary_end_idx is None
        assert result.has_tertiary is False

    def test_empty_test_raises(self, make_test):
        test = make_test(strain_series=[])
        with pytest.raises(ValueError, match="empty test"):
            classify_stages(test, k1=2, k2=2)

    def test_k1_larger_than_available_plateaus_returns_none(
        self, make_test, strain_from_plateau_lengths
    ):
        """k1 bigger than the number of classifiable plateaus should return
        cleanly rather than raise (range() over a negative span is empty)."""
        strain = strain_from_plateau_lengths([1, 2, 3])
        test = make_test(strain_series=strain)

        result = classify_stages(test, k1=100, k2=2)

        assert result.primary_end_idx is None
        assert result.secondary_end_idx is None


# ---------------------------------------------------------------------------
# classify_stages -- primary -> secondary, no tertiary
# ---------------------------------------------------------------------------

class TestClassifyStagesSecondaryOnly:
    def test_secondary_onset_no_tertiary(self, make_test, strain_from_plateau_lengths):
        """
        Plateau lengths: [1, 2, 3, 5, 5, 5, 5, 5], k1=3.
        Final plateau (idx 7, len 5) excluded -> classified over
        [1, 2, 3, 5, 5, 5, 5] (7 plateaus, local indices 0-6).

        Secondary onset: first i where n_points[i] >= n_points[i+1] for
        both steps of a 3-run. That's i=3 (values 5,5,5 -- flat, and 5>=5
        holds for both comparisons). No earlier i qualifies since
        1<2<3<5 is strictly increasing.

        => primary_end_idx = end_idx of local plateau 2 (the len-3 plateau)
        => no k2=2 run of STRICTLY decreasing lengths exists anywhere
           afterwards (it's flat, not decreasing) -> no tertiary
        => secondary_end_idx = end_idx of the last classified plateau
           (local index 6, i.e. the last of the flat 5s before the excluded
           final plateau)

        Index arithmetic (cumulative plateau lengths):
            plateau0: len1 -> end_idx 0
            plateau1: len2 -> end_idx 2
            plateau2: len3 -> end_idx 5      <- expected primary_end_idx
            plateau3: len5 -> end_idx 10
            plateau4: len5 -> end_idx 15
            plateau5: len5 -> end_idx 20
            plateau6: len5 -> end_idx 25     <- expected secondary_end_idx
            plateau7 (final, excluded): len5 -> end_idx 30
        """
        strain = strain_from_plateau_lengths([1, 2, 3, 5, 5, 5, 5, 5])
        test = make_test(strain_series=strain)

        result = classify_stages(test, k1=3, k2=2)

        assert result.primary_end_idx == 5
        assert result.secondary_end_idx == 25
        assert result.has_tertiary is False


# ---------------------------------------------------------------------------
# classify_stages -- primary -> secondary -> tertiary
# ---------------------------------------------------------------------------

class TestClassifyStagesTertiary:
    def test_full_three_stage_detection(self, make_test, strain_from_plateau_lengths):
        """
        Tertiary detection is a THRESHOLD check against primary_end_length,
        not a pairwise strictly-decreasing check. Plateau lengths:
        [1, 2, 5, 6, 6, 6, 3, 3, 3, 1], k1=2, k2=2.

        Final plateau (len 1) excluded -> classified over
        [1, 2, 5, 6, 6, 6, 3, 3, 3] (9 plateaus, local indices 0-8).

        Secondary onset (k1=2): first i where n_points[i] >= n_points[i+1].
        1<2<5<6, then 6>=6 at i=3 (plateaus 3,4). => secondary_start=3
        => primary_end_plateau_idx=2 (len 5) => primary_end_length=5

        Tertiary onset (k2=2): search from i = 3+2 = 5. Threshold is
        "< primary_end_length (5)", not "< previous plateau":
            i=5: window [6, 3] -> 6 < 5? No. Fails.
            i=6: window [3, 3] -> both < 5? Yes. tertiary_start=6.
        (Note plateau 6 was already < 5 at i=5 too, but i=5 fails because
        plateau 5 (len 6) is still >= primary_end_length -- it's a WINDOW
        check, not "the first single short plateau".)

        Index arithmetic (cumulative plateau lengths):
            plateau0: len1 -> end_idx 0
            plateau1: len2 -> end_idx 2
            plateau2: len5 -> end_idx 7      <- expected primary_end_idx
            plateau3: len6 -> end_idx 13
            plateau4: len6 -> end_idx 19
            plateau5: len6 -> end_idx 25     <- expected secondary_end_idx
            plateau6: len3 -> end_idx 28
            plateau7: len3 -> end_idx 31
            plateau8: len3 -> end_idx 34
            plateau9 (final, excluded): len1 -> end_idx 35
        """
        strain = strain_from_plateau_lengths([1, 2, 5, 6, 6, 6, 3, 3, 3, 1])
        test = make_test(strain_series=strain)

        result = classify_stages(test, k1=2, k2=2)

        assert result.primary_end_idx == 7
        assert result.secondary_end_idx == 25
        assert result.has_tertiary is True

    def test_plateau_equal_to_primary_length_does_not_count_as_tertiary(
        self, make_test, strain_from_plateau_lengths
    ):
        """
        Threshold check is STRICT (<, not <=). A plateau exactly equal to
        primary_end_length should never trigger tertiary on its own.
        Lengths [1, 2, 4, 4, 4, 4, 4], k1=2, k2=2: classified over
        [1, 2, 4, 4, 4, 4] (final excluded). Secondary onset at i=2
        (4>=4). primary_end_length=2 (plateau1, len 2) -- wait, deliberately
        chosen so every post-onset plateau (len 4) is >= primary_end_length,
        so tertiary can never trigger regardless of k2.
        """
        strain = strain_from_plateau_lengths([1, 2, 4, 4, 4, 4, 4])
        test = make_test(strain_series=strain)

        result = classify_stages(test, k1=2, k2=2)

        assert result.has_tertiary is False


# ---------------------------------------------------------------------------
# classify_stages -- edge-case hyperparameter values (k1=1 / k2=1 / ties)
#
# These pin down CURRENT behavior rather than asserting it is necessarily
# "correct" -- k1=1 and flat leading runs interact with classify_stages'
# use of `plateaus_for_classification[secondary_start_plateau_idx - 1]` in a
# way that relies on Python's negative-index wraparound whenever
# secondary_start_plateau_idx == 0. Worth reviewing this behavior explicitly
# before choosing a production k1 -- see the sensitivity sweep script.
# ---------------------------------------------------------------------------

class TestClassifyStagesEdgeCases:
    def test_k1_equals_one_triggers_at_first_multi_point_plateau(
        self, make_test, strain_from_plateau_lengths
    ):
        """
        k1=1 makes the pairwise "run" check vacuously True for any i
        (range(i, i+0) is empty), BUT the `n_points > 1` filter still
        applies to the single-plateau window at i. So secondary_start_plateau_idx
        is the index of the FIRST classified plateau with n_points > 1 --
        not always 0.

        Lengths [1, 2, 3, 4, 5]: final plateau (len 5) excluded -> classified
        over [1, 2, 3, 4]. Plateau 0 has n_points=1, fails the filter.
        Plateau 1 (n_points=2) is the first to pass -> secondary_start_plateau_idx=1.

        => primary_end_plateau_idx = 0 -> primary_end_idx = plateaus_for_classification[0].end_idx = 0

        This is a DIFFERENT edge case from test_flat_run_from_the_start_triggers_secondary_at_zero
        below: here it's the n_points>1 filter driving the index, not the
        negative-index wraparound (which only bites when the first
        classified plateau already has n_points > 1).
        """
        strain = strain_from_plateau_lengths([1, 2, 3, 4, 5])
        test = make_test(strain_series=strain)

        result = classify_stages(test, k1=1, k2=2)

        assert result.primary_end_idx == 0

    def test_k1_equals_one_all_single_point_plateaus_returns_none(
        self, make_test, strain_from_plateau_lengths
    ):
        """
        If every classified plateau has n_points == 1 (no plateau ever
        passes the n_points > 1 filter), secondary_start_plateau_idx stays
        None regardless of k1=1's vacuous pairwise check -- primary_end_idx
        and secondary_end_idx should both be None.
        """
        strain = strain_from_plateau_lengths([1, 1, 1, 1])
        test = make_test(strain_series=strain)

        result = classify_stages(test, k1=1, k2=2)

        assert result.primary_end_idx is None
        assert result.secondary_end_idx is None

