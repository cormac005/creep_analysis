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
        Plateau lengths: [1, 2, 3, 5, 5, 5, 4, 3, 2, 1], k1=2, k2=2.
        Final plateau (idx 9, len 1) excluded -> classified over
        [1, 2, 3, 5, 5, 5, 4, 3, 2] (9 plateaus, local indices 0-8).

        Secondary onset (k1=2, single-step non-increase): first i where
        n_points[i] >= n_points[i+1]. 1<2<3<5 all strictly increasing, then
        5>=5 at i=3 (local plateau 3 vs 4). => secondary_start_plateau_idx=3
        => primary_end_idx = end_idx of local plateau 2 (len-3 plateau)

        Tertiary onset (k2=2, single-step strict decrease), searched from
        i = secondary_start_plateau_idx + k1 = 3+2 = 5 onward: n_points[5]=5,
        n_points[6]=4 -> 5>4 at i=5. => tertiary_start_plateau_idx=5
        => secondary_end_idx = end_idx of local plateau 4 (the plateau
           right before the tertiary_start index)

        Index arithmetic:
            plateau0: len1 -> end_idx 0
            plateau1: len2 -> end_idx 2
            plateau2: len3 -> end_idx 5      <- expected primary_end_idx
            plateau3: len5 -> end_idx 10
            plateau4: len5 -> end_idx 15     <- expected secondary_end_idx
            plateau5: len5 -> end_idx 20
            plateau6: len4 -> end_idx 24
            plateau7: len3 -> end_idx 27
            plateau8: len2 -> end_idx 29
            plateau9 (final, excluded): len1 -> end_idx 30
        """
        strain = strain_from_plateau_lengths([1, 2, 3, 5, 5, 5, 4, 3, 2, 1])
        test = make_test(strain_series=strain)

        result = classify_stages(test, k1=2, k2=2)

        assert result.primary_end_idx == 5
        assert result.secondary_end_idx == 15
        assert result.has_tertiary is True

    def test_final_plateau_dip_is_correctly_ignored(
        self, make_test, strain_from_plateau_lengths
    ):
        """
        Plateau lengths [1, 2, 3, 4, 2]: the ONLY decrease in the whole
        sequence is the final plateau (4 -> 2). Since the final plateau is
        always excluded from classification, [1,2,3,4] (strictly increasing)
        is all that's actually evaluated -- no secondary/tertiary should be
        detected. This pins down that a low final-plateau count (a test
        artificially cut short) doesn't get misread as a real transition.
        """
        strain = strain_from_plateau_lengths([1, 2, 3, 4, 2])
        test = make_test(strain_series=strain)

        result = classify_stages(test, k1=2, k2=2)

        assert result.primary_end_idx is None
        assert result.secondary_end_idx is None


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
    def test_k1_equals_one_triggers_immediately(
        self, make_test, strain_from_plateau_lengths
    ):
        """
        k1=1 means the "run" check has an empty inner range (range(i, i+0)),
        which `all()` treats as vacuously True for ANY i -- so
        secondary_start_plateau_idx is always 0, regardless of data.

        This makes primary_end_idx =
            plateaus_for_classification[0 - 1].end_idx
          = plateaus_for_classification[-1].end_idx  (Python negative index!)

        i.e. primary_end_idx ends up equal to the LAST classified plateau's
        end_idx, not the first. Flag this to yourself before using k1=1 in
        production -- it is very unlikely to be the intended behavior.
        """
        strain = strain_from_plateau_lengths([1, 2, 3, 4, 5])
        test = make_test(strain_series=strain)

        result = classify_stages(test, k1=1, k2=2)

        # Pins current (surprising) behavior so a future refactor that
        # changes it doesn't slip by unnoticed.
        last_classified_end_idx = _find_plateaus(strain)[-2].end_idx  # [:-1] excludes final
        assert result.primary_end_idx == last_classified_end_idx

    def test_flat_run_from_the_start_triggers_secondary_at_zero(
        self, make_test, strain_from_plateau_lengths
    ):
        """
        Plateau lengths [3, 3, 3, 3], k1=3: classified over [3, 3, 3]
        (final plateau excluded). All three are equal (non-increasing under
        `>=`), so secondary_start_plateau_idx=0 immediately -- exercising
        the same negative-index edge case as test_k1_equals_one above, but
        via a tie rather than k1=1.
        """
        strain = strain_from_plateau_lengths([3, 3, 3, 3])
        test = make_test(strain_series=strain)

        result = classify_stages(test, k1=3, k2=2)

        last_classified_end_idx = _find_plateaus(strain)[-2].end_idx
        assert result.primary_end_idx == last_classified_end_idx