from robustness import hac_coverage_study, sample_length_study


def test_sample_length_study_schema_and_long_history_reduces_error():
    # Small deterministic smoke study: the full 200-world run is a report
    # generation step, not a unit-test burden.
    out = sample_length_study(n_worlds=12, horizons=(1, 8), seed=111)
    assert out["horizons_years"] == [1, 8]
    assert out["results"]["8"]["mean_mae_across_factors"] \
        < out["results"]["1"]["mean_mae_across_factors"]


def test_hac_study_schema():
    out = hac_coverage_study(n_worlds=12, phis=(0.0, 0.6), years=3,
                             seed=222)
    assert set(out["results"]) == {"0.0", "0.6"}
    for row in out["results"].values():
        assert 0 <= row["mean_iid_95_coverage"] <= 1
        assert 0 <= row["mean_nw_95_coverage"] <= 1
