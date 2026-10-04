import json

import numpy as np
import pandas as pd
import pytest

from evaluate_design_sensitivity import evaluate_saved_run, main, sensitivity_log_values


def make_run(path):
    path.mkdir()
    (path/"config.json").write_text(json.dumps({"data":{"samples_per_design":2}}))
    rows=[]
    for design in (4,5):
        for location in (1,2):
            for time in (0.,.001,.002):
                rows.append(dict(run_number=design*2+location,time=time,
                    acceleration_true_g=10*design+location+time*100,
                    acceleration_pred_g=45+location+time*100))
    pd.DataFrame(rows).sample(frac=1,random_state=1).to_csv(path/"test_acceleration_histories.csv",index=False)


def test_backfill_exports_numeric_metrics_and_plots_without_model_or_raw_data(tmp_path):
    run=tmp_path/"run"
    make_run(run)
    main([str(run)])
    report=json.loads((run/"test_design_sensitivity.json").read_text())
    assert report["design_difference_skill"] == report["hic15"]["design_difference_skill"] == 0
    assert report["num_matched_locations"] == 2
    assert report["impact_xy_checked"] is False
    assert len(report["source_sha256"]) == 64
    assert (run/"test_design_sensitivity.png").stat().st_size>0
    frame=pd.read_csv(run/"test_design_sensitivity_by_location.csv")
    assert len(frame)==2 and "hic15_design_difference_skill" in frame
    log=sensitivity_log_values(report)
    assert log["test/design_sensitivity/design_difference_skill"] == 0
    assert "test/design_sensitivity/difference_alignment_cosine" not in log
    assert all(np.isfinite(x) for x in log.values())


def test_explicit_location_cohort_is_respected_and_missing_locations_fail(tmp_path):
    run=tmp_path/"run"
    make_run(run)
    report=evaluate_saved_run(run,locations=[2])
    assert report["num_matched_locations"] == 1
    assert report["location_ids"] == [2]
    assert report["evaluated_run_numbers"] == [10,12]
    with pytest.raises(ValueError,match="absent"):
        evaluate_saved_run(run,locations=[3])


def test_duplicate_time_rows_are_not_silently_averaged(tmp_path):
    run=tmp_path/"run"
    make_run(run)
    path=run/"test_acceleration_histories.csv"
    frame=pd.read_csv(path)
    pd.concat([frame,frame.iloc[:1]]).to_csv(path,index=False)
    with pytest.raises(ValueError,match="strictly increasing"):
        evaluate_saved_run(run)
