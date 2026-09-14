from experiments.arc.smoke import run


def test_end_to_end_segment_training_saves_labeled_development_artifacts(tmp_path):
    result=run(str(tmp_path),steps=2)
    assert result['task']=='synthetic_color_shift_not_ARC'
    assert len(result['history'])==4
    assert (tmp_path/'summary.json').exists() and (tmp_path/'smoke.pt').exists()
    for row in result['history']:
        assert row['mlp_calls']==2
        assert all(s['converged_fraction']==1. for s in row['solver'])
        assert all(s['backward_residual']<=1e-5 for s in row['solver'])
