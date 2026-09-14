import numpy as np
import pytest
import torch
from experiments.arc.codec import encode,decode,transform,inverse,parse_identifier,grid_key
from experiments.arc.evaluate import rank_candidates,score_candidates


@pytest.mark.parametrize('tid',range(8))
def test_rectangular_augmentation_roundtrip_and_black(tid):
    grid=np.arange(15).reshape(3,5)%10
    colors=np.array([0,4,3,2,1,9,8,7,6,5])
    a=transform(colors[grid],tid)
    assert np.array_equal(inverse(a,tid,colors),grid)
    assert np.array_equal(decode(encode(a)),a)
    name=f"task|||t{tid}|||"+''.join(map(str,colors))
    base,t,p=parse_identifier(name)
    assert base=='task' and t==tid and np.array_equal(p,colors)


def test_padding_eos_and_black_are_distinct():
    grid=np.array([[0,1],[2,0]])
    encoded=encode(grid)
    assert encoded[0]==2 and encoded[2]==1 and encoded[3]==0
    assert np.array_equal(decode(encoded),grid)
    assert decode(np.zeros(900,dtype=int)) is None


def row(task,q,grid,cid,confidence=0.):
    return dict(task_id=task,query_index=q,grid=grid,source='one_checkpoint',candidate_id=cid,confidence=confidence)


def test_vote_primary_confidence_tiebreak_oracle_not_submission():
    records=[row('a',0,[[1]],'0',.9),row('a',0,[[2]],'1',.1),row('a',0,[[2]],'2',.1),
             row('a',0,[[3]],'3',.05)]
    ranked,_=rank_candidates(records)
    assert ranked['a',0][0]['grid']==[[2]]
    score,submission=score_candidates(records,{'a':[[[3]]], 'missing':[[[0]]]})
    assert score['top1']==0 and score['top2']==0 and score['oracle_coverage']==50
    assert submission['a'][0]['attempt_1']==[[2]]
    assert submission['a'][0]['attempt_2']==[[1]]


def test_task_macro_metric_and_reset_between_calls():
    records=[row('a',0,[[1]],'0')]
    score,_=score_candidates(records,{'a':[[[1]],[[2]]],'b':[[[2]]]})
    assert score['top1']==25.  # .5 on task A, zero on B
    score2,_=score_candidates([],{'a':[[[1]],[[2]]],'b':[[[2]]]})
    assert score2['top1']==0.


def test_duplicate_events_and_bad_grids_do_not_inflate_results():
    r=row('a',0,[[2]],'id')
    with pytest.raises(ValueError,match='Duplicate'):
        rank_candidates([r,r])
    score,_=score_candidates([row('a',0,None,'id')],{'a':[[[0]]]})
    assert score['top1']==0 and score['invalid_candidates']==1
