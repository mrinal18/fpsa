"""Integration tests against the actual pinned upstream source, never mocks."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import numpy as np
import pytest
import torch


@pytest.fixture
def upstream(monkeypatch):
    value=os.environ.get('FPSA_TRM_REFERENCE')
    if not value or not Path(value).is_dir():
        pytest.skip('Set FPSA_TRM_REFERENCE to the pinned TRM checkout')
    monkeypatch.syspath_prepend(value)
    return Path(value)


def test_official_sparse_optimizer_and_adapter_carry(upstream):
    from src.fpsa_arc.trm_adapter import FPSAARC_ACTV1,ARCLossHead
    from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
    cfg=dict(hidden_size=16,num_heads=2,layers=1,seq_len=9,puzzle_emb_len=1,
             puzzle_emb_ndim=16,num_puzzle_identifiers=8,batch_size=2,
             halt_max_steps=2,max_iter=40,max_iter_eval=40)
    model=FPSAARC_ACTV1(cfg)
    head=ARCLossHead(model)
    batch=dict(inputs=torch.randint(2,12,(2,9)),labels=torch.randint(2,12,(2,9)),
               puzzle_identifiers=torch.tensor([1,2]))
    carry=head.initial_carry(batch)
    new,loss,metrics,pred,_=head(return_keys=['preds'],carry=carry,batch=batch)
    loss.backward()
    assert model.puzzle_emb.local_weights.grad is not None
    opt=CastedSparseEmbeddingSignSGD_Distributed(model.puzzle_emb.buffers(),world_size=1,lr=.01)
    before=model.puzzle_emb.weights.clone()
    opt.step()
    assert not torch.equal(before,model.puzzle_emb.weights)
    assert torch.equal(before[3:],model.puzzle_emb.weights[3:])
    model.eval()
    carry=head.initial_carry(batch)
    with torch.inference_mode():
        carry,*_=head(return_keys=[],carry=carry,batch=batch)
        assert not carry.halted.any()
        replacement={k:torch.zeros_like(v) for k,v in batch.items()}
        carry,*_=head(return_keys=[],carry=carry,batch=replacement)
        assert carry.halted.all()
        assert torch.equal(carry.current_data['inputs'],batch['inputs'])


def test_codec_matches_official_transforms(upstream):
    from dataset.common import dihedral_transform
    from experiments.arc.codec import transform
    grid=np.arange(15).reshape(3,5)
    for t in range(8):
        assert np.array_equal(transform(grid,t),dihedral_transform(grid,t))


def test_official_builder_keeps_query_targets_out_of_train(upstream,tmp_path):
    pytest.importorskip('argdantic')
    from dataset.build_arc_dataset import DataProcessConfig,convert_dataset
    prefix=tmp_path/'arc'
    puzzles={'a':dict(train=[dict(input=[[1,2]],output=[[3,4]])],test=[dict(input=[[5]],output=[[9]])])}
    for subset in ['training','evaluation']:
        Path(f'{prefix}_{subset}_challenges.json').write_text(json.dumps(puzzles))
        Path(f'{prefix}_{subset}_solutions.json').write_text(json.dumps({'a':[[[9]]]}))
    # Use distinct task identifiers across subsets.
    evaluation={'b':puzzles['a']}
    Path(f'{prefix}_evaluation_challenges.json').write_text(json.dumps(evaluation))
    Path(f'{prefix}_evaluation_solutions.json').write_text(json.dumps({'b':[[[9]]]}))
    dest=tmp_path/'prepared'
    convert_dataset(DataProcessConfig(input_file_prefix=str(prefix),output_dir=str(dest),
                      subsets=['training','evaluation'],test_set_name='evaluation',num_aug=0))
    ids=json.loads((dest/'identifiers.json').read_text())
    train_ids=np.load(dest/'train/all__puzzle_identifiers.npy')
    boundaries=np.load(dest/'train/all__puzzle_indices.npy')
    labels=np.load(dest/'train/all__labels.npy')
    for pidx,tid in enumerate(train_ids):
        if ids[tid]=='b':
            assert len(labels[boundaries[pidx]:boundaries[pidx+1]])==1
            assert labels[boundaries[pidx]][0]==5  # demo color 3 -> token 5, not query 9 -> 11


def test_official_ema_changes_dense_weights_not_task_table(upstream):
    from models.ema import EMAHelper
    from src.fpsa_arc.trm_adapter import FPSAARC_ACTV1
    cfg=dict(hidden_size=16,num_heads=2,layers=1,seq_len=9,puzzle_emb_len=1,
             puzzle_emb_ndim=16,num_puzzle_identifiers=8,batch_size=2,halt_max_steps=2)
    model=FPSAARC_ACTV1(cfg)
    ema=EMAHelper(mu=.5)
    ema.register(model)
    original=model.inner.core.embedding.weight.detach().clone()
    with torch.no_grad():
        model.inner.core.embedding.weight.add_(2.)
        model.puzzle_emb.weights.fill_(3.)
    ema.update(model)
    snapshot=ema.ema_copy(model)
    assert torch.allclose(snapshot.inner.core.embedding.weight,original+1.)
    assert (snapshot.puzzle_emb.weights==3.).all()
    assert torch.allclose(model.inner.core.embedding.weight,original+2.)


def test_official_hydra_config_composes_both_models_without_cuda(upstream, tmp_path):
    # Config composition must not import the fused CUDA optimizer on CPU.
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from experiments.arc.upstream import install_shims
    install_shims(upstream)
    for arch in ['fpsa_arc_refine', 'trm']:
        with initialize_config_dir(config_dir=str(upstream / 'config'), version_base=None):
            resolved = compose(config_name='cfg_pretrain', overrides=[
                f'arch={arch}', f'+checkpoint_path={tmp_path}',
                '+run_name=config_test', 'ema=True',
                '+eval_save_outputs=[inputs,puzzle_identifiers,q_halt_logits,preds]',
                'evaluators=[{name:arc@ARC,aggregated_voting:false}]'])
            resolved = OmegaConf.to_container(resolved, resolve=True)
        assert resolved['ema'] is True
        assert resolved['evaluators'][0]['aggregated_voting'] is False
        assert 'preds' in resolved['eval_save_outputs']
        assert resolved['arch']['name'].endswith(
            '@FPSAARC_ACTV1' if arch != 'trm' else '@TinyRecursiveReasoningModel_ACTV1')
