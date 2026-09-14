from dataclasses import replace
import pytest
import torch
from src.fpsa_arc import ARCConfig, ARCReasoner, Carry
from src.fpsa_arc.losses import supervised_loss, per_example_loss
from src.fpsa_arc.legacy import LegacyARC


def config(**kw):
    return ARCConfig(**(dict(hidden_size=16,num_heads=2,layers=2,seq_len=9,
                    puzzle_emb_len=0,max_iter=60,max_iter_eval=60)|kw))


def test_mlp_once_evidence_once_and_answer_carried():
    torch.manual_seed(5)
    model=ARCReasoner(config())
    counts={'mlp':0,'values':0}
    handles=[]
    def mlp(*a):counts['mlp']+=1
    def values(*a):counts['values']+=1
    for stage in model.stages:
        handles.extend([stage.mlp.register_forward_hook(mlp), stage.attention.ve.register_forward_hook(values)])
    x=torch.randint(0,12,(2,9))
    carry,out=model.forward_segment(x)
    assert counts=={'mlp':2,'values':2}
    assert out['nfe']>2 and out['mlp_calls']==2
    assert not carry.answer.requires_grad and all(not z.requires_grad for z in carry.latent)
    for h in handles:h.remove()
    zero = model.initial_carry(2)
    altered = Carry(torch.randn_like(zero.answer),zero.latent)
    _,other=model.forward_segment(x,altered)
    assert not torch.allclose(out['logits'],other['logits'])


def test_all_stages_receive_correct_implicit_gradient_vs_deep_unroll():
    torch.manual_seed(6)
    c=config(fp_tol=1e-10,backward_tol=1e-10,max_iter=80,max_iter_eval=80)
    implicit=ARCReasoner(c).double()
    unroll=ARCReasoner(replace(c,mode='unroll')).double()
    unroll.load_state_dict(implicit.state_dict())
    x=torch.randint(0,12,(2,9)); target=torch.randint(0,12,(2,9))
    for model in [implicit,unroll]:
        _,out=model.forward_segment(x)
        loss,_=supervised_loss(out,target)
        loss.backward()
    for (name,p),(n2,q) in zip(implicit.named_parameters(),unroll.named_parameters()):
        assert name==n2
        if p.grad is None or q.grad is None:
            assert p.grad is None and q.grad is None,name
        else:
            assert torch.allclose(p.grad,q.grad,atol=5e-7,rtol=1e-4),name


def test_bptt_forward_depth_is_same_in_train_and_eval():
    torch.manual_seed(8)
    model=ARCReasoner(config(mode='unroll',max_iter=4,max_iter_eval=4))
    x=torch.randint(0,12,(2,9))
    _,a=model.forward_segment(x)
    model.eval()
    with torch.no_grad():_,b=model.forward_segment(x)
    assert torch.equal(a['logits'],b['logits'])
    assert not a['infos'][0].is_equilibrium
    with pytest.raises(ValueError,match='identical'):
        config(mode='unroll',max_iter=4,max_iter_eval=8)


def test_task_embedding_required_and_receives_gradient():
    model=ARCReasoner(config(puzzle_emb_len=2))
    x=torch.randint(0,12,(2,9))
    with pytest.raises(ValueError,match='Task embeddings'):
        model.forward_segment(x)
    emb=torch.randn(2,16,requires_grad=True)
    _,out=model.forward_segment(x,task_embedding=emb)
    out['logits'].square().mean().backward()
    assert emb.grad is not None and emb.grad.norm()>0


def test_parameter_accounting_separates_external_learned_table():
    model=ARCReasoner(ARCConfig())
    report=model.parameter_report(1000,512)
    assert report['core_parameters']==6337043
    assert report['task_embedding_values']==512000
    assert report['total_learned_values']==report['core_parameters']+512000


def test_stablemax_has_no_unselected_branch_pole_and_ignores_padding():
    logits=torch.tensor([[[1.,-1.,2.],[0.,1.,-4.]]],requires_grad=True)
    labels=torch.tensor([[0,-100]])
    per,valid,_=per_example_loss(logits,labels)
    per.sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.equal(logits.grad[:,1],torch.zeros_like(logits.grad[:,1]))
    assert valid.all()


def test_loss_weights_examples_equally_not_grid_areas():
    logits=torch.tensor([[[3.,0.],[0.,3.],[0.,3.]],[[0.,3.],[0.,3.],[0.,3.]]])
    labels=torch.tensor([[0,-100,-100],[0,0,0]])
    per,_,_=per_example_loss(logits,labels)
    out={'logits':logits,'q_halt_logits':torch.zeros(2)}
    loss,_=supervised_loss(out,labels,halt_weight=0.)
    assert torch.allclose(loss,per.mean())
    assert not torch.allclose(loss,(per[0]+3*per[1])/4)


def test_legacy_operator_can_use_new_solvers_without_touching_old_source():
    torch.manual_seed(9)
    model=LegacyARC(config(layers=1,max_iter=100,max_iter_eval=100,fp_tol=1e-5))
    x=torch.randint(0,12,(2,9))
    _,out=model.forward_segment(x)
    out['logits'].square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert out['infos'][0].converged.all()


def test_jacobian_probe_regularizer_is_differentiable():
    model=ARCReasoner(config(stability_weight=.1,stability_target=1e-5,stability_probes=1))
    x=torch.randint(0,12,(1,9))
    _,out=model.forward_segment(x)
    assert out['stability_loss'].requires_grad
    out['stability_loss'].backward()
    assert any(p.grad is not None and p.grad.norm()>0 for p in model.parameters())


def test_zero_task_embeddings_do_not_create_zero_cosine_query_anchors():
    for seed in range(12):
        torch.manual_seed(seed)
        model=ARCReasoner(config(layers=1,puzzle_emb_len=4,max_iter=48,max_iter_eval=48))
        x=torch.randint(0,12,(2,9))
        task=torch.zeros(2,16)
        encoded=model.encode(x,task)
        assert (encoded[:,:4].norm(dim=-1)>0).all()
        with torch.no_grad():
            _,out=model.forward_segment(x,task_embedding=task)
        assert out['infos'][0].converged.all()
