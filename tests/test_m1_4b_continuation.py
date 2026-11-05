from pathlib import Path
import sys
import copy
import pytest
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from m1_4b_continuation import validate_metadata, require_match
from m1_4_bounded_overfit import CONFIG, fingerprint
from m1_1_satlas_smoke import checkpoint_digest


def test_stochastic_uninterrupted_and_restored_update(tmp_path):
    def construct():
        m=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Dropout(.3),torch.nn.Linear(4,1))
        return m,torch.optim.AdamW(m.parameters(),lr=1e-4)
    def update(m,o):
        o.zero_grad(set_to_none=True)
        loss=m(torch.ones(2,3)).square().mean(); loss.backward()
        gradients=fingerprint({n:p.grad for n,p in m.named_parameters()})
        o.step()
        return dict(loss=loss.item(),gradient=gradients,model=fingerprint(m.state_dict()),
                    optimizer=fingerprint(o.state_dict()),rng=fingerprint(torch.get_rng_state()))
    torch.manual_seed(0)
    model,opt=construct()
    for _ in range(3): update(model,opt)
    path=tmp_path/'checkpoint.pth'
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state()),path)
    reference=update(model,opt)
    state=torch.load(path,map_location='cpu',weights_only=True)
    resumed,optimizer=construct()
    resumed.load_state_dict(state['model'],strict=True); optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['rng'])
    assert all(require_match(update(resumed,optimizer),reference).values())


@pytest.mark.parametrize('changed',['config','step'])
def test_config_and_step_rejection(changed):
    baseline=dict(inputs=[],dataset_revision='r',initial_checkpoint={},experiment='fixture')
    state=baseline|dict(config=copy.deepcopy(CONFIG),seed=0,completed_steps=20)
    if changed=='config': state['config']['lr']=.1
    else: state['completed_steps']=19
    with pytest.raises(ValueError): validate_metadata(state,baseline)


def test_mismatch_fails_without_tolerance():
    with pytest.raises(ValueError,match='Strict mismatch'):
        require_match({'loss':1.000000001},{'loss':1.0})


def test_checkpoint_hash_mismatch(tmp_path):
    path=tmp_path/'checkpoint.pth'; path.write_bytes(b'fixture')
    with pytest.raises(ValueError,match='SHA-256 mismatch'):
        checkpoint_digest(path,'0'*64)
