from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from m1_4_bounded_overfit import metrics, fingerprint, finite_parameters, CONFIG


def test_metrics_and_undefined_denominators():
    truth=torch.tensor([0,1,1,0]); pred=torch.tensor([0,1,0,1])
    r=metrics(pred,truth)
    assert r['iou']==1/3 and r['f1']==.5 and r['precision']==.5 and r['recall']==.5
    empty=metrics(torch.zeros(3),torch.zeros(3))
    assert all(empty[k] is None for k in ['iou','f1','precision','recall'])


def test_safe_checkpoint_optimizer_roundtrip(tmp_path):
    torch.manual_seed(0)
    layer=torch.nn.Linear(2,1)
    optimizer=torch.optim.AdamW(layer.parameters(),lr=1e-4)
    before=layer.weight.detach().clone()
    layer(torch.ones(1,2)).sum().backward(); optimizer.step()
    assert not torch.equal(before,layer.weight)
    finite_parameters(layer)
    original=dict(model=layer.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),step=1)
    path=tmp_path/'safe.pth'; torch.save(original,path)
    restored=torch.load(path,map_location='cpu',weights_only=True)
    copy=torch.nn.Linear(2,1); copy.load_state_dict(restored['model'])
    opt=torch.optim.AdamW(copy.parameters(),lr=1e-4); opt.load_state_dict(restored['optimizer'])
    assert fingerprint(copy.state_dict())==fingerprint(original['model'])
    assert fingerprint(opt.state_dict())==fingerprint(original['optimizer'])
    assert fingerprint(restored['rng'])==fingerprint(original['rng'])
    with torch.no_grad(): copy.weight.add_(1)
    assert fingerprint(copy.state_dict())!=fingerprint(original['model'])


def test_bounded_configuration():
    assert CONFIG['steps']==20 and CONFIG['device']=='cpu' and CONFIG['seed']==0
    assert CONFIG['lr']==1e-4 and CONFIG['patch_size']==256


def test_nonfinite_parameter_stops():
    import pytest
    layer=torch.nn.Linear(2,1)
    with torch.no_grad(): layer.weight.fill_(float('nan'))
    with pytest.raises(ValueError,match='Nonfinite'):
        finite_parameters(layer)
