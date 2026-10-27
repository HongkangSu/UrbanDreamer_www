import torch
from urban_dreamer.models.world_model import LowFrequencyGraphPrior, Dynamics, DynamicsOutput, ConditionEncoding


def test_existing_soft_source_retains_all_modes_and_normalized_energy():
    graph=torch.tensor([[0.,1.,0.,0.],[1.,0.,1.,0.],[0.,1.,0.,0.],[0.,0.,0.,0.]])
    prior=LowFrequencyGraphPrior(graph,rank=2,noise_scale=.1,prior_filter='soft',prior_filter_strength=.1)
    impulses=torch.eye(4).unsqueeze(-1).expand(4,4,64)
    perturbations=prior.sample_noise(torch.zeros_like(impulses),impulses)
    assert torch.linalg.matrix_rank(perturbations[:,:,0]) == 4
    torch.testing.assert_close(perturbations.square().sum(),torch.tensor(.1**2*2*64))
    symmetric=(graph+graph.T)/2
    degree=symmetric.sum(-1); inv=degree.clamp_min(1e-12).rsqrt()
    lap=torch.diag((degree>0).float())-inv[:,None]*symmetric*inv[None,:]
    torch.testing.assert_close((torch.eye(4)+.1*lap)@prior.soft_filter_operator,torch.eye(4))


def test_latent_prior_increment_and_noise_share_the_soft_operator():
    torch.manual_seed(31)
    graph=torch.tensor([[0.,1.,0.],[1.,0.,1.],[0.,1.,0.]])
    prior=LowFrequencyGraphPrior(graph,rank=2,noise_scale=.1,prior_filter='soft',
        prior_filter_strength=.1,prior_action_conditioned=False,prior_filter_input=False)
    with torch.no_grad():
        prior.output.weight.copy_(torch.eye(64))
    captured={}
    before=prior.state_proj.register_forward_pre_hook(
        lambda module,args:captured.update(input=args[0].detach().clone()))
    after=prior.output.register_forward_hook(
        lambda module,args,result:captured.update(raw_increment=result.detach().clone()))
    state=torch.randn(1,3,64)
    condition=ConditionEncoding(torch.zeros(1,3,256))
    increment=prior(state,condition)
    before.remove();after.remove()
    torch.testing.assert_close(captured['input'],state)
    torch.testing.assert_close(increment,prior.soft_filter_operator@captured['raw_increment'])
    noise=torch.randn_like(state)
    torch.testing.assert_close(prior.sample_noise(state,noise),
        prior.noise_scale*prior.noise_normalization*(prior.soft_filter_operator@noise))


def test_euler_checkpointing_preserves_solution_and_gradient():
    class Field(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(.2))

        def forward(self, action, *, packed_latents, flow_time, condition):
            return DynamicsOutput(self.weight * packed_latents + condition.edge_tokens)

    condition = ConditionEncoding(torch.ones(1,1,2,3)*.1)
    model = Field()
    initial = torch.ones(1,1,2,3)
    action = torch.zeros(1,1,2)
    plain = Dynamics.solve(model, initial, action, condition, steps=10)
    first_gradient = torch.autograd.grad(plain.sum(), model.weight)[0]
    saved = Dynamics.solve(model, initial, action, condition, steps=10, checkpoint_interval=2)
    saved_gradient = torch.autograd.grad(saved.sum(), model.weight)[0]
    expected = initial.clone()
    for _ in range(10):
        expected = expected + (.2 * expected + .1)/10
    torch.testing.assert_close(plain, expected)
    torch.testing.assert_close(saved, plain)
    torch.testing.assert_close(saved_gradient, first_gradient)
