import torch

output = torch.Tensor([[1, 2, 3, 4, 5]])

target = torch.Tensor([0., 1., 0., 0., 0.])
smooth_dist = torch.ones_like(target) / output.size()[1]  # torch.Tensor([0.2, 0.2, 0.2, 0.2, 0.2])
smooth_eps = 0.1

# smooth_dist = smooth_dist.unsqueeze(0)
target.lerp_(smooth_dist, smooth_eps)

print(target)