import copy
import torch
import torch.nn as nn

def get_clones(module, num_of_deep_copies):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_of_deep_copies)])

class ValueFunctionNormaliser(nn.Module):
    def __init__(self, mean, std):
        super(ValueFunctionNormaliser, self).__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.mean = nn.Parameter(mean * torch.ones(1, dtype=torch.float32, device=self.dummy_param.device))
        self.std = nn.Parameter(std * torch.ones(1, dtype=torch.float32, device=self.dummy_param.device))
        self.mean_np = float(self.mean.cpu().data.numpy())
        self.std_np = float(self.std.cpu().data.numpy())

    def normalise(self, values):
        return (values - self.mean_np) / (self.std_np + 1e-4)

    def denormalise(self, normalised_values):
        return self.mean_np + normalised_values * self.std_np