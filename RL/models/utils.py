import copy
import torch
import torch.nn as nn

def get_clones(module, num_of_deep_copies):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_of_deep_copies)])

class ValueFunctionNormaliser(nn.Module):
    def __init__(self, mean, std):
        super(ValueFunctionNormaliser, self).__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.mean = nn.Parameter(torch.zeros(1, dtype=torch.float32, device=self.dummy_param.device))
        self.std = nn.Parameter(torch.zeros(1, dtype=torch.float32, device=self.dummy_param.device))

    def normalise(self, values):
        return (values - self.mean) / self.std

    def denormalise(self, normalised_values):
        return self.mean + normalised_values * self.std