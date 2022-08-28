import torch
from torch import nn

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.cv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)

    def forward(self, x):
        return self.cv1(x)


    @torch.no_grad()
    def linearize(self):
        signs = {}
        for name, param in self.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

def synflow(layer):
    if layer.weight.grad is not None:
        return torch.abs(layer.weight * layer.weight.grad)
    else:
        return torch.zeros_like(layer.weight)

model = Model()
model.linearize()
output = model(torch.ones(1, 3, 416, 416))
torch.sum(output).backward()

metric = 0
for layer in model.modules():
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        metric += synflow(layer)

all_metrics = metric.sum()
all_metrics.backward()
