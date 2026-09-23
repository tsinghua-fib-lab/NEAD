import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, *dims, activation=nn.ReLU, out_activation=nn.Identity, dropout=0.0):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if activation is not None:
                layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        else:
            layers.append(nn.Linear(dims[-2], dims[-1]))
            if out_activation is not None:
                layers.append(out_activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
