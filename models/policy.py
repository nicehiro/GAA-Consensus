from torch import nn
from utils import build_net


class Policy(nn.Module):
    def __init__(
        self, features_dim: int, output_dim: int, hidden_sizes: int, hidden_layers: int
    ) -> None:
        super(Policy, self).__init__()
        self.model = build_net(
            features_dim,
            output_dim,
            [hidden_sizes for _ in range(hidden_layers)],
            activation=nn.ReLU(),
            output_activation=nn.Identity(),
        )

    def forward(self, x):
        x = self.model(x)
        return x
