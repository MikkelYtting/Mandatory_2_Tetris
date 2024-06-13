import torch.nn as nn

class DeepQNetwork(nn.Module):
    # En simpel Deep Q-Network (DQN) til forstærkningslæring
    def __init__(self):
        super(DeepQNetwork, self).__init__()

        # Første lag: Linear (4 -> 64) + ReLU aktivering
        self.conv1 = nn.Sequential(nn.Linear(4, 64), nn.ReLU(inplace=True))
        # Andet lag: Linear (64 -> 64) + ReLU aktivering
        self.conv2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
        # Tredje lag: Linear (64 -> 1)
        self.conv3 = nn.Sequential(nn.Linear(64, 1))

        # Initialiser vægte
        self._create_weights()

    def _create_weights(self):
        # Initialiser vægte med Xavier uniform og bias til 0
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Fremadrettet beregning gennem netværket
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x
