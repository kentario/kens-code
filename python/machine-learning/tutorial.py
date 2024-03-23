import torch
import torch.nn as nn

"""
a is number of nodes in next layer
b is number of nodes in current layer

weights                 inputs    biases
for layer               for layer for layer
+-                  -+ +-  -+     +-  -+     +-  -+
| W0N0 W0N1     W0Nb | | N0 |     | B0 |     | N0 |
| W1N0 W1N1 ... W1Nb | | N1 |     | B1 |     | N1 |
| W2N0 W2N1     W2Nb | | N2 |  +  | B2 |  =  | N2 |
| ...  ...      ...  | | .. |     | .. |     | .. |
| WaN0 WaN1     WaNb | | Nb |     | Bb |     | Na |
+-                  -+ +-  -+     +-  -+     +-  -+
"""

class ff(nn.Module):
    def __init__(self, nodes_per_layer):
        super().__init__()
        self.weights = []
        self.biases = []
        for i in range(len(nodes_per_layer)):
            if (i <= 0):
                continue
            # weights width = num nodes in this layer.
            # weights height = num nodes in next layer.
            weight = nn.Parameter(torch.rand(nodes_per_layer[i], nodes_per_layer[i - 1]))
            self.weights.append(weight)

    def forward (self, x):
        # x should be a tensor
        for weight in self.weights:
            x = torch.matmul(weight, x)
        return x

def main ():
    torch.manual_seed(0)
    nodes_per_layer = [2, 1]
    test = ff(nodes_per_layer)
    print(test.weights)
    print(test.forward(torch.tensor([0., 1.])))
    
if __name__ == "__main__":
    main()
