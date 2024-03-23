import torch
import torch.nn as nn
import torch.nn.functional as functional

"""
a is number of nodes in next layer
b is number of nodes in current layer


weights                   inputs    biases
for layer                 for layer for layer
                        +--               --+
+-                  -+  | +-  -+     +-  -+ |   +-  -+
| W0N0 W0N1     W0Nb |  | | N0 |     | B0 | |   | N0 |
| W1N0 W1N1 ... W1Nb |  | | N1 |     | B1 | |   | N1 |
| W2N0 W2N1     W2Nb |  | | N2 |  +  | B2 | |=  | N2 |
| ...  ...      ...  |  | | .. |     | .. | |   | .. |
| WaN0 WaN1     WaNb |  | | Nb |     | Bb | |   | Na |
+-                  -+  | +-  -+     +-  -+ |    +-  -+
                        +--               --+

When multiplying matrix W with vector N,
the number of columns in matrix W needs to equal the rows in vector V
the output will be a vector with the number of rows in matrix W
"""

class ff(nn.Module):
    def __init__(self, nodes_per_layer):
        super().__init__()
        self.weights = []
        self.biases = []
        for i in range(len(nodes_per_layer)):
            # bias should have same size of current or i - 1
            bias = nn.Parameter(torch.rand(nodes_per_layer[i - 1]))
            self.biases.append(bias)
            # skip the first one because nothing leads into it.
            if (i <= 0):
                continue
            # height is next, or i
            # width is current or i - 1
            # tensors are height x width
            weight = nn.Parameter(torch.rand(nodes_per_layer[i], nodes_per_layer[i - 1]))
            self.weights.append(weight)

    def forward (self, x):
        # Input vector width should be 1
        if (x.size(dim=1) != 1):
            return None
        # Input vector height should match first weight width
        if (x.size(dim=0) != self.weights[0].size(dim=1)):
            return None

        for i, weight in enumerate(self.weights):
            x = torch.mm(weight, x)
            x = x + self.biases[i]
            
        return x

def main ():
    torch.manual_seed(0)
    nodes_per_layer = [2, 3, 1]
    test = ff(nodes_per_layer)
    in_tensor = torch.tensor([[0.],
                              [1.]])
    out_tensor = test.forward(in_tensor)
    print(f"in tensor: {in_tensor}")
    print(f"out tensor: {out_tensor}")
    
if __name__ == "__main__":
    main()
