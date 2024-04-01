import torch
import torch.optim as optim
import torch.nn as nn

# A feed forward neural network that has 2 inputs, 1 hidden layer of 2 nodes, and 1 output.
class learn_logic(nn.Module):
    def __init__ (self):
        super().__init__()

        self.l1 = nn.Linear(2, 2)
        self.l2 = nn.Linear(2, 1)
        
    def forward (self, x):
        x = torch.tanh(self.l1(x))
        x = torch.celu(self.l2(x))

        return x
    
def main ():
    xor = learn_logic()
    xor_training_data = torch.tensor([[0., 0.],
                                      [0., 1.],
                                      [1., 0.],
                                      [1., 1.]])
    
    xor_training_labels = torch.tensor([[0.],
                                        [1.],
                                        [1.],
                                        [0.]])

    loss_f = nn.MSELoss()
    optimizer = optim.SGD(xor.parameters(), lr=0.01)
    max_epochs = 1000

    for epoch in range(max_epochs):
        # For each datapoint, calculate the output and update the weights and biases accordingly.
        for i in range(len(xor_training_data)):
            output = xor(xor_training_data)
            loss = loss_f(output, xor_training_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch % (max_epochs/10) == 0):
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    print(xor(xor_training_data))
    print(loss_f(xor(xor_training_data), xor_training_labels))
    
if __name__ == "__main__":
    main()



