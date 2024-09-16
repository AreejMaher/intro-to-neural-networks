import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from torchvision import datasets, transforms            # Load MNIST dataset (PyTorch's built-in dataset)


class Net(nn.Module):                                   ## Define the neural network architecture
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 30)
        self.fc2 = nn.Linear(30, 10)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Prepare data
Train_Dataset = datasets.MNIST('.', train=True, download=True, transform=transforms.ToTensor())
Test_Dataset = datasets.MNIST('.', train=False, download=True, transform=transforms.ToTensor())

Train_Loader = DataLoader(Train_Dataset, batch_size=10, shuffle=True)
Test_Loader = DataLoader(Test_Dataset, batch_size=10, shuffle=False)


model = Net()                                           ### Instantiate model, loss function, and optimizer
Criterion = nn.CrossEntropyLoss()
Optimizer = optim.SGD(model.parameters(), lr=3.0)

for epoch in range(30):                                 #### Train the model
    for data, target in Train_Loader:
        data = data.view(-1, 784)                       ##### Flatten the images
        Optimizer.zero_grad()
        output = model(data)
        loss = Criterion(output, target)
        loss.backward()
        Optimizer.step()

# Evaluate model on test data
Correct = 0
Total = 0
with torch.no_grad():
    for data, target in Test_Loader:
        data = data.view(-1, 784)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        Total += target.size(0)
        Correct += (predicted == target).sum().item()

Accuracy = Correct / Total
print(f"Accuracy: {Accuracy * 100:.2f}%")