import pytest
import torch
from torch import nn
from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))

    def forward(self, x):
        return self.net(x)


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=3)  # Output: (batch, 4, 6, 6)
        self.fc = nn.Linear(4 * 6 * 6, 10)  # Correct size: 144 → 10 classes

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 100)
        self.fc2 = nn.Linear(100, 100)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


@pytest.fixture
def tiny_mlp():
    return TinyMLP()


@pytest.fixture
def tiny_cnn():
    return TinyCNN()


@pytest.fixture
def simple_cnn():
    return SimpleCNN()


@pytest.fixture
def model_and_optimizer():
    # Create a simple model and optimizer for testing
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc = nn.Linear(10, 10)

        def forward(self, x):
            return self.fc(x)

    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return model, optimizer


@pytest.fixture
def binary_dataset():
    data = [torch.tensor([i], dtype=torch.float32) for i in range(10)]
    labels = [i % 2 for i in range(10)]  # binary labels
    return SimpleDataset(data, labels)


@pytest.fixture
def ternary_dataset():
    data = [i for i in range(12)]
    labels = [i % 3 for i in range(12)]  # Three classes: 0, 1, and 2
    return SimpleDataset(data, labels)
