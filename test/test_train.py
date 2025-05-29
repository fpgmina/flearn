import numpy as np
import pytest
import torch
from torch import nn
from unittest.mock import MagicMock
from core.train import (
    _train,
    compute_predictions,
)


@pytest.fixture
def mock_model():
    # Create a simple model for testing
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 10),  # Example for image input of size 28x28
        nn.Softmax(dim=1),
    )
    return model


@pytest.fixture
def mock_train_loader():
    torch.manual_seed(42)
    # Create a fake dataloader
    return torch.utils.data.DataLoader(
        # input                   # target
        [(torch.randn(1, 28, 28), torch.tensor(1))] * 100,
        batch_size=2,
    )


@pytest.fixture
def mock_loss_func():
    return nn.CrossEntropyLoss()


@pytest.fixture
def mock_optimizer(mock_model):
    torch.manual_seed(42)
    return torch.optim.SGD(mock_model.parameters(), lr=0.001)


@pytest.fixture
def mock_training_params():
    # Mock the TrainingParams class
    mock_params = MagicMock()
    mock_params.model = MagicMock()
    mock_params.loss_function = nn.CrossEntropyLoss()
    mock_params.optimizer = torch.optim.SGD(mock_params.model.parameters(), lr=0.001)
    mock_params.epochs = 2
    mock_params.training_name = "test_model"
    mock_params.optimizer_class = torch.optim.SGD
    mock_params.optimizer_params = None
    return mock_params


# Test for the _train function
def test_train(mock_model, mock_train_loader, mock_loss_func, mock_optimizer):
    train_loss, train_accuracy, steps = _train(
        model=mock_model,
        train_loader=mock_train_loader,
        loss_func=mock_loss_func,
        optimizer=mock_optimizer,
    )
    assert np.allclose(train_loss, 1.7284637522)
    assert np.allclose(train_accuracy, 90)


# Test for compute_predictions
def test_compute_predictions(mock_model, mock_train_loader, mock_loss_func):
    model = mock_model
    predictions, labels, loss, accuracy = compute_predictions(
        model=model,
        dataloader=mock_train_loader,
        loss_function=mock_loss_func,
    )
    # Check that predictions and labels have the correct shape
    assert len(predictions) == len(labels)
    assert np.allclose(loss, 2.2803399562835693)


# Test for train_model
# def test_train_model(mock_training_params, mock_train_loader, mock_loss_func, mock_optimizer):
# Mock wandb.init to prevent actual wandb API calls during testing
# with pytest.raises(AssertionError):  # Check if it raises assertion errors
#     best_acc = train_model(
#         training_params=mock_training_params,
#         train_loader=mock_train_loader,
#         val_loader=None,
#         project_name="test_project",
#     )
#
# assert best_acc >= 0
