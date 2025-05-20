# train pizza steak sushi
import torch
from torch import nn
from dataset.pss import get_dataloaders
from models.transfer_learning_net import get_model, get_pretrained_model
from core.train import train_model
from core.train_params import TrainingParams

if __name__ == "__main__":
    train_dataloader, test_dataloader, class_names = get_dataloaders()
    pretrained_model = get_pretrained_model()
    model = get_model(model=pretrained_model, class_names=class_names)
    params = TrainingParams(
        training_name="transfer_learning",
        model=model,
        loss_function=nn.CrossEntropyLoss(),
        learning_rate=1e-3,
        optimizer_class=torch.optim.Adam,  # type: ignore
        epochs=10,
    )
    train_model(
        training_params=params,
        train_loader=train_dataloader,
        val_loader=test_dataloader,
        project_name="mldl_lab5",
    )
