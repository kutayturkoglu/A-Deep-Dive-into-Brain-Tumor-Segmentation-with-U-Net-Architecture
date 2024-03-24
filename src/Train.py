import torch
from tqdm import tqdm

def train(model, optimizer, loss_fn, loader, device):
    """
    Function to train the model.

    Args:
        model: The neural network model to be trained.
        optimizer: The optimizer used for updating the model parameters.
        loss_fn: The loss function used for computing the loss.
        loader: The data loader containing the training data.
        device: The device (CPU or GPU) where the computation will take place.

    Returns:
        The average loss over the training data for the epoch.
    """
    epoch_loss = 0.0
    model.train()

    for x, y in tqdm(loader):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(loader)

def valid(model, loader, loss_fn, device):
    """
    Function to validate the model.

    Args:
        model: The neural network model to be validated.
        loader: The data loader containing the validation data.
        loss_fn: The loss function used for computing the loss.
        device: The device (CPU or GPU) where the computation will take place.

    Returns:
        The average loss over the validation data for the epoch.
    """
    epoch_loss = 0.0
    model.eval()

    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)
        pred = model(x)
        loss = loss_fn(pred, y)
        epoch_loss += loss.item()
    return epoch_loss / len(loader)
