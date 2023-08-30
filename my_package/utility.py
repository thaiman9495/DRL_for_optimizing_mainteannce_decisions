import torch
import torch.nn as nn


def compute_mse(model, data_loader, data_len, device):
    """
    This function aims at computing mean squared error

    :param model: newest neural model
    :param data_loader: data loader
    :param data_len: len of dataset
    :param device: GPU or CPU
    :return: mean squared error
    """
    model.eval()                                      # Put model in evaluation mode
    loss_function = nn.MSELoss(reduction='sum')       # Sum of squared loss function
    total_loss = 0.0                                  # Initilize total loss
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.unsqueeze(1).to(device)
            y_estimated = model(x)
            loss = loss_function(y_estimated, y)
            total_loss += loss

    mean_loss = total_loss / data_len

    model.train()                                      # Put model back to training mode
    return mean_loss
