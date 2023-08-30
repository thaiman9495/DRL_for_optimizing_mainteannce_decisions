import torch
import torch.nn as nn
import torch.utils.data as data
import pandas as pd
import datetime

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from my_package.model import Cost_Prediction_Network
from my_package.utility import compute_mse

# Define important parameters
n_epochs = 500                        # Number of epochs for training
learning_rate = 0.005                 # Initial learning
lr_step = 100                         # Learning rate is decayed after every lr_step
lr_decay_constant = 0.6               # Learning rate is multiplied with ler_decay_constant after every lr_step
batch_size = 32                       # Size of input batch for training
eval_batch_size = 256                 # Batch size for evaluation mode --> speed up GPU
n_epoch_log = 2                       # After n_step_log epochs, model will be evaluated
hidden_layers = [100, 100]            # Number of neurons of hidden layers

n_components = 15

# Set training device: GPU or CPU
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("This program is trained on GPU\n")
else:
    device = torch.device("cpu")
    print("This program is trained on CPU\n")

# Load dataset using DataFrame
df = pd.read_excel('maintenance_data.xlsx')
n_inspections = len(df)                                   # Get length of dataset
n_samples_training = int(0.8*n_inspections)               # Number of samples for training
n_samples_testing = n_inspections - n_samples_training    # Number of samples for testing

# Separate features and target form data and convert to numpy
X = df.drop(['System_status', 'Cost'], axis=1).to_numpy()       # Feature
y = df['Cost'].to_numpy()                                       # Target

# Create tensor dataset
dataset = data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))

# Split data into training and evaluation set
train_data, eval_data = data.random_split(dataset, [n_samples_training, n_samples_testing])

# Create data loader
train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
eval_loader = data.DataLoader(eval_data, batch_size=eval_batch_size)

train_loader_eval = data.DataLoader(train_data, batch_size=eval_batch_size)  # Speed up by GPu


# Configure neural network
cost_model = Cost_Prediction_Network(2*n_components, 1, hidden_layers).to(device)

# Configure loss function and optimizer
loss_function = nn.MSELoss()
optimizer = Adam(cost_model.parameters(), lr=learning_rate)
lr_scheduler = StepLR(optimizer=optimizer, step_size=lr_step, gamma=lr_decay_constant)

# Define some list for illustration purpose
train_loss_list = []
test_loss_list = []
epoch_list = []

# Start training
starting_time = datetime.datetime.now()

for epoch in range(1, n_epochs+1):
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.unsqueeze(1).to(device)             # Increase the size of tensor

        cost_model.zero_grad()                                # Clear gradient
        y_estimated = cost_model(x_batch)                     # Compute the network's output
        loss = loss_function(y_estimated, y_batch)            # Compute loss
        loss.backward()                                       # Back propagation
        optimizer.step()                                      # Update network's parameters

    if epoch > 20 and epoch % n_epoch_log == 0:
        print("Training --------------------------------------->")
        print(f"Epoch {epoch}")

        train_loss = compute_mse(cost_model, train_loader_eval, n_samples_training, device)
        test_lost = compute_mse(cost_model, eval_loader, n_samples_testing, device)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_lost)

        epoch_list.append(epoch)

        print(f"Loss in training set:       {train_loss}")
        print(f"Loss in testing set:        {test_lost}")

        print("---------------------------------------> Completed\n")

    # Change learning rate
    lr_scheduler.step()

torch.save(cost_model, 'data_holder/cost_model.pt')
torch.save(train_loss_list, 'data_holder/loss_train.pt')
torch.save(test_loss_list, 'data_holder/loss_test.pt')
torch.save(epoch_list, 'data_holder/epoch.pt')

ending_time = datetime.datetime.now()
print(f"Time for training: {ending_time - starting_time}")





