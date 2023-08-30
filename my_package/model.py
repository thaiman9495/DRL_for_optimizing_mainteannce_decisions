import torch
import torch.nn as nn
import torch.nn.functional as F


class Cost_Prediction_Network(nn.Module):
    """
    Fully connected feed forward neural network
    Note:
    Cost is always posivtive, so the activation function of the output of the neural network is ReLU
    """
    def __init__(self, n_inputs, n_outputs, n_hidden_neurons):
        """
        :param n_inputs: number of inputs
        :param n_outputs: number of outputs
        :param n_hidden_neurons: an array constaining the number of neurons in all hidden layers
        """
        super().__init__()
        self.n_hidden_layers = len(n_hidden_neurons)
        self.fc = nn.ModuleList()
        for i in range(self.n_hidden_layers+1):
            # Input layer
            if i == 0:
                self.fc.append(nn.Linear(n_inputs, n_hidden_neurons[i]))
            else:
                # Output layer
                if i == self.n_hidden_layers:
                    self.fc.append(nn.Linear(n_hidden_neurons[i-1], n_outputs))
                else:
                    # Hidden layer
                    self.fc.append(nn.Linear(n_hidden_neurons[i-1], n_hidden_neurons[i]))

    def forward(self, x):
        x1 = F.relu(self.fc[0](x))
        for i in range(1, self.n_hidden_layers+1):
            x1 = F.relu(self.fc[i](x1))

        return x1


class Dueling_Deep_Q_Network(nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden_shared, hidden_value, hidden_advantage):
        super().__init__()
        self.n_hidden_layers_shared = len(hidden_shared)            # Number of hidden layers in shared module
        self.n_hidden_layers_value = len(hidden_value)              # Number of hidden layers in value module
        self.n_hidden_layers_advantage = len(hidden_advantage)      # Number of hidden layers in advatage module
        self.fc_shared = nn.ModuleList()                            # Layers in shared module
        self.fc_value = nn.ModuleList()                             # Layers in value module
        self.fc_advantage = nn.ModuleList()                         # Layers in advantage module

        # Shared module
        self.fc_shared.append(nn.Linear(n_inputs, hidden_shared[0]))
        for i in range(1, self.n_hidden_layers_shared):
            self.fc_shared.append(nn.Linear(hidden_shared[i-1], hidden_shared[i]))

        # Value module
        self.fc_value.append(nn.Linear(hidden_shared[-1], hidden_value[0]))
        for i in range(1, self.n_hidden_layers_value):
            self.fc_value.append(nn.Linear(hidden_value[i-1], hidden_value[i]))
        self.fc_value.append(nn.Linear(hidden_value[-1], 1))                                 # Ouput value function

        # Advantage module
        self.fc_advantage.append(nn.Linear(hidden_shared[-1], hidden_advantage[0]))
        for i in range(1, self.n_hidden_layers_advantage):
            self.fc_advantage.append(nn.Linear(hidden_advantage[i-1], hidden_advantage[i]))
        self.fc_advantage.append(nn.Linear(hidden_advantage[-1], n_outputs))                 # Uotput advantage fucntion

    def forward(self, state):
        # Shared module
        x = F.relu(self.fc_shared[0](state))
        for i in range(1, self.n_hidden_layers_shared):
            x = F.relu(self.fc_shared[i](x))

        # Value module
        v = F.relu(self.fc_value[0](x))
        for i in range(1, self.n_hidden_layers_value):
            v = F.relu(self.fc_value[i](v))
        v = self.fc_value[-1](v)

        # Advantage module
        a = F.relu(self.fc_advantage[0](x))
        for i in range(1, self.n_hidden_layers_advantage):
            a = F.relu(self.fc_advantage[i](a))
        a = self.fc_advantage[-1](a)

        # Output action-value function
        a_mean = torch.mean(a, dim=1, keepdim=True)
        q = v + a - a_mean

        return q


class VDN_Network(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_components, hidden_shared, hidden_value, hidden_advantage):
        super().__init__()
        self.n_layers_shared_module = len(hidden_shared)
        self.n_layers_advantage_module = len(hidden_advantage)
        self.n_layers_value_module = len(hidden_value)

        # Initialize shared module
        self.fc_shared = nn.ModuleList()
        self.fc_shared.append(nn.Linear(n_inputs, hidden_shared[0]))
        for i in range(1, self.n_layers_shared_module):
            self.fc_shared.append(nn.Linear(hidden_shared[i - 1], hidden_shared[i]))

        # Initialize value module
        self.fc_value = nn.ModuleList()
        self.fc_value.append(nn.Linear(hidden_shared[-1], hidden_value[0]))
        for i in range(1, self.n_layers_value_module):
            self.fc_value.append(nn.Linear(hidden_value[i-1], hidden_value[i]))
        self.fc_value.append(nn.Linear(hidden_value[-1], 1))

        # Initialize advantage modules
        self.fc_advantage = nn.ModuleList()
        for _ in range(n_components):
            self.fc_advantage.append(nn.ModuleList())

        for fc in self.fc_advantage:
            fc.append(nn.Linear(hidden_shared[-1], hidden_advantage[0]))
            for i in range(1, self.n_layers_advantage_module):
                fc.append(nn.Linear(hidden_advantage[i-1], hidden_advantage[i]))
            fc.append(nn.Linear(hidden_advantage[-1], n_outputs))

    def forward(self, state):
        # Shared module
        x = F.relu(self.fc_shared[0](state))
        for i in range(1, self.n_layers_shared_module):
            x = F.relu(self.fc_shared[i](x))

        # Value module
        v = F.relu(self.fc_value[0](x))
        for i in range(1, self.n_layers_value_module):
            v = F.relu(self.fc_value[i](v))
        v = self.fc_value[-1](v)

        # Advantage module
        q_list = []
        for fc in self.fc_advantage:
            a = F.relu(fc[0](x))
            for i in range(1, self.n_layers_advantage_module):
                a = F.relu(fc[i](a))
            a = fc[-1](a)
            a_mean = torch.mean(a, dim=1, keepdim=True)
            q = v + a - a_mean
            q_list.append(q)

        return q_list





