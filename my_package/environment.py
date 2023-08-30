import torch
import numpy as np

from itertools import product


class Parameter:
    def __init__(self):
        self.n_components = 15                          # Number of components
        self.n_component_states = 5                    # Number of states per component

        # Parameter for computing cost
        self.c_s = 20.0                                # Set up cost for system
        # Inspection cost for each component
        self.c_ins = [5.0, 2.0, 2.0, 3.0, 3.0, 3.0,  4.0, 4.0, 4.0, 4.0,  3.0, 3.0, 3.0, 3.0, 3.0]
        # Cost for replacement (perfect maintenance)
        self.c_r = [70.0, 65.0, 65.0, 50.0, 50.0, 50.0,  55.0, 55.0, 55.0, 55.0, 45.0, 45.0, 45.0, 45.0, 45.0]
        # Imperfect maintenance constant
        self.eta = [3.0, 2.0, 2.0,  1.0, 1.0, 1.0,  4.0, 4.0, 4.0, 4.0,  5.0, 5.0, 5.0, 5.0, 5.0]
        self.c_dt = 800.0                              # Downtime cost
        self.c_penalty = 1500.0                         # Cost for penalizing wrong action

        # Transition matrices
        self.transition_matrices = np.zeros((self.n_components, self.n_component_states, self.n_component_states))
        P_1 = np.array([[0.30, 0.30, 0.20, 0.15, 0.05],
                        [0.00, 0.20, 0.30, 0.30, 0.20],
                        [0.00, 0.00, 0.30, 0.40, 0.30],
                        [0.00, 0.00, 0.00, 0.40, 0.60],
                        [0.00, 0.00, 0.00, 0.00, 1.00]])
        P_2 = np.array([[0.10, 0.30, 0.30, 0.20, 0.10],
                        [0.00, 0.10, 0.30, 0.30, 0.30],
                        [0.00, 0.00, 0.30, 0.30, 0.40],
                        [0.00, 0.00, 0.00, 0.20, 0.80],
                        [0.00, 0.00, 0.00, 0.00, 1.00]])
        P_3 = np.array([[0.20, 0.20, 0.30, 0.30, 0.00],
                        [0.00, 0.10, 0.30, 0.30, 0.30],
                        [0.00, 0.00, 0.20, 0.30, 0.50],
                        [0.00, 0.00, 0.00, 0.30, 0.70],
                        [0.00, 0.00, 0.00, 0.00, 1.00]])
        P_4 = np.array([[0.25, 0.30, 0.20, 0.20, 0.05],
                        [0.00, 0.10, 0.20, 0.30, 0.40],
                        [0.00, 0.00, 0.20, 0.30, 0.50],
                        [0.00, 0.00, 0.00, 0.20, 0.80],
                        [0.00, 0.00, 0.00, 0.00, 1.00]])
        P_5 = np.array([[0.25, 0.20, 0.30, 0.20, 0.05],
                        [0.00, 0.15, 0.20, 0.40, 0.25],
                        [0.00, 0.00, 0.20, 0.40, 0.40],
                        [0.00, 0.00, 0.00, 0.20, 0.80],
                        [0.00, 0.00, 0.00, 0.00, 1.00]])

        self.transition_matrices[0, :, :] = P_1

        self.transition_matrices[1, :, :] = P_2
        self.transition_matrices[2, :, :] = P_2

        self.transition_matrices[3, :, :] = P_3
        self.transition_matrices[4, :, :] = P_3
        self.transition_matrices[5, :, :] = P_3

        self.transition_matrices[6, :, :] = P_4
        self.transition_matrices[7, :, :] = P_4
        self.transition_matrices[8, :, :] = P_4
        self.transition_matrices[9, :, :] = P_4

        self.transition_matrices[10, :, :] = P_5
        self.transition_matrices[11, :, :] = P_5
        self.transition_matrices[12, :, :] = P_5
        self.transition_matrices[13, :, :] = P_5
        self.transition_matrices[14, :, :] = P_5

    @staticmethod
    def structure_function(s):
        # s is a vector containing states of all components
        # s[i] = 0 ---> Component is failed at inspection time
        # s[i] = 1 ---> Component is functioning at inspection time

        sub_1 = s[0]
        sub_2 = 1 - (1 - s[1]) * (1 - s[2])
        sub_3 = 1 - (1 - s[3]) * (1 - s[4]) * (1 - s[5])
        sub_4 = 1 - (1 - s[6]) * (1 - s[7]) * (1 - s[8]) * (1 - s[9])
        sub_5 = 1 - (1 - s[10]) * (1 - s[11]) * (1 - s[12]) * (1 - s[13]) * (1 - s[14])
        return sub_1 * sub_2 * sub_3 * sub_4 * sub_5

    def transition_function(self, state, next_state):
        transition_probability = 1.0
        for i in range(self.n_components):
            transition_probability *= self.transition_matrices[i, state[i], next_state[i]]

        return transition_probability

    def compute_reliability(self, state):
        # r is array consisting of realibility of all components
        r = np.zeros((self.n_components,))
        for i in range(self.n_components):
            r[i] = 1.0 - self.transition_matrices[i, state[i], self.n_component_states - 1]

        # Compute system's reliability
        r_system = self.structure_function(r)
        return r_system

    def compute_potential_downtime(self, state):
        return self.c_dt * (1.0 - self.compute_reliability(state))


class Component:
    def __init__(self, transition_matrix: np.ndarray):
        self.transition_matrix = transition_matrix           # Transition matrix
        self.n_states = transition_matrix.shape[0]           # Number of states
        self.state = 0                                       # Initialize component's state as new
        self.state_space = np.arange(self.n_states)          # State space

    def degrade(self):
        """
        This function describes the degradtion process of a component according to its transition matrix
        :return: state after degredation
        """
        self.state = np.random.choice(self.state_space, p=self.transition_matrix[self.state, :])
        return self.state

    def reset(self):
        """
        This function resets the state of a component to as good as new (state number zero)
        :return: state
        """
        self.state = 0
        return self.state

    def set_state(self, state):
        self.state = state

    def perform_action(self, action: int):
        """
        This objective of this function is to calculate the state after maintenance of a component
        :param action: action must be integer and its value is less than or equal to
         the value of the component's current state
        :return: State after maintenance
        """
        self.state -= action
        return self.state


class System(Parameter):
    def __init__(self):
        super().__init__()
        # A list for holding all components
        self.component = [Component(self.transition_matrices[i]) for i in range(self.n_components)]

        # Initialize all components as new
        self.state = np.zeros((self.n_components,), dtype=int)

    def degrade(self):
        """ This function compute system' next state, s_{k+1},  from its state after maintenance s'_k """
        for i in range(self.n_components):
            self.state[i] = self.component[i].degrade()

    def reset(self):
        """ Reset all components to as good as new (state number zero) """
        for i in range(self.n_components):
            self.state[i] = self.component[i].reset()

    def get_state(self):
        """Get the system's current state"""
        return self.state.copy()

    def set_state(self, state: np.ndarray):
        """Set system to the specific state"""
        self.state = state
        for i in range(self.n_components):
            self.component[i].set_state(state[i])

    def system_failed(self, state):
        """
        This function aims at checking wheather the system is failed at inspection time
        :param state: system's current state
        :return: 0 if the system sitll functioning or 1 if the system is in failed state
        """

        # s is the array with size of n_components elements
        # Each element of s expresses component's state corresponding its order
        # If s[i] = 0 ---> Component i is in failed state
        # If s[i] = 1 ---> Component i is still functioning
        s = np.ones((self.n_components,))
        for i in range(self.n_components):
            if state[i] == self.n_component_states - 1:
                s[i] = 0

        # Check whether the system is still functioning
        # s_system = 0 ---> The system is still functioning
        # s_system = 1 ---> The system fails
        s_system = 1.0 - float(self.structure_function(s))

        return s_system

    def perform_action(self, action, is_first_step=False):
        pass


class System_Data_Generation(System):
    """ This class is used to generate data for maintenance cost forcasting """
    def __init__(self):
        super().__init__()

    def compute_maintenance_cost(self, state, action, is_fist_step=False):
        if is_fist_step:
            # At the begining, the system is assumed to be new, so we don't need to inspect the system
            # As a result, inspection cost is equal to zero at that moment
            maintenance_cost = 0
        else:
            # Indicator for setup cost
            I_setup = 0.0 if action.sum() == 0 else 1.0

            # Compute maintenance cost of all components
            c_m = 0.0
            for i in range(self.n_components):
                if state[i] != 0:
                    c_m += self.c_r[i] * (action[i] / state[i]) ** self.eta[i]

            maintenance_cost = c_m + I_setup * self.c_s + sum(self.c_ins)

        return maintenance_cost

    def perform_action(self, action, is_first_step=False):
        """
        :param action: system's state before maintenance
        :param is_first_step: first inspection
        :return: system's status (fail or not), system's state after maintenance, maintenance cost
        """

        # Check whether system fails or not
        I_f = self.system_failed(self.state)

        # Compute maintenance cost of all components
        maintenance_cost = self.compute_maintenance_cost(self.state, action, is_first_step)

        # Implement action
        for i in range(self.n_components):
            self.state[i] = self.component[i].perform_action(action[i])

        # Save state aftermaintenance
        state_am = self.state.copy()

        # Let system degrade naturally
        self.degrade()

        return I_f, state_am, maintenance_cost


class System_Deterministic_IM(System):
    def __init__(self, transition_matrices, cost_model, device):
        super().__init__()
        # Estimated transition matrices
        self.transition_matrices = transition_matrices.copy()

        # Trained cost model
        self.cost_model = cost_model

        # GPU or CPU
        self.device = device

        # Create action space
        self.action_space = np.array(tuple(product(range(self.n_component_states), repeat=self.n_components)),
                                     dtype=int)

        # Number of all possible actions
        self.n_actions = len(self.action_space)

    def implement_deterministic_action(self, action, is_action_wrong=False):
        """
        action is determistic and not the index
        """

        if is_action_wrong:
            # Penalty cost
            cost = self.c_penalty

            # Set all component in failed state at next inspection
            self.set_state(np.ones(self.n_components, dtype=int) * (self.n_component_states - 1))

            return self.state.copy(), cost

        else:
            # Save state before maintenance
            state_bm = self.state.copy()

            # Check whether the system is failed at inspection time
            I_dt = self.system_failed(state_bm)

            # Implement action
            for i in range(self.n_components):
                self.state[i] = self.component[i].perform_action(action[i])

            # Compute maintenance cost
            my_input = torch.tensor(np.append(state_bm, self.state), dtype=torch.float32).to(self.device)
            c_m = self.cost_model(my_input).item()

            # Compute total cost
            cost = c_m + I_dt * self.c_dt

            # Let the system degrade gradually
            self.degrade()

            return self.state.copy(), cost

    def perform_action(self, action_in, is_action_index=True, is_first_step=False):
        # Get action using action index
        if is_action_index:
            action = self.action_space[action_in, :]
        else:
            action = action_in

        # Check whether action wrong or not
        is_action_wrong = False
        for i in range(self.n_components):
            if self.state[i] < action[i]:
                is_action_wrong = True

        # Perform action
        next_state, cost = self.implement_deterministic_action(action, is_action_wrong)

        return next_state, cost

    def reward_function(self, state, action):
        # Check whether system is failed at inspection time
        I_dt = self.system_failed(state)

        # Compute system's state after maintenance
        state_am = state - action

        # Compute maintenance cost
        my_input = torch.tensor(np.append(state, state_am), dtype=torch.float32).to(self.device)
        c_m = self.cost_model(my_input).item()

        # Compute total cost
        cost = c_m + I_dt * self.c_dt

        return -cost


class System_Random_IM(System_Deterministic_IM):
    def __init__(self, transition_matrices, cost_model, device):
        super().__init__(transition_matrices, cost_model, device)

        # Create action space
        self.action_space = np.array(tuple(product(range(3), repeat=self.n_components)), dtype=int)

        # Number of all possible actions
        self.n_actions = len(self.action_space)

    def perform_action(self, action_in, is_action_index=True, is_first_step=False):
        # Get random action from action index
        if is_action_index:
            random_action = self.action_space[action_in, :]
        else:
            random_action = action_in

        # Check if the random action is wrong or not
        is_action_wrong = False
        for i in range(self.n_components):
            if self.state[i] == 0 and random_action[i] != 0:
                is_action_wrong = True

        # Get deterministic action from random action
        action = np.zeros((self.n_components,), dtype=int)
        for i in range(self.n_components):
            if random_action[i] == 0:
                action[i] = 0
            else:
                if random_action[i] == 1:
                    action[i] = np.random.randint(self.state[i] + 1)
                else:
                    action[i] = self.state[i]

        # Implement deterministic action
        next_state, cost = self.implement_deterministic_action(action, is_action_wrong)

        return next_state, cost


class System_Deterministic_IM_Ground_Truth(System_Data_Generation):
    def __init__(self, alpha=0.1):
        super().__init__()

        old_transition_matrices = self.transition_matrices.copy()

        for i in range(self.n_components):
            for j in range(self.n_component_states - 1):
                self.transition_matrices[i, j, :] = old_transition_matrices[i, j, :] * (1.0 - alpha)
                self.transition_matrices[i, j, j] = 1.0 - np.sum(self.transition_matrices[i, j, j + 1:])

        # Create action space
        # self.action_space = np.array(tuple(product(range(self.n_component_states), repeat=self.n_components)),
        #                              dtype=int)

        # Number of all possible actions
        # self.n_actions = len(self.action_space)
        # self.n_actions = len(self.action_space)

    def perform_action(self, action_in, is_action_index=True, is_first_step=False):
        # Get action using action index
        # if is_action_index:
        #     action = self.action_space[action_in, :]
        # else:
        #     action = action_in

        action = action_in
        # Check wheather action wrong or not
        is_action_wrong = False
        for i in range(self.n_components):
            if self.state[i] < action[i]:
                is_action_wrong = True

        # Perform action
        if is_action_wrong:
            # Penalty cost
            cost = self.c_penalty

            # Set all component in failed state at next inspection
            self.set_state(np.ones(self.n_components, dtype=int) * (self.n_component_states - 1))

            return self.state.copy(), cost
        else:
            # Check whether system fails or not
            I_dt = self.system_failed(self.state)

            # Compute maintenance cost
            maintenance_cost = self.compute_maintenance_cost(self.state, action, is_first_step)

            # Compute total cost
            cost = maintenance_cost + I_dt * self.c_dt

            # Implement action
            for i in range(self.n_components):
                self.state[i] = self.component[i].perform_action(action[i])

            # Let system degrade naturally
            self.degrade()

            return self.state.copy(), cost

    def reward_function(self, state, action):
        # Check whether system is failed at inspection time
        I_dt = self.system_failed(state)

        # Compute system's state after maintenance
        state_am = state - action

        # Compute maintenance cost
        c_m = self.compute_maintenance_cost(state, action)

        # Compute total cost
        cost = c_m + I_dt * self.c_dt

        return -cost


class System_Random_IM_Ground_Truth(System_Data_Generation):
    def __init__(self, alpha=0.1):
        super().__init__()

        old_transition_matrices = self.transition_matrices.copy()

        for i in range(self.n_components):
            for j in range(self.n_component_states - 1):
                self.transition_matrices[i, j, :] = old_transition_matrices[i, j, :] * (1.0 - alpha)
                self.transition_matrices[i, j, j] = 1.0 - np.sum(self.transition_matrices[i, j, j + 1:])

        # Create action space
        self.action_space = np.array(tuple(product(range(3), repeat=self.n_components)), dtype=int)

        # Number of all possible actions
        self.n_actions = len(self.action_space)

    def perform_action(self, action_in, is_action_index=True, is_first_step=False):
        # Get random action from action index
        if is_action_index:
            random_action = self.action_space[action_in, :]
        else:
            random_action = action_in

        # Check if the random action is wrong or not
        is_action_wrong = False
        for i in range(self.n_components):
            if self.state[i] == 0 and random_action[i] != 0:
                is_action_wrong = True

        # Get deterministic action from random action
        action = np.zeros((self.n_components,), dtype=int)
        for i in range(self.n_components):
            if random_action[i] == 0:
                action[i] = 0
            else:
                if random_action[i] == 1:
                    action[i] = np.random.randint(self.state[i] + 1)
                else:
                    action[i] = self.state[i]

        # Perform action
        if is_action_wrong:
            # Penalty cost
            cost = self.c_penalty

            # Set all component in failed state at next inspection
            self.set_state(np.ones(self.n_components, dtype=int) * (self.n_component_states - 1))

            return self.state.copy(), cost
        else:
            # Check whether system fails or not
            I_dt = self.system_failed(self.state)

            # Compute maintenance cost
            maintenance_cost = self.compute_maintenance_cost(self.state, action, is_first_step)

            # Compute total cost
            cost = maintenance_cost + I_dt * self.c_dt

            # Implement action
            for i in range(self.n_components):
                self.state[i] = self.component[i].perform_action(action[i])

            # Let system degrade naturally
            self.degrade()

            return self.state.copy(), cost




