import datetime
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from my_package.model import Dueling_Deep_Q_Network, VDN_Network
from my_package.buffer import Uniform_Replay_Buffer


class Dueling_DDQN_Agent:
    def __init__(self, n_components, n_actions,
                 hidden_shared, hidden_value, hidden_advantage,
                 lr, lr_step_size, lr_gamma, lr_min, n_target,
                 buffer_batch_size, buffer_capacity,
                 gamma, device):

        # Pre-configure
        self.device = device
        self.gamma = gamma
        self.n_actions = n_actions
        self.n_target = n_target
        self.lr_min = lr_min

        # Policy netwrok
        self.policy_net = Dueling_Deep_Q_Network(n_components, n_actions,
                                                 hidden_shared, hidden_value, hidden_advantage).to(self.device)
        # print(self.policy_net)
        self.policy_net_optimizer = \
            optim.Adam([{'params': self.policy_net.fc_shared.parameters(), 'lr': lr / np.sqrt(2.0)},
                        {'params': self.policy_net.fc_value.parameters(), 'lr': lr},
                        {'params': self.policy_net.fc_advantage.parameters(), 'lr': lr}])
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.policy_net_optimizer, lr_step_size, lr_gamma)
        self.policy_net_loss = nn.MSELoss()

        # Target network
        self.target_net = Dueling_Deep_Q_Network(n_components, n_actions,
                                                 hidden_shared, hidden_value, hidden_advantage).to(self.device)

        # Target networks are always in evaluation mode
        self.target_net.eval()

        # Copy weights from the original networks to the target networks
        for param_target, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param_target.data.copy_(param.data)

        # Buffer
        self.buffer = Uniform_Replay_Buffer(buffer_capacity, n_components, buffer_batch_size, is_action_index=True)
        self.batch_size = buffer_batch_size

    def choose_action(self, state, epsilon):
        state = torch.atleast_2d(torch.FloatTensor(state).to(self.device))

        # Select action following epsilon-greedy policy
        a = np.random.uniform()
        if a < epsilon:
            action = np.random.randint(self.n_actions)
        else:
            with torch.no_grad():
                action = torch.argmax(self.policy_net(state))  # Get the action with maximal value
                action = action.item()                         # Change from tensor to integer

        return action

    def update_buffer(self, state, action, reward, next_state):
        self.buffer.push(state, np.atleast_1d(action), np.atleast_1d(reward), next_state)

    def update_policy(self, counter_step):
        # Update actor and critic network
        if len(self.buffer) > self.batch_size:
            # Sample a mini batch from relay buffer
            batch_state, batch_action, batch_reward, batch_next_state = self.buffer.sample()

            # Put these batches into GPU
            batch_state = torch.FloatTensor(batch_state).to(self.device)
            batch_action = torch.LongTensor(batch_action).to(self.device)
            batch_reward = torch.FloatTensor(batch_reward).to(self.device)
            batch_next_state = torch.FloatTensor(batch_next_state).to(self.device)

            # ---------------------------------------------------------------------------------------------------------
            # Update policy network
            # ---------------------------------------------------------------------------------------------------------

            # Compute q_policy
            q_policy = self.policy_net(batch_state).gather(1, batch_action)

            # Compute q_target
            action_max = torch.argmax(self.policy_net(batch_next_state), 1).unsqueeze(1)
            q_target = batch_reward + self.gamma * self.target_net(batch_next_state).gather(1, action_max)

            # Compute loss and update policy network
            loss = self.policy_net_loss(q_policy, q_target)
            self.policy_net.zero_grad()
            loss.backward()

            # Clip gradient
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)

            self.policy_net_optimizer.step()

            if self.lr_scheduler.get_last_lr()[-1] > self.lr_min:
                self.lr_scheduler.step()

            # ---------------------------------------------------------------------------------------------------------
            # Update target networks (hard update)
            # ---------------------------------------------------------------------------------------------------------
            if counter_step % self.n_target == 0:
                for param_target, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                    param_target.data.copy_(param.data)

            # tau = 0.001
            # for param_target, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            #     param_target.data.copy_(tau * param.data + (1.0 - tau) * param_target.data)

    def save_policy(self, step, path):
        torch.save(self.policy_net.state_dict(), path + f'policy_{step}.pt')


class VDN_Agent:
    def __init__(self, n_components, n_component_actions,
                 hidden_shared, hidden_value, hidden_advantage,
                 lr, lr_step_size, lr_gamma, lr_min, n_target,
                 buffer_batch_size, buffer_capacity, gamma, device):
        # Pre-configure
        self.device = device
        self.gamma = gamma
        self.n_component_actions = n_component_actions
        self.n_target = n_target
        self.n_components = n_components
        self.lr_min = lr_min

        # Policy network
        self.policy_net = VDN_Network(n_components, n_component_actions, n_components,
                                      hidden_shared, hidden_value, hidden_advantage).to(self.device)
        self.policy_net_optimizer =\
            optim.Adam([{'params': self.policy_net.fc_shared.parameters(), 'lr': lr / (n_components + 1)},
                        {'params': self.policy_net.fc_value.parameters(), 'lr': lr},
                        {'params': self.policy_net.fc_advantage.parameters(), 'lr': lr}
                        ])
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.policy_net_optimizer, lr_step_size, lr_gamma)
        self.policy_net_loss = nn.MSELoss()

        # Target network
        self.target_net = VDN_Network(n_components, n_component_actions, n_components,
                                      hidden_shared, hidden_value, hidden_advantage).to(self.device)
        # Target networks are always in evaluation mode
        self.target_net.eval()

        # Copy weights from the original networks to the target networks
        for param_target, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param_target.data.copy_(param.data)

        # Buffer
        self.buffer = Uniform_Replay_Buffer(buffer_capacity, n_components, buffer_batch_size,
                                                is_action_index=False)
        self.batch_size = buffer_batch_size

    def choose_action(self, state, epsilon):
        state_t = torch.atleast_2d(torch.FloatTensor(state)).to(self.device)

        # Select action following epsilon-greedy policy
        a = np.random.uniform()
        if a < epsilon:
            action = np.random.randint(self.n_component_actions, size=(self.n_components,))
        else:
            with torch.no_grad():
                q_list = self.policy_net(state_t)                      # Get output of VDN network
                action = [torch.argmax(q).item() for q in q_list]    # Get actions
                action = np.array(action, dtype=int)                 # Convert to numpy

        return action

    def update_buffer(self, state, action, reward, next_state):
        self.buffer.push(state, np.atleast_1d(action), np.atleast_1d(reward), next_state)

    def update_policy(self, counter_step):
        # Update actor and critic network
        if len(self.buffer) > self.batch_size:
            # Sample a mini batch from relay buffer
            batch_state, batch_action, batch_reward, batch_next_state = self.buffer.sample()

            # Put these batches into GPU
            batch_state = torch.FloatTensor(batch_state).to(self.device)
            batch_action = torch.LongTensor(batch_action).to(self.device)
            batch_reward = torch.FloatTensor(batch_reward).to(self.device)
            batch_next_state = torch.FloatTensor(batch_next_state).to(self.device)

            # Compute q
            q_list = self.policy_net(batch_state)
            q = q_list[0].gather(1, torch.unsqueeze(batch_action[:, 0], 1))
            for i in range(1, self.n_components):
                q += q_list[i].gather(1, torch.unsqueeze(batch_action[:, i], 1))

            # Compute q_next
            q_next_list = self.policy_net(batch_next_state)

            # Compute action max
            action_max_list = []
            for q_next in q_next_list:
                action_max_list.append(torch.argmax(q_next, 1).unsqueeze(1))

            # Compute q_target
            q_next_target_list = self.target_net(batch_next_state)
            q_next_target = q_next_target_list[0].gather(1, action_max_list[0])
            for i in range(1, self.n_components):
                q_next_target += q_next_target_list[i].gather(1, action_max_list[i])

            q_target = batch_reward + self.gamma * q_next_target

            # Compute loss and update policy network
            loss = self.policy_net_loss(q, q_target)

            self.policy_net.zero_grad()
            loss.backward()

            # Clip gradient
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)

            self.policy_net_optimizer.step()

            if self.lr_scheduler.get_last_lr()[-1] > self.lr_min:
                self.lr_scheduler.step()

            # target networks (hard update)
            if counter_step % self.n_target == 0:
                for param_target, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                    param_target.data.copy_(param.data)

    def save_policy(self, step, path):
        torch.save(self.policy_net.state_dict(), path + f'policy_{step}.pt')


class Generic_Algorithm:
    def __init__(self, n_components, n_c_states, n_iterations, population_size,
                 crossover_prob, mutation_prob, n_interventions, n_runs):
        self.n_iterations = n_iterations
        self.population_size = population_size
        self.chromonsome_size = n_components
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

        self.n_interventions = n_interventions
        self.n_runs = n_runs
        self.n_c_states = n_c_states
        self.n_components = n_components

        # Create population
        self.population = np.random.randint(1, n_c_states-1, size=(self.population_size, self.chromonsome_size))
        print(self.population)

    def _compute_fitness(self, env, threshold):
        cost_rate = np.zeros(self.n_runs)

        for i in range(self.n_runs):
            total_cost = 0.0
            env.reset()
            for _ in range(self.n_interventions):
                # Get current state
                state = env.get_state()

                # Choose an action
                action = self._choose_action(state, threshold)

                # Perform action
                _, cost = env.perform_action(action, is_action_index=False)

                total_cost += cost

            cost_rate[i] = total_cost / self.n_interventions

        return np.mean(cost_rate)

    def _choose_action(self, state, threshold):
        n_components = len(state)
        action = np.zeros(n_components, dtype=int)

        for i in range(n_components):
            if state[i] < threshold[i]:
                action[i] = 0
            else:
                if state[i] < self.n_c_states - 1:
                    action[i] = 1
                else:
                    action[i] = 2

        return action

    def _select_parent(self, score: np.ndarray):
        n_parents = int(self.population_size / 2)
        total_score = score.sum()

        temp = score.argsort()
        rank = np.empty_like(temp)
        rank[temp] = np.arange(len(score))
        max_temp = np.max(rank)
        rank = np.array([max_temp - rank[i] for i in range(len(rank))])
        rank_total = rank.sum()

        p_score = np.array([rank[i] / rank_total for i in range(self.population_size)])

        father_index = np.random.choice(range(self.population_size), (n_parents,), p=p_score)
        mother_index = np.random.choice(range(self.population_size), (n_parents,), p=p_score)

        return father_index, mother_index

    def _generate_new_population(self, father_index, mother_index):
        new_population = []
        for i in range(len(father_index)):
            father = self.population[father_index[i], :]
            mother = self.population[mother_index[i], :]

            if np.random.rand() < self.crossover_prob:
                crossover_point = np.random.randint(self.chromonsome_size)
                # Crossover
                child_1 = np.concatenate((father[:crossover_point], mother[crossover_point:]))
                child_2 = np.concatenate((mother[:crossover_point], father[crossover_point:]))

                # Muttaion
                for j in range(self.chromonsome_size):
                    if np.random.rand() < self.mutation_prob:
                        child_1[j] = np.random.randint(1, self.n_c_states-1)
                        child_2[j] = np.random.randint(1, self.n_c_states-1)
            else:
                child_1 = father.copy()
                child_2 = mother.copy()

            new_population.append(child_1)
            new_population.append(child_2)

        return np.array(new_population, dtype=int)

    def train(self, env, path_log):
        # Initialization
        best_index = 0
        best_value = self._compute_fitness(env, self.population[best_index])
        score = np.zeros(self.population_size)

        # Log
        log_iteration = []
        log_best_value = []
        log_best_chromosome = []

        starting_time = datetime.datetime.now()
        # Main loop
        for i in range(self.n_iterations):
            # Evaluate all chromonsomes in the popolation
            for j in range(self.population_size):
                score[j] = self._compute_fitness(env, self.population[j])

            # Obtain new best chromonsome
            best_index = np.argmin(score)
            best_value = score[best_index]

            print(f'iteration {i}: threshold: {self.population[best_index]} --> cost rate: {best_value}')

            # Hold values for logging
            log_iteration.append(i)
            log_best_value.append(best_value)
            log_best_chromosome.append(self.population[best_index])

            # Select parents
            father_index, mother_index = self._select_parent(score)

            # Generate new generation
            self.population = self._generate_new_population(father_index, mother_index)

        ending_time = datetime.datetime.now()
        training_time = ending_time - starting_time
        print(f"Training time: {training_time}")

        torch.save(log_iteration, path_log + '/iteration.pt')
        torch.save(log_best_value, path_log + '/cost_rate.pt')
        torch.save(log_best_chromosome, path_log + '/threshold.pt')
        torch.save(training_time, path_log + '/training_time.pt')

        plt.plot(log_iteration, log_best_value)
        plt.show()






