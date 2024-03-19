import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import AdamW
import gymnasium as gym
from matplotlib import pyplot as plt


def argmax(x):
    ''' Own variant of np.argmax with random tie breaking, function of assignment 1  '''
    try:
        return np.random.choice(np.where(x == np.max(x))[0])
    except:
        return np.argmax(x)


def softmax(x, temp):
    ''' Computes the softmax of vector x with temperature parameter 'temp', function of assignment 1 '''
    x = x / temp  # scale by temperature
    z = x - max(x)  # substract max to prevent overflow of softmax
    return np.exp(z) / np.sum(np.exp(z))  # compute softmax


class DQAgent:
    def __init__(self, state_size, action_size, replay_memory_capacity=1000, epsilon=0.5):
        self.state_size = state_size  # Set output size for Q-function
        self.action_size = action_size  # Set input size for Q-function
        self.memory = deque(maxlen=replay_memory_capacity)  # Initialize replay memory to capacity N
        self.gamma = 0.95  # Future rewards are discounted by gamma per time-step
        self.learning_rate = 0.001  # Learning rate for gradient descent
        self.action_value_function = self._build_model()  # Initialize action-value function Q with random weights
        self.epsilon = epsilon  # Probability for random action
        self.epsilon_min = 0.01
        self.episodes_max = 100

    def _build_model(self):  # Takes state as input, outputs Q values for legal actions
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=AdamW(learning_rate=self.learning_rate), metrics=['accuracy'])
        return model

    def store_transition(self, s, a, r, s_next, done):  # Stores transition in replay memory
        self.memory.append((s, a, r, s_next, done))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for s, a, r, s_next, done in minibatch:
            if done:
                y = r
            else:
                Q_next = np.amax(self.action_value_function.predict(s_next, verbose=0)[0])
                y = r + self.gamma * Q_next
            Q = self.action_value_function.predict(s, verbose=0)
            Q[0][a] = y

            self.action_value_function.fit(s, Q, verbose=0)  # Keras builds the loss function akin to equation 3 as outlined in the DQN paper.

    def select_action(self, s, current_episode, method="egreedy"):
        random_action = env.action_space.sample() # Random action
        Q_values = self.action_value_function.predict(s, verbose=0)  # Get Q-values from Q-function
        greedy_action = np.argmax(Q_values[0])  # action associated with the maximum Q-value
        if method == "egreedy":
            if self.epsilon >= np.random.rand():
                return random_action
            else:
                return greedy_action
        elif method == "anneal_epsilon":
            self.linear_annealing_epsilon(current_episode)
            if self.epsilon >= np.random.rand():
                return random_action
            else:
                return greedy_action
        elif method == "boltzmann":
            return argmax(softmax(Q_values, self.epsilon))

    def linear_annealing_epsilon(self, current_episode):
        ''' Function decay of epsilon parameter '''
        gradient_epsilon = (self.epsilon_min - 1.0) / self.episodes_max
        self.epsilon = gradient_epsilon * current_episode + 1.0


if __name__ == "__main__":

    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQAgent(state_size, action_size)
    episodes = agent.episodes_max
    batch_size = 16
    rewards = []

    for e in range(episodes):
        s, _ = env.reset()
        s = np.array([s]).reshape([1, state_size])
        done = False
        time = 0
        while not done:
            a = agent.select_action(s, e, "anneal_epsilon")
            step_result = env.step(a)  # Execute action_t in emulator and observe next_state, reward and terminal state
            s_next, r, done, _ = step_result[:4]
            s_next = np.reshape(s_next, [1, state_size])
            agent.store_transition(s, a, r, s_next, done)

            if len(agent.memory) >= batch_size:  # Only perform experience replay once memory is full
                agent.replay(batch_size)

            s = s_next
            time += 1

        print('episode:', e + 1, 'time:', time)
        rewards.append(time)
    plt.plot(rewards)
    plt.title("Reward progression ")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()
    env.close()
