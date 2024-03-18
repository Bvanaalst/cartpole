import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import gymnasium as gym
from matplotlib import pyplot as plt


class DQAgent:
    def __init__(self, state_size, action_size, replay_memory_capacity=1000, epsilon=0.5):
        self.state_size = state_size  # Set output size for Q-function
        self.action_size = action_size  # Set input size for Q-function
        self.memory = deque(maxlen=replay_memory_capacity)  # Initialize replay memory to capacity N
        self.gamma = 0.95  # Future rewards are discounted by gamma per time-step
        self.learning_rate = 0.001  # Learning rate for gradient descent
        self.action_value_function = self._build_model()  # Initialize action-value function Q with random weights
        self.epsilon = epsilon  # Probability for random action
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def _build_model(self):  # Takes state as input, outputs Q values for legal actions
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='tanh'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
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

            self.action_value_function.fit(s, Q, verbose=0) # Keras builds the loss function akin to equation 3 as outlined in the DQN paper.

    def select_action(self, s):
        if self.epsilon >= np.random.rand():
            return env.action_space.sample()  # Return a random action
        else:
            a = self.action_value_function.predict(s, verbose=0)  # Get Q-values from Q-function
            return np.argmax(a[0])  # Return the action associated with the maximum Q-value

    def annealing_epsilon(self, ):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


if __name__ == "__main__":

    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQAgent(state_size, action_size)
    # print(state_size)
    episodes = 50
    batch_size = 16
    rewards = []

    for e in range(episodes):
        s, _ = env.reset()
        # print('s1: ', s)
        # print("State shape:", s)
        s = np.array([s]).reshape([1, state_size])
        # print('s2: ', s)
        done = False
        time = 0
        while not done:
            a = agent.select_action(s)
            step_result = env.step(a)  # Execute action_t in emulator and observe next_state, reward and terminal state
            s_next, r, done, _ = step_result[:4]
            s_next = np.reshape(s_next, [1, state_size])
            agent.store_transition(s, a, r, s_next, done)
            state = s_next

            if len(agent.memory) > batch_size: # Only perform experience replay once memory is full
                agent.replay(batch_size)

            time += 1
            agent.annealing_epsilon()

        print('episode:', e + 1, 'time:', time)

        # print("Episode: {}, Total Reward: {}".format(e+1, total_reward))
    print(rewards)
    plt.plot(rewards)
    env.close()
