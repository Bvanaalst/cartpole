import numpy as np
import random
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.optimizers import Adam
import keras.backend as K
import gymnasium as gym


class Agent(object):
    def __init__(self, ALPHA=1e-4, GAMMA=0.99, n_actions=2, state_size=4):
        self.gamma = GAMMA
        self.lr = ALPHA
        self.G = 0
        self.state_size = state_size
        self.n_actions = n_actions
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.gradient_memory = []
        self.probabilities = []
        self.model = self._build_model()
        self.action_space = [i for i in range(n_actions)]

    def _build_model(self):
        model = Sequential()
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.n_actions, activation="softmax"))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.lr))
        return model

    def choose_action(self, observation):
        state = observation[np.newaxis, :]
        probabilities = self.model.predict(state, verbose=0)[0]
        action = np.random.choice(self.action_space, p=probabilities)

        return action, probabilities

    def store_transition(self, observation, action, reward, probabilities):
        encoded_action = np.zeros(self.n_actions)                                            # one-hot encoding
        encoded_action[action] = 1

        self.gradient_memory.append(np.array(encoded_action).astype('float32') - probabilities)
        self.probabilities.append(probabilities)
        self.state_memory.append(observation)
        self.action_memory.append(encoded_action)
        self.reward_memory.append(reward)

    def update_policy(self):
        #state_memory = np.array(self.state_memory)
        #action_memory = np.array(self.action_memory)
        reward_memory = np.array(self.reward_memory)


        G = np.zeros_like(reward_memory)
        for t in range(len(reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(reward_memory)):
                G_sum += reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum
        mean = np.mean(G)
        std = np.std(G) + 1e-9
        self.G = (G - mean) / std

        gradients = np.vstack(self.gradient_memory)
        discounted_rewards = np.vstack(self.G)
        gradients *= discounted_rewards
        states = np.squeeze(np.vstack([self.state_memory]))
        actions = np.squeeze(np.vstack(self.action_memory))
        Y = self.probabilities + self.lr * np.vstack([gradients])

        #self.model.train_on_batch(states, Y)
        self.model.train_on_batch(states, actions, sample_weight=self.G)

        #print('states: ', states)
        #print('actions: ', actions)
        #print('G: ', self.G)
        #print('rewards', reward_memory)
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.gradient_memory = []
        self.probabilities = []


def test():
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = Agent()
    n_episodes = 2000
    score_history = []
    for i in range(n_episodes):
        done = False
        score = 0
        s = env.reset()
        s = s[0]
        while not done:
            a, prob = agent.choose_action(s)
            step_result = env.step(a)  # Execute action_t in emulator and observe next_state, reward and terminal state
            s_next, r, done, _ = step_result[:4]
            #print('s_next', s_next)

            agent.store_transition(s, a, r, prob)

            score += r

        score_history.append(score)
        agent.update_policy()
        print('episode ', i, ' score ', score, 'average_score ', np.mean(score_history[-100:]))


test()

