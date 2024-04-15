import numpy as np
import random
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.optimizers import Adam
import keras.backend as K
import gymnasium as gym


class Agent(object):
    def __init__(self, ALPHA, GAMMA=0.99, n_actions=2, state_size=4):
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
        self.model = self.build_policy_network()
        self.action_space = [i for i in range(n_actions)]

    def build_policy_network(self):
        # input = Input(shape=(self.state_size,))
        # #advantages = Input(shape=[1])
        # dense1 = Dense(self.fc1_dims, activation='relu')(input)
        # dense2 = Dense(self.fc2_dims, activation='relu')(dense1)
        # probs = Dense(self.n_actions, activation='softmax')(dense2)
        #
        # policy = Model(inputs=[input, advantages], outputs=[probs])
        #
        # policy.compile(optimizer=Adam(lr=self.lr), loss='categorical_crossentropy')
        #
        # predict = Model(inputs=[input], outputs=[probs])
        #
        # return policy, predict
        model = Sequential()
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.n_actions, activation="softmax"))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.lr))
        return model

    def choose_action(self, observation):
        state = observation[np.newaxis, :]
        probabilities = self.model.predict(state, verbose=0)[0]
        #print('probabilities', probabilities)
        action = np.random.choice(self.action_space, p=probabilities)

        return action, probabilities

    def store_transition(self, observation, action, reward, probabilities):
        encoded_action = np.zeros(self.n_actions)                                            # one-hot encoding
        #print(self.action_space)
        #print('encoded_action', encoded_action)
        #print('action', action)
        encoded_action[action] = 1

        self.gradient_memory.append(np.array(encoded_action).astype('float32') - probabilities)
        self.probabilities.append(probabilities)
        self.state_memory.append(observation)
        self.action_memory.append(action)
        self.reward_memory.append(reward)

    def update_policy(self):
        #state_memory = np.array(self.state_memory)
        #action_memory = np.array(self.action_memory)
        reward_memory = np.array(self.reward_memory)

        # actions_encoded = np.zeros((len(action_memory), self.n_actions))
        # for i, action in enumerate(action_memory):
        #     actions_encoded[i, action] = 1

        G = np.zeros_like(reward_memory)
        for t in range(len(reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(reward_memory)):
                G_sum += reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum
        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        self.G = (G - mean) / std

        gradients = np.vstack(self.gradient_memory)
        rewards = np.vstack(self.G)
        print('gradients: ', gradients)
        print('rewards: ', rewards)
        gradients *= rewards
        X = np.squeeze(np.vstack([self.state_memory]))
        Y = self.probabilities + self.lr * np.squeeze(np.vstack([gradients]))
        self.model.train_on_batch(X, Y)

        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.gradient_memory = []
        self.probabilities = []


def test():
    #print('going to test')
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = Agent(state_size, action_size)
    n_episodes = 1000
    score_history = []
    for i in range(n_episodes):
        done = False
        score = 0
        s = env.reset()
        s = s[0]
        #print('s reset', s)
        while not done:
            a, prob = agent.choose_action(s)
            step_result = env.step(a)  # Execute action_t in emulator and observe next_state, reward and terminal state
            s_next, r, done, _ = step_result[:4]
            #print('s_next', s_next)

            agent.store_transition(s, a, r, prob)
            #rint(agent.action_memory)
            #s = s_next
            #print('s_next', s)
            score += r
            #print(score)
        score_history.append(score)
        agent.update_policy()
        print('episode ', i, ' score ', score, 'average_score ', np.mean(score_history[-100:]))


test()

