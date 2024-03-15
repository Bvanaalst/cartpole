
import numpy as np
import gymnasium as gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from collections import deque
import random


class DeepQAgent:

    def __init__(self):
        self.model = self.create_Q()
        self.lr = 0.001
        self.gamma = 0.95
        self.buffer = deque(maxlen=1000)
        self.epsilon = 1
        #self.model = self.create_Q()

    def append_buffer(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def select_action(self, s):
        if self.epsilon >= np.random.rand():
            return env.action_space.sample()
        else:
            a = self.model.predict(s)
            return np.argmax(a[0])

    def replay(self, batch_size):
        if len(self.buffer) < batch_size:
            return
        batch = random.sample(self.buffer, batch_size)

        for s, a, r, s_next, done in batch:
            if done:
                Q_target = r
            else:
                Q_target = r + self.gamma * np.max(self.model.predict(s_next).flatten())

            Q_train = self.model.predict(s)
            Q_train[0][a] = Q_target
            self.model.fit(s, Q_train, verbose=0)

    def create_Q(self):
        model = Sequential()
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(2, activation='linear'))
        model.compile(optimizer=RMSprop(rho=0.95, epsilon=0.01), loss="mse", metrics=['accuracy'])
        return model



env = gym.make('CartPole-v1', render_mode="human")
agentQ = DeepQAgent()



#print(env.observation_space.shape[0])
print(env.action_space)
print(env.observation_space.shape[0])#[0]

episodes = 20

for e in range(episodes):
    s = env.reset()

    a = agentQ.select_action(s)

    s_next, r, done, truncated, info = env.step(a)
    agentQ.append_buffer(s, a, s_next, r, done)
    s = s_next
    agentQ.replay(10)
    if done:
        print('done')







# env.action_space.seed(42)
#
# observation, info = env.reset(seed=42)
#
 #for _ in range(1000):
#     observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
#
#     if terminated or truncated:
#         observation, info = env.reset()
#
# env.close()
#
# #a = np.array
