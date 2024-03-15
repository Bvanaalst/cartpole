#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import gym


# In[2]:


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  
        self.gamma = 0.95 
        self.learning_rate = 0.001
        self.model = self._build_model()
        
        
    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
    
    def buffer(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            print(target_f[0])
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
            
    def act(self, state):
        return np.argmax(self.model.predict(state)[0])
    

    
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = DQNAgent(state_size, action_size)

episodes = 10
batch_size = 32

for e in range(episodes):
    state, _ = env.reset()
    print("State shape:", state)
    state = np.array([state]).reshape([1, state_size])
    done = False
    total_reward = 0

    while not done:
        
        action = agent.act(state)
        step_result = env.step(action)
        next_state, reward, done, _ = step_result[:4]
        next_state = np.reshape(next_state, [1, state_size])
        agent.buffer(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    
    print("Episode: {}, Total Reward: {}".format(e+1, total_reward))

env.close()


# In[ ]:




