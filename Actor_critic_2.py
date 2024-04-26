import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras import backend as K

import gymnasium as gym


class CustomEntropyRegularization:
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, true_actions, predicted_actions):
        policy_entropy = -K.mean(K.sum(predicted_actions * K.log(predicted_actions + 1e-10), axis=-1))
        return self.weight * policy_entropy


class Agent(object):
    def __init__(self, alpha=1e-4, gamma=0.99, n_actions=2, state_size=4, n=5, baseline=True, bootstrap=True):
        self.gamma = gamma
        self.lr = alpha
        self.G = 0
        self.n = n
        self.state_size = state_size
        self.n_actions = n_actions
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.gradient_memory = []
        self.probabilities = []
        self.model_actor = self._build_actor_model()
        self.model_critic = self._build_critic_model()
        self.action_space = [i for i in range(n_actions)]
        self.baseline = baseline
        self.bootstrap = bootstrap

    def _build_actor_model(self): # Takes state as input, outputs probability legal actions
        model = Sequential()
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.n_actions, activation="softmax"))

        # Define the custom entropy regularization loss
        entropy_loss = CustomEntropyRegularization(weight=0.01)  # Adjust the weight as needed

        # Compile the model with the combined loss
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.lr),
                      loss_weights=[1., entropy_loss])  # Use entropy regularization
        return model

    def _build_critic_model(self): # Takes state as input, outputs Q value for critic correction
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.lr), metrics=['accuracy'])
        return model

    def choose_action(self, observation):
        state = observation[np.newaxis, :] #Predict action probabilities based on the current state
        probabilities = self.model_actor.predict(state, verbose=0)[0]
        action = np.random.choice(self.action_space, p=probabilities) # Choose an action randomly based on the predicted probabilities

        return action, probabilities

    def store_transition(self, observation, action, reward, probabilities):
        encoded_action = np.zeros(self.n_actions)                                            # one-hot encoding
        encoded_action[action] = 1

        self.state_memory.append(observation)
        self.action_memory.append(encoded_action)
        self.reward_memory.append(reward)

    def update_policy(self):
        reward_memory = np.array(self.reward_memory)
        states = np.squeeze(np.vstack([self.state_memory]))
        actions = np.squeeze(np.vstack(self.action_memory))
        #values = self.model_critic(states)[:, 0]
        values_2 = np.squeeze(self.model_critic.predict(states))
        G = np.zeros_like(reward_memory)

        for t in range(len(reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(reward_memory)):
                G_sum += reward_memory[k-1] * discount
                discount *= self.gamma
            G[t] = G_sum
            n = min(self.n, len(reward_memory) - t)
            if self.bootstrap:
                G[t] += (self.gamma ** n) * values_2[t + n - 1] # Bootstrapping
        mean = np.mean(G)
        std = np.std(G) + 1e-9
        self.G = (G - mean) / std

        if self.baseline:
            advantages = self.G - values_2 # Baseline subtraction
            self.model_actor.train_on_batch(states, actions, sample_weight=advantages)
        else:
            self.model_actor.train_on_batch(states, actions, sample_weight=self.G)

        self.model_critic.train_on_batch(states, self.G)
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []


def actor_critic(n_episodes, learning_rate, gamma, n, baseline, bootstrap):
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = Agent(alpha=learning_rate, gamma=gamma, n_actions=action_size, state_size=state_size, n=n,
                  baseline=baseline, bootstrap=bootstrap)
    eval_timesteps = []
    eval_returns = []
    eval_avg_returns = []
    for i in range(n_episodes):
        done = False
        score = 0
        s = env.reset()
        s = s[0]
        while not done:
            a, prob = agent.choose_action(s)
            step_result = env.step(a)  # Execute action_t in emulator and observe next_state, reward and terminal state
            s_next, r, done, _ = step_result[:4]
            agent.store_transition(s, a, r, prob)
            s = s_next
            score += r
        agent.update_policy()
        eval_timesteps.append(i)
        eval_returns.append(score)
        eval_avg_returns.append(np.mean(eval_returns[-100:]))
        print('episode ', i, ' score ', score, 'average_score ', np.mean(eval_returns[-100:]))
    return np.array(eval_returns), np.array(eval_timesteps), np.array(eval_avg_returns)
