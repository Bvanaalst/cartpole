import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import gymnasium as gym
from matplotlib import pyplot as plt


class Agent(object):
    def __init__(self, alpha=1e-4, gamma=0.99, n_actions=2, state_size=4):
        self.gamma = gamma
        self.lr = alpha
        self.G = 0
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

    def _build_actor_model(self):
        model = Sequential()
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.n_actions, activation="softmax"))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.lr))
        return model

    def _build_critic_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.lr), metrics=['accuracy'])
        return model

    def choose_action(self, observation):
        state = observation[np.newaxis, :]
        probabilities = self.model_actor.predict(state, verbose=0)[0]
        action = np.random.choice(self.action_space, p=probabilities)

        return action, probabilities

    def store_transition(self, observation, action, reward, probabilities):
        encoded_action = np.zeros(self.n_actions)                                            # one-hot encoding
        encoded_action[action] = 1

        self.state_memory.append(observation)
        self.action_memory.append(encoded_action)
        self.reward_memory.append(reward)

    def update_policy(self):
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

        states = np.squeeze(np.vstack([self.state_memory]))
        actions = np.squeeze(np.vstack(self.action_memory))

        values = self.model_critic(states)[:, 0]
        #print(values)

        advantages = self.G - values
        #print("states: ", states)
        #print('actions: ', actions)
        #print('G: ', self.G)
        self.model_actor.train_on_batch(states, actions, sample_weight=advantages)
        self.model_critic.train_on_batch(states, self.G)
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []


def test():
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = Agent(n_actions=action_size, state_size=state_size)
    n_episodes = 500
    score_history = []
    average_score = []
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
        score_history.append(score)
        agent.update_policy()
        print('episode ', i, ' score ', score, 'average_score ', np.mean(score_history[-100:]))
        average_score.append(np.mean(score_history[-100:]))
    plt.scatter(score_history, color='blue')
    plt.plot(average_score, color='red')
    plt.title("Reward progression ")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()
test()

