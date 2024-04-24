import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras import backend as K
import gymnasium as gym
from matplotlib import pyplot as plt


class CustomEntropyRegularization:
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, true_actions, predicted_actions):
        policy_entropy = -K.mean(K.sum(predicted_actions * K.log(predicted_actions + 1e-10), axis=-1))
        return self.weight * policy_entropy

class Agent(object):
    def __init__(self, alpha=1e-4, gamma=0.99, n_actions=2, state_size=4):
        self.gamma = gamma # Future rewards are discounted by gamma per time-step
        self.lr = alpha # Learning rate for gradient ascent
        self.G = 0
        self.state_size = state_size
        self.n_actions = n_actions # Set output size for policy network
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.gradient_memory = []
        self.probabilities = []
        self.model = self._build_model()
        self.action_space = [i for i in range(n_actions)] # Used for picking a random action

    def _build_model(self): # Takes state as input, outputs probability legal actions
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

    def choose_action(self, observation):
        state = observation[np.newaxis, :]    # Predict action probabilities based on the current state
        probabilities = self.model.predict(state, verbose=0)[0]
        action = np.random.choice(self.action_space, p=probabilities) # Choose an action randomly based on the predicted probabilities

        return action, probabilities

    def store_transition(self, observation, action, reward, probabilities):
        encoded_action = np.zeros(self.n_actions)   # one-hot encoding
        encoded_action[action] = 1

        self.state_memory.append(observation)
        self.action_memory.append(encoded_action)
        self.reward_memory.append(reward)

    def update_policy(self):
        reward_memory = np.array(self.reward_memory)

        G = np.zeros_like(reward_memory)  # Initialize an array to store the discounted returns

        for t in range(len(reward_memory)): # Calculate the discounted returns (G)
            G_sum = 0 # Initialize the sum of discounted rewards
            discount = 1 # Initialize the discount factor
            for k in range(t, len(reward_memory)):
                G_sum += reward_memory[k] * discount # Calculate the discounted sum of rewards from time step t onwards
                discount *= self.gamma # Update the discount factor for the next time step
            G[t] = G_sum  # Store the calculated discounted sum in the G array

        mean = np.mean(G)
        std = np.std(G) + 1e-9
        self.G = (G - mean) / std # Normalize the discounted returns (G) to have zero mean and unit variance

        states = np.squeeze(np.vstack([self.state_memory]))
        actions = np.squeeze(np.vstack(self.action_memory))
        self.model.train_on_batch(states, actions, sample_weight=self.G) # Train the model on a batch of states and actions, with weights given by the normalized returns (G)

        # Clear the memory buffers
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
        while not done or score >= 500:
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

