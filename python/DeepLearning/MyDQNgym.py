from collections import deque
import numpy as np
import random

import gym
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras import backend as K

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95   # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)
    
    def _build_model(self):
        model = Sequential()
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss, optimizer=Adam(lr=self.learning_rate))
        ############################# TODO #####################################
        # Create a "simple" network, with 2-4 layers. Your network 
        # takes in the observations so the input_dim should match 
        # the size of observation space. Your network should output the probability 
        # of taking each action, so it's output size should match the 
        # action_size. Keep the last layer's activation as linear and use 
        # mean squared error for loss. Return your model after compiling it. 
        ########################### END TODO ###################################
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())    
    
    def remember(self, state, action, reward, next_state, done):
        ############################# TODO #####################################
        # Create a tuple of state, action, reward, next_state and done 
        # and append this tuple to the memory.
        ########################### END TODO ###################################
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # In this function we calculate and return the next action.
        # We are going to implement epsilon greedy logic here. 
        # With probability epsilon, return a random action and return that
        # With probability 1-epsilon return the action that the model predicts. 
        if np.random.rand() <= self.epsilon:
            # return TODO here goes the random action, delete the following pass as well
            return random.randrange(self.action_size)
        else:
            # return TODO here goes the predicted action, delete the following pass as well
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])

    def replay(self, batch_size):
        # We'll sample from our memories and get a handful of them and store them in minibatch 
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if not done:
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
                # Calculate the total discounted reward according to the Q-Learning formula
                # your formula should look something like this
                # target = current_reward + discounted maximum value obtained by next state
            else:
                target[0][action] = reward
            self.model.fit(state, target, epochs=1, verbose=0)
            
        # Decay the epsilon value 
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    
    # TODO What's the state size for CartPole game?
    state_size = env.observation_space.shape[0]
    # TODO What's the action size for CartPole game?
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size)
    done = False
    batch_size = 32 # Feel free to play with these 
    EPISODES = 100   # You shouldn't really need more than 100 episodes to get a score of 100

    
    for eps in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            
            # TODO Get an action from the agent
            action = agent.act(state)
            # TODO Send this action to the env and get the next_state, reward, done values
            next_state, reward, done, _ = env.step(action)
            
            # DO NOT CHANGE THE FOLLOWING 2 LINES 
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            
            # TODO Tell the agent to remember this memory
            # agent.remember(........)
            agent.remember(state, action, reward, next_state, done)
            
            # DO NOT CHANGE BELOW THIS LINE
            state = next_state
            if done:
                agent.update_target_model()
                print("episode: {}/{}, score: {}, eps: {:.2}".format(eps, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        if eps % 10 == 0:
            agent.save("./cartpole-dqn.h5")
    
    