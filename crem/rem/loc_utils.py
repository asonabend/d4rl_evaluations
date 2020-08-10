import numpy as np

# Code based on:
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

# Simple replay buffer
class ReplayBuffer(object):
	def __init__(self):
		self.storage = []

	# Expects tuples of (state, next_state, action, reward, done)
	def add(self, data):
		self.storage.append(data)

	def sample(self, batch_size):
		ind = np.random.randint(0, len(self.storage), size=batch_size)
		state, next_state, action, reward, done = [], [], [], [], []

		for i in ind:
			s, s2, a, r, d = self.storage[i]
			state.append(np.array(s, copy=False))
			next_state.append(np.array(s2, copy=False))
			action.append(np.array(a, copy=False))
			reward.append(np.array(r, copy=False))
			done.append(np.array(d, copy=False))

		return (np.array(state),
			np.array(next_state),
			np.array(action),
			np.array(reward).reshape(-1, 1),
			np.array(done).reshape(-1, 1))

	def save(self, filename):
		np.save("./buffers/"+filename+".npy", self.storage, allow_pickle=True)

	def load(self, filename):
		self.storage = np.load("./buffers/"+filename+".npy", allow_pickle=True)

###########################################################################
################  openAI wrapper for Riverswim environment ################
###########################################################################


import gym
from gym import spaces

class riverSwim(gym.Env):
    """Riverswim Environment that follows gym interface"""
    #metadata = {'render.modes': ['human']}
    def __init__(self,episode_length):
        super(riverSwim, self).__init__()
        self.time = 0
        self.state = 0
        self.episode_length = episode_length
        self.done = False
        self._max_episode_steps = episode_length -1
        # Define action and observation space
        N_DISCRETE_ACTIONS = 2
        N_DISCRETE_STATES = 6 
        N_TIME_STEPS = episode_length
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation_space = spaces.MultiDiscrete((N_DISCRETE_STATES, N_TIME_STEPS))
        self.observation_dim = 2
    # Execute one time step within the environment
    def step(self,action):
        if action not in [0,1]:
            return 'error'        
        if self.time == self.episode_length:
            self.done = True
        self.time += 1
        self.reward = 0 # no reward
        if self.state == 0:
            if action == 1: # swim to the right
                if np.random.binomial(1,.6)==1: # w.p. = 0.6 get to the right otherwise stay in state = 0
                    self.state = 1                
            else: #action == 0, stays in state = 0 and has the reward 5/1000
                self.reward = 5/1000
        elif self.state == 5:
            if action == 1: # swim to the right
                if np.random.binomial(1,.6)==1: # w.p. = 0.6 swim succesfully to the right 
                    self.reward = 1
                else: # w.p. 0.4 current takes it to the left
                    self.state = 4
            else: # action == 0
                self.state = 4
        else: #states 1,2,3,4
            if action == 1: # swim to the right
                dice = np.random.choice(3, 1, p=[.05,0.6,0.35])
                if dice==0: # w.p. = 0.05 current takes it to the left
                    self.state -= 1
                elif dice==2: # w.p. = 0.6 current it stays in the same state, w.p. 0.35 gets to the right
                    self.state += 1
            else: # action == 0
                self.state -= 1
        info = 'na'
        return np.array([self.state, self.time]),self.reward,self.done,info
    def reset(self):
    # Reset the state of the environment to an initial state
        self.time = 0
        self.state = 0
        self.done = False
        return np.array([self.state, self.time])
