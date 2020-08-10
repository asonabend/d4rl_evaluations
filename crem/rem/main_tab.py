import gym
import numpy as np
import torch
import argparse
import os
import d4rl
import pickle
from EGOPSRL_functions import *        

from loc_utils import *
#import DDPG
#import BCQ
#import TD3
import REM
#import RSEM
#import DDPG_REM


# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, eval_episodes=10000):
	avg_reward = 0.
	for _ in range(eval_episodes):
		obs = env.reset()
		done = False
		while not done:
			action = np.round(policy.select_action(np.array(obs))[0])
			if action == -1:
				action += 1
			#print(action)
			obs, reward, done, _ = env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print ("---------------------------------------")
	print ("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
	print ("---------------------------------------")
	return avg_reward


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--env_name", default='Riverswim')				# OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)					# Sets Gym, PyTorch and Numpy seeds
	#parser.add_argument("--buffer_type", default="Robust")				# Prepends name to filename.
	parser.add_argument("--eval_freq", default=5e3, type=float)			# How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=float)		# Max time steps to run environment for
	parser.add_argument("--agent_name", default="REM")
	parser.add_argument("--lr", default=1e-3, type=float)
	parser.add_argument("--num_heads", default=100, type=int)
	parser.add_argument("--prefix", default="default")
	#args = parser.parse_args()
	args = parser.parse_args([])

	file_name = "%s_%s_%s_%s" % (args.agent_name, args.env_name, str(args.seed), str(args.lr))
	if args.agent_name == 'REM':
	  	file_name += '_%s' % (args.num_heads)
	if args.prefix != "default":
		file_name += '_%s' % (args.prefix)
	#buffer_name = "%s_%s_%s" % (args.buffer_type, args.env_name, str(args.seed))
	print ("---------------------------------------")
	print ("Settings: " + file_name)
	print ("---------------------------------------")

	results_dir = "./results_%s" % (args.agent_name)
	if not os.path.exists(results_dir):
	  os.makedirs(results_dir)
	if args.env_name == 'Riverswim':
		env = riverSwim(20)
		action_dim = 1
		max_action = float(1)
	else:
		env = gym.make(args.env_name)
		action_dim = env.action_space.shape[0]
		max_action = float(env.action_space.high[0])
	state_dim = env.observation_space.shape[0]

	env.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)


	# Initialize policy
	kwargs = {'lr': args.lr}
	if args.agent_name in ['REM', 'RSEM', 'DDPG_REM']:
	  kwargs.update(num_heads=args.num_heads)
	if args.agent_name == 'BCQ':
	  policy_agent = BCQ.BCQ
	elif args.agent_name == 'TD3':
	  policy_agent = TD3.TD3
	elif args.agent_name == 'REM':
	  policy_agent = REM.REM
	elif args.agent_name == 'RSEM':
	  policy_agent = RSEM.RSEM
	elif args.agent_name == 'DDPG_REM':
	  policy_agent = DDPG_REM.DDPG_REM
	elif args.agent_name == 'DDPG':
	 policy_agent = DDPG.DDPG
	 kwargs.pop('lr')
	policy = policy_agent(state_dim, action_dim, max_action, **kwargs)

	# Load buffer
	replay_buffer = ReplayBuffer()
	#replay_buffer.load(buffer_name)
    # Fill up the replay buffer from the observed datasets
	if args.env_name in 'Riverswim':
		#os.chdir('/Users/aaron/Dropbox (HMS)/Documents/Research/RL/BNN-RL/ESPSRL')
		data_dict = pickle.load( open("/Users/aaron/Dropbox (HMS)/Documents/Research/RL/BNN-RL/Data/Riverswim/observed_datasets/riverswim_PSRL_rndm0.37_obs_data10000.p", "rb" ) )
		#os.chdir('/Users/aaron/Dropbox (HMS)/Documents/Research/RL/BNN-RL/Code/d4rl_evaluations/crem/rem')
		# Transform observed data frame into replay buffer:
		for t in range(data_dict['H_tk'].shape[0]):
			state, action, reward, next_state =  data_dict['H_tk'][t,:]
			if next_state < 0:
				next_state = 10
			done = (t % 19 == 0)
			done_float = float(done)
			episode_start = (t % 20 == 0)
			# Expects tuples of (state, next_state, action, reward, done)
			replay_buffer.add((np.array([state,t % 20]),  np.array([next_state,t +1 % 20]),action, reward,  done))  
			#np.mean(replay_buffer.sample(100000)[3])*20
	else:
		dataset = env.get_dataset()
		N = dataset['rewards'].shape[0]
		for i in range(N-1):
			obs = dataset['observations'][i]
			new_obs = dataset['observations'][i+1]
			action = dataset['actions'][i]
			reward = dataset['rewards'][i]
			done_bool = bool(dataset['terminals'][i])
			replay_buffer.add((obs, new_obs, action, reward, done_bool))

	evaluations = []

	episode_num = 0
	done = True

	training_iters = 0
	while training_iters < args.max_timesteps:
		pol_vals = policy.train(replay_buffer, iterations=int(args.eval_freq))

		evaluations.append(evaluate_policy(policy))
		#np.save(results_dir + "/" + file_name, evaluations)
		print(evaluations)
		training_iters += args.eval_freq
		print ("Training iterations: " + str(training_iters))
