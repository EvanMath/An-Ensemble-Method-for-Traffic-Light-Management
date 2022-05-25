from distutils.log import info
import os
from pyexpat import model
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
from sumo_rl import SumoEnvironment
import traci

from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.utils import obs_as_tensor, get_device
from stable_baselines3.common.distributions import CategoricalDistribution
from collections import Counter
import numpy as np
import torch

def predict_proba(model, state):
    obs = obs_as_tensor(state.reshape([1,-1]), model.policy.device)
    dis = model.policy.get_distribution(obs)
    probs = dis.distribution.probs
    probs_np = probs.cpu().detach().numpy()    
    return probs_np[0]

def majority_vote(a1, a2, a3):
    l = [a1, a2, a3]

    for _ in range(3):
        if not isinstance(l[_], int):
            l[_] = int(l[_])            

    occurence_count = Counter(l)
    all_equal = True
    for i in range(len(occurence_count.most_common())):
        if occurence_count.most_common()[i][1] > 1:
            all_equal = False
    if not all_equal:
        return max(set(l), key=l.count)
    else:
        return np.random.choice(l)

def majority_vote_with_probs(p1, p2, p3):
    '''
    Takes three different probability vectors in and outputs a randomly sampled 
    action from n_action according to majority voting scheme

    From: https://medium.com/@tu_53768/ensemble-reinforcement-learning-b06e28eec31c#:~:text=from%20these%20vectors.-,Majority%20vote,-First%2C%20the%20majority
    '''
    a = range(4)
    a1 = np.random.choice(a=a, p=p1)
    a2 = np.random.choice(a=a, p=p2)
    a3 = np.random.choice(a=a, p=p3)
    l = [a1, a2, a3]
    return max(set(l), key=l.count)        

def action_probability(model, obs):
    action_predicted = model.predict(obs)[0]
    exploration_eps = model.exploration_rate
    exploitation_eps = 1 - exploration_eps
    probs_np = [exploration_eps/4 for i in range(4)]
    probs_np[action_predicted] = exploitation_eps + exploration_eps/4
    return np.array(probs_np)



if __name__ == '__main__':

    env = SumoEnvironment(net_file='/home/talos/MSc_Thesis/nets/2way-single-intersection/single-intersection.net.xml',
                            route_file='/home/talos/MSc_Thesis/nets/2way-single-intersection/single-intersection-vhvh.rou.xml',
                            out_csv_name='/home/talos/MSc_Thesis/outputs/2way-single-intersection/ensTrain-MV-probs',
                            single_agent=True,
                            use_gui=False,
                            num_seconds=100000)                                        


    # Remeber to load models from May 12
    model_dqn = DQN.load('/home/talos/MSc_Thesis/models/DQN-05122022 163445/101000')
    model_ppo = PPO.load('/home/talos/MSc_Thesis/models/PPO-05122022 164805/101000')
    model_a2c = A2C.load('/home/talos/MSc_Thesis/models/A2C-05122022 165915/101000')

    EPISODES = 6

    for ep in range(EPISODES):
        obs = env.reset()
        done = False

        while not done:
            env.render()
            action = majority_vote(action_probability(model_dqn, obs), predict_proba(model_ppo, obs), predict_proba(model_a2c, obs))
            obs, reward, done, info = env.step(action)
    
    env.close()