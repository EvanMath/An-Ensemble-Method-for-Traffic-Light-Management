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
    """
    Takes as input a model and an observation. It returns the probability distribution for each action.
    Works only on PPO and A2C algorithms.
    """
    obs = obs_as_tensor(state.reshape([1,-1]), model.policy.device)
    dis = model.policy.get_distribution(obs)
    probs = dis.distribution.probs
    probs_np = probs.cpu().detach().numpy()    
    return probs_np[0]

def majority_vote(p1, p2, p3):
    """
    Receives as input three actions and outputs the action that receives the most votes.
    In case all action receive 1 vote it outputs a random action.
    """
    l = [np.argmax(p) for p in [p1, p2, p3]]

    # for _ in range(3):
    #     if not isinstance(l[_], int):
    #         l[_] = int(l[_])            

    occurence_count = Counter(l)
    all_equal = False

    if len(set(l)) < 3:
        return max(set(l), key=l.count)
    else:
        return rank_voting(p1, p2, p3)

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

def action_probability(model, state):
    """
    Takes in a model and an observation. It outputs the probability for each action.
    It works for DQN.
    """
    action_predicted = model.predict(state)[0]
    exploration_eps = model.exploration_rate
    exploitation_eps = 1 - exploration_eps
    probs_np = [exploration_eps/4 for i in range(4)]
    probs_np[action_predicted] = exploitation_eps + exploration_eps/4
    return np.array(probs_np)

def rank_voting(p1, p2, p3):
    soft_vote = np.sum(np.stack((p1, p2, p3)), axis=0)
    return np.argmax(soft_vote)

# def action_probability_2(model, obs):
#     obs_tensor, _ = model.q_net.obs_to_tensor(obs)
#     q_values = model.q_net(obs_tensor)
#     return np.array(q_values.cpu().detach().numpy())    

def boltzmann_prob(p1, p2, p3, T=0.5):
    '''
    Takes three different probability vectors in and outputs a randomly sampled 
    action from n_action with probability equals the average probability of the 
    normalized exponentiated input vectors, with a temperature T controlling
    the degree of spread for the out vector
    '''
    a = range(4)
    boltz_ps = [np.exp(prob/T)/sum(np.exp(prob/T)) for prob in [p1, p2, p3]]
    p = (boltz_ps[0] + boltz_ps[1] + boltz_ps[2])/3
    p /= np.sum(p)  # To avoid (ValueError: np-random-choice-probabilities-do-not-sum-to-1)[https://stackoverflow.com/questions/46539431/np-random-choice-probabilities-do-not-sum-to-1]
    a = np.random.choice(a=a, p=p)
    return a



if __name__ == '__main__':

    env = SumoEnvironment(net_file='/home/talos/MSc_Thesis/nets/2way-single-intersection/single-intersection.net.xml',
                            route_file='/home/talos/MSc_Thesis/nets/2way-single-intersection/single-intersection-vhvh.rou.xml',
                            out_csv_name='/home/talos/MSc_Thesis/outputs/2way-single-intersection/ensTrain-MVupd',
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