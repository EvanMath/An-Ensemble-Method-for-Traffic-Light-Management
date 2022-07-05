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

def majority_vote(p1, p2, p3=[]):
    """
    Receives as input three actions and outputs the action that receives the most votes.
    In case all action receive 1 vote it outputs a random action.
    """
    l = [np.argmax(p) for p in [p1, p2, p3] if len(p) > 1]

    if len(set(l)) < 3:
        return max(set(l), key=l.count)
    else:
        return rank_voting(p1, p2, p3)

def majority_vote_1(p1, p2, p3):
    """
    Receives as input three actions and outputs the action that receives the most votes.
    In case all action receive 1 vote it outputs a random action.
    """
    l = [np.argmax(p) for p in [p1, p2, p3] if len(p) > 1]

    if len(set(l)) < 3:
        return max(set(l), key=l.count)
    else:
        return np.random.choice(a=l, p=[.1, .75, .15])        

def majority_vote_with_probs(p1, p2, p3=[]):
    '''
    Takes three different probability vectors in and outputs a randomly sampled 
    action from n_action according to majority voting scheme

    From: https://medium.com/@tu_53768/ensemble-reinforcement-learning-b06e28eec31c#:~:text=from%20these%20vectors.-,Majority%20vote,-First%2C%20the%20majority
    '''
    a = range(4)
    a1 = np.random.choice(a=a, p=p1)
    a2 = np.random.choice(a=a, p=p2)

    if len(p3) > 1:
        a3 = np.random.choice(a=a, p=p3)
        l = [a1, a2, a3]
    else:
        l = [a1, a2]

    return max(set(l), key=l.count)

def soft_voting(p1, p2, p3=[], probs=False):

    if len(p3) > 1:
        soft_vote = np.sum(np.stack((p1, p2, p3)), axis=0)
    else:
        soft_vote = np.sum(np.stack((p1, p2)), axis=0)
    if not probs:
        return np.argmax(soft_vote)
    else:
        return soft_vote

def rank_voting_1(p1, p2, p3=np.zeros(4), probs=False):
    p = np.stack((p1, p2, p3), axis=0)

    max_probs = np.max(p, axis=0)
    best_action = np.argmax(max_probs)

    if not probs:
        return best_action
    else:
        return max_probs

def rank_voting(p1, p2, p3=[], probs=False):

    if len(p3) > 1:
        for p in [p1, p2, p3]:
            p_copy = np.sort(p)
            for i in range(len(p)):
                p[np.where(p==p_copy[i])] = (i+1)*p_copy[i]
        soft_vote = np.sum(np.stack((p1, p2, p3)), axis=0)
    else:
        soft_vote = np.sum(np.stack((p1, p2)), axis=0)
    if not probs:
        return np.argmax(soft_vote)
    else:
        return soft_vote        

def average_voting(p1, p2, p3=[], T=0.5):
    '''
    Takes in different probability vectors and outputs a randomly sampled 
    action from n_action with probability equals the average probability of the 
    normalized exponentiated input vectors, with a temperature T controlling
    the degree of spread for the out vector (Boltzmann Probabilities)
    '''
    a = range(4)
    if len(p3) > 1:
        boltz_ps = [np.exp(prob/T)/sum(np.exp(prob/T)) for prob in [p1, p2, p3]]
        p = (boltz_ps[0] + boltz_ps[1] + boltz_ps[2])/3
        p /= np.sum(p)  # To avoid (ValueError: np-random-choice-probabilities-do-not-sum-to-1)[https://stackoverflow.com/questions/46539431/np-random-choice-probabilities-do-not-sum-to-1]
        a = np.random.choice(a=a, p=p)
    else:
        boltz_ps = [np.exp(prob/T)/sum(np.exp(prob/T)) for prob in [p1, p2]]
        p = (boltz_ps[0] + boltz_ps[1])/2
        p /= np.sum(p)  # To avoid (ValueError: np-random-choice-probabilities-do-not-sum-to-1)[https://stackoverflow.com/questions/46539431/np-random-choice-probabilities-do-not-sum-to-1]
        a = np.random.choice(a=a, p=p)
    return a



if __name__ == '__main__':

    env = SumoEnvironment(net_file='/home/talos/MSc_Thesis/nets/2way-single-intersection/single-intersection.net.xml',
                            route_file='/home/talos/MSc_Thesis/nets/2way-single-intersection/single-intersection-vhvh.rou.xml',
                            # out_csv_name='/home/talos/MSc_Thesis/outputs/2way-single-intersection/A2C-trained-5',
                            single_agent=True,
                            use_gui=True,
                            num_seconds=100000,
                            sumo_seed=42)                                        



    model_dqn = DQN.load('/home/talos/MSc_Thesis/models/DQN-06012022 171843/DQN2M')   # Best performance on predictions DQN-trained-4
    model_ppo = PPO.load('/home/talos/MSc_Thesis/models/PPO-05122022 164805/101000')    # Best performance on predictions PPO-trained-3
    model_a2c = A2C.load('/home/talos/MSc_Thesis/models/A2C-05312022 165157/A2C1.5M')  # Best performance on predictions A2C-trained-0


    # model = A2C.load('/home/talos/MSc_Thesis/models/A2C-06062022 082522/A2C100')
    EPISODES = 1

    for ep in range(EPISODES):
        obs = env.reset()
        done = False

        while not done:
            env.render()
            action = rank_voting(action_probability(model_dqn, obs), predict_proba(model_a2c, obs))
            # action, _ = model_ppo.predict(obs)
            obs, reward, done, info = env.step(action)
    
    env.close()