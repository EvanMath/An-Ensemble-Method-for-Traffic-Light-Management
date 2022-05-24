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
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy

def majority_vote(a1, a2, a3):
    '''
    Takes three different probability vectors in and outputs a randomly sampled 
    action from n_action according to majority voting scheme
    '''
    l = [a1, a2, a3]
    return max(l, key=l.count)


if __name__ == '__main__':

    env = SumoEnvironment(net_file='/home/talos/MSc_Thesis/sumo-rl/nets/2way-single-intersection/single-intersection.net.xml',
                            route_file='/home/talos/MSc_Thesis/sumo-rl/nets/2way-single-intersection/single-intersection-vhvh.rou.xml',
                            out_csv_name='/home/talos/MSc_Thesis/sumo-rl/outputs/2way-single-intersection/ensemble-trainer-test',
                            single_agent=True,
                            use_gui=False,
                            num_seconds=100000)                                        


    # Remeber to load models from May 12
    model_dqn = DQN.load('/home/talos/MSc_Thesis/models/DQN-05122022 163445/101000')
    model_ppo = PPO.load('/home/talos/MSc_Thesis/models/PPO-05122022 164805/101000')
    model_a2c = A2C.load('/home/talos/MSc_Thesis/models/A2C-05122022 165915/101000')
    
    # mean_reward, std_reward = evaluate_policy(model=model_ppo, env=Monitor(env), n_eval_episodes=5, render=False)
    # print(f"Mean Episode Reward: {mean_reward}\nStandard Deviation Episode Reward: {std_reward}")

    EPISODES = 6

    for ep in range(EPISODES):
        obs = env.reset()
        done = False

        while not done:
            env.render()
            action = majority_vote(model_ppo.predict(obs)[0], model_dqn.predict(obs)[0], model_a2c.predict(obs)[0])
            obs, reward, done, info = env.step(action)
    
    env.close()