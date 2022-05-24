import gym
from stable_baselines3.ppo.ppo import PPO
import os
import time
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
from sumo_rl import SumoEnvironment
import traci


if __name__ == '__main__':

    # Make directories for different models and logs
    # Logs are going to be used on tensorboard
    t = time.strftime("%m%d%Y %H%M%S", time.localtime())

    models_dir = f"/home/talos/MSc_Thesis/models/PPO-{t}"
    logdir = f"/home/talos/MSc_Thesis/logs/PPO-{t}"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir) 

    env = SumoEnvironment(net_file='/home/talos/MSc_Thesis/nets/2way-single-intersection/single-intersection.net.xml',
                            route_file='/home/talos/MSc_Thesis/nets/2way-single-intersection/single-intersection-vhvh.rou.xml',
                            out_csv_name='/home/talos/MSc_Thesis/outputs/2way-single-intersection/ppoPOSREW',
                            single_agent=True,
                            use_gui=False,
                            num_seconds=100000)

    model = PPO(env=env, policy="MlpPolicy", verbose=1, learning_rate=.001, tensorboard_log=logdir)

    TIMESTEPS = 10100
    for i in range(1, 11):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPOposrew")
        model.save(f"{models_dir}/{TIMESTEPS*i}")