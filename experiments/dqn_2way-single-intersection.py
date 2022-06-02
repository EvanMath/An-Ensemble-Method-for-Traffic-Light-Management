import gym
from stable_baselines3.dqn.dqn import DQN
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

    models_dir = f"/home/talos/MSc_Thesis/models/DQN-{t}"
    logdir = f"/home/talos/MSc_Thesis/logs/DQN-{t}"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir) 

    env = SumoEnvironment(net_file='/home/talos/MSc_Thesis/nets/2way-single-intersection/single-intersection.net.xml',
                            route_file='/home/talos/MSc_Thesis/nets/2way-single-intersection/single-intersection-vhvh.rou.xml',
                            out_csv_name='/home/talos/MSc_Thesis/outputs/2way-single-intersection/DQN2M',
                            single_agent=True,
                            use_gui=False,
                            num_seconds=100000,
                            sumo_seed=42    # default 'random'
                            )

    model = DQN(
        env=env,
        policy="MlpPolicy",
        learning_rate=0.001,
        buffer_size=50000,  # changed
        learning_starts=1000,   # changed
        # train_freq=1,
        target_update_interval=500,
        gamma=0.99,
        batch_size=32,
        exploration_initial_eps=0.95,    # changed
        exploration_final_eps=0.005,
        verbose=1,
        tensorboard_log=logdir
    )
    TIMESTEPS = 2000000
    # for i in range(1, 11):
    #     model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DQNposrew")
    #     model.save(f"{models_dir}/{TIMESTEPS*i}")

    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DQN2M")
    model.save(f"{models_dir}/DQN2M")