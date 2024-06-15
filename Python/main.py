from tabnanny import verbose
import numpy as np
import gym
import argparse
import torch as th

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

from stable_baselines3 import PPO
from stable_baselines3 import A2C

from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor


from sb3_contrib import RecurrentPPO


PATH_ENV_ASS1 = "./build/Build1"
PATH_ENV_ASS2 = "./build/Build2"
LOG_PATH = "./Logging/"



def create_unity_env(time_scale = 25.0, env_num = 1, worker_id =1):
    """
    Function to create a simulation environment.
    """
    channel_engine = EngineConfigurationChannel()
    channel_env = EnvironmentParametersChannel()

    if env_num == 1:
        path_env = PATH_ENV_ASS1
    if env_num == 2:
        path_env = PATH_ENV_ASS2

    unity_env = UnityEnvironment(path_env,  no_graphics=True, \
        side_channels=[channel_engine, channel_env], worker_id=worker_id)
    env = UnityToGymWrapper(unity_env, uint8_visual=True, allow_multiple_obs=False, flatten_branched=False )

    channel_engine.set_configuration_parameters(time_scale = time_scale)
    return env


def forwards(algo, num_timesteps, seed, learning_rate = 0.0003, net_arch =[64,64] , activation_fn = "Tanh", env_num = 1):
    """
    Run the experiments. 
    """
    RUN_NAME= algo.__name__ +"_DATA_"+str(learning_rate)+"_" + str(net_arch)+ "_" + activation_fn + "_"
    print(LOG_PATH + RUN_NAME + str(seed))
    logger = configure(LOG_PATH + RUN_NAME + str(seed), ["stdout", "csv", "log", "tensorboard", "json"]) # Logger needed to get JSONs
    env =  create_unity_env(time_scale = 10.0, env_num = env_num)



    policy_kwargs = dict(activation_fn=eval("th.nn."+activation_fn),
                     net_arch=net_arch)
    
    model =  PPO("MlpPolicy", env = env, verbose=1, seed=seed, learning_rate=learning_rate, policy_kwargs=policy_kwargs)
    model.set_logger(logger)

    # Create the evaluation environment and wrap it with the Monitor wrapper
    eval_env = create_unity_env(time_scale=10.0, env_num=env_num, worker_id= 2)

    eval_callback = EvalCallback(eval_env, best_model_save_path=LOG_PATH + RUN_NAME + str(seed),
                                log_path=LOG_PATH + RUN_NAME + str(seed), eval_freq=1000,
                                deterministic=True, render=False)
    
    print(model.policy)
    
    model.learn(total_timesteps=num_timesteps, callback=[eval_callback])
    env.reset()
    eval_env.reset()
    rew, std = evaluate_policy(model, env, return_episode_rewards=True)
    results = {'reward':rew, 'std': std}
    logger.log(results)
    #print("Ep Reward " + str(float(rew)) + " std div " + str(float(std)))
    np.savez(LOG_PATH + RUN_NAME + str(seed)+ "/eval_values.npz", rew=rew, std=std)
    model.save("./"+ LOG_PATH + RUN_NAME + str(seed) + "/last_model") 
    eval_env.close()
    env.close()
    
def playback_model(model_path, num_episodes, env_num = 1):
    """
    Function to run the pretrained model in the environment for a number of episodes.
    """
    env = create_unity_env(time_scale = 1.0, env_num = env_num)
    model = PPO.load(model_path)

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        episode_length = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            print(f"Actions:{ action } ")
            obs, reward, done, info = env.step(action)
            total_reward += reward
            episode_length += 1

        print(f"Episode {episode + 1}: Total Reward = {total_reward}; Episode Length = { episode_length } ")


if __name__ == '__main__':
    # Create the top-level parser
    parser = argparse.ArgumentParser(description="Train or playback a RL model")

    subparsers = parser.add_subparsers(dest="action", help="Action to perform")

    # Create the parser for the "train" action
    parser_train = subparsers.add_parser("train", help="Train a new model")
    parser_train.add_argument('--algo', required=True, help='Algorithm to be used for training: PPO or TRPO')
    parser_train.add_argument('--num_timesteps', type=int, required=True, help='Number of time steps for training')
    parser_train.add_argument('--seed', type=int, required=True, help='Seed for the RL algorithm. Example: 100, 200, 300, 400, 500')
    parser_train.add_argument('--env', type=int, required=True, help='Number of the env. 1 = small, 2 = big')

    # Create the parser for the "playback" action
    parser_playback = subparsers.add_parser("playback", help="Playback a pretrained model")
    parser_playback.add_argument('--model_path', required=True, help='Path to the pretrained model for playback')
    parser_playback.add_argument('--num_episodes', type=int, required=True, help='Number of episodes to run for playback')
    parser_playback.add_argument('--env', type=int, required=True, help='Number of the env. 1 = small, 2 = big')

    # Parse the arguments
    args = parser.parse_args()

    if args.action == 'train':
        forwards(eval(args.algo), args.num_timesteps, args.seed, args.env)
    elif args.action == 'playback':
        playback_model(args.model_path, args.num_episodes, args.env)
