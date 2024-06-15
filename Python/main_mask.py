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
from stable_baselines3.common.vec_env import patch_gym

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
# This is a drop-in replacement for EvalCallback
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

PATH_ENV = "./build/InitialBuild"
LOG_PATH = "./Logging/"


class ActionMaskingEnv(gym.Wrapper):
    def __init__(self, env, num_pieces):
        super(ActionMaskingEnv, self).__init__(env)
        self.num_pieces = num_pieces
        self.pieces_moved = set()
        self.current_piece = None

    def step(self, action):
        piece_action, move_action = action
        if self.current_piece is None or piece_action != self.current_piece:
            self.current_piece = piece_action
            self.pieces_moved.add(piece_action)
        obs, reward, done, info = self.env.step([piece_action, move_action])
        return obs, reward, done, info

    def reset(self):
        self.pieces_moved.clear()
        self.current_piece = None
        return self.env.reset()

    def get_action_mask(self):
        # Create a mask for all actions
        action_mask = np.ones((self.num_pieces, 3), dtype=bool)
        for piece in self.pieces_moved:
            # Mask all actions related to moved pieces
            action_mask[piece, :] = False
        return action_mask


def mask_fn(env: gym.Env) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.get_action_mask()

def create_unity_env(time_scale = 25.0):
    """
    Function to create a simulation environment.
    """
    channel_engine = EngineConfigurationChannel()
    channel_env = EnvironmentParametersChannel()

    path_env = PATH_ENV

    unity_env = UnityEnvironment(path_env,  no_graphics=False, \
        side_channels=[channel_engine, channel_env])
    env = UnityToGymWrapper(unity_env, uint8_visual=False, allow_multiple_obs=False, flatten_branched=False )

    channel_engine.set_configuration_parameters(time_scale = time_scale)
    return env


def forwards(algo, num_timesteps, seed, learning_rate = 0.0003, net_arch =[64,64] , activation_fn = "Tanh"):
    """
    Run the experiments. 
    """
    RUN_NAME= algo.__name__ +"_Short_easy_"+str(learning_rate)+"_" + str(net_arch)+ "_" + activation_fn + "_"
    print(LOG_PATH + RUN_NAME + str(seed))
    logger = configure(LOG_PATH + RUN_NAME + str(seed), ["stdout", "csv", "log", "tensorboard", "json"]) # Logger needed to get JSONs
    env =  create_unity_env(time_scale = 25.0)

    
    policy_kwargs = dict(activation_fn=eval("th.nn."+activation_fn),
                     net_arch=net_arch)
    

    env = ActionMaskingEnv(env, 4)
    env = ActionMasker(env, mask_fn)

    #env = patch_gym._patch_env(env)

    model = PPO("MlpPolicy", env = env, verbose=1, seed=seed, learning_rate=learning_rate, policy_kwargs=policy_kwargs)
    model.set_logger(logger)
    eval_callback = EvalCallback(env, best_model_save_path=LOG_PATH + RUN_NAME + str(seed),
                                log_path=LOG_PATH + RUN_NAME + str(seed), eval_freq=1000,
                                deterministic=True, render=False)
    
    print(model.policy)
    
    model.learn(total_timesteps=num_timesteps, callback=[eval_callback])
    env.reset()
    rew, std = evaluate_policy(model, env, return_episode_rewards=True)
    results = {'reward':rew, 'std': std}
    logger.log(results)
    #print("Ep Reward " + str(float(rew)) + " std div " + str(float(std)))
    np.savez(LOG_PATH + RUN_NAME + str(seed)+ "/eval_values.npz", rew=rew, std=std)
    model.save("./"+ LOG_PATH + RUN_NAME + str(seed) + "/last_model") 
    env.close()
    
def playback_model(model_path, num_episodes):
    """
    Function to run the pretrained model in the environment for a number of episodes.
    """
    env = create_unity_env(time_scale = 1.0)
    model = PPO.load(model_path)

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        episode_length = 0

        while not done:
            action_masks = get_action_masks(env)
            action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
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

    # Create the parser for the "playback" action
    parser_playback = subparsers.add_parser("playback", help="Playback a pretrained model")
    parser_playback.add_argument('--model_path', required=True, help='Path to the pretrained model for playback')
    parser_playback.add_argument('--num_episodes', type=int, required=True, help='Number of episodes to run for playback')

    # Parse the arguments
    args = parser.parse_args()

    if args.action == 'train':
        forwards(eval(args.algo), args.num_timesteps, args.seed)
    elif args.action == 'playback':
        playback_model(args.model_path, args.num_episodes)
