from tabnanny import verbose
import numpy as np
import gym
import argparse
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

PATH_ENV = "./build/InitialBuild"
LOG_PATH = "./Logging/"

class HighLevelEnv(gym.Env):
    def __init__(self, unity_env, low_level_models):
        super(HighLevelEnv, self).__init__()
        self.unity_env = unity_env
        self.num_pieces = len(low_level_models)
        self.low_level_models = low_level_models
        self.action_space = gym.spaces.Discrete(self.num_pieces)
        self.observation_space = self.unity_env.observation_space

    def step(self, action):
        piece_action = action
        low_level_model = self.low_level_models[piece_action]

        total_reward = 0
        done = False
        low_level_obs = self.unity_env.reset()
        step_count = 0

        while not done and step_count < 100:  # Limiting steps to avoid infinite loops
            move_action, _ = low_level_model.predict(low_level_obs, deterministic=True)
            low_level_obs, reward, done, info = self.unity_env.step([piece_action, move_action])
            total_reward += reward
            step_count += 1

        reward = total_reward
        obs = low_level_obs
        return obs, reward, done, info

    def reset(self):
        return self.unity_env.reset()

    def render(self, mode='human'):
        return self.unity_env.render(mode)

    def seed(self, seed=None):
        self.unity_env.seed(seed)
        np.random.seed(seed)


class LowLevelEnv(gym.Env):
    def __init__(self, unity_env, current_piece):
        super(LowLevelEnv, self).__init__()
        self.unity_env = unity_env
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = self.unity_env.observation_space
        self.current_piece = current_piece


    def step(self, action):
        move_action = action
        return self.unity_env.step([self.current_piece, move_action])

    def reset(self):
        return self.unity_env.reset()

    def render(self, mode='human'):
        return self.unity_env.render(mode)

    def seed(self, seed=None):
        self.unity_env.seed(seed)
        np.random.seed(seed)


def create_unity_env(time_scale=25.0):
    channel_engine = EngineConfigurationChannel()
    channel_env = EnvironmentParametersChannel()
    unity_env = UnityEnvironment(PATH_ENV, no_graphics=False, side_channels=[channel_engine, channel_env])
    env = UnityToGymWrapper(unity_env, uint8_visual=False, allow_multiple_obs=False, flatten_branched=False)
    channel_engine.set_configuration_parameters(time_scale=time_scale)
    return env


def train_hierarchical_model(env, num_pieces, high_level_timesteps, low_level_timesteps, seed, learning_rate, net_arch, activation_fn):
    low_level_models = []
    for piece_id in range(num_pieces):
        low_level_env = DummyVecEnv([lambda: LowLevelEnv(env, piece_id)])
        low_level_env.seed(seed)
        low_level_model = PPO("MlpPolicy", low_level_env, verbose=1, seed=seed, learning_rate=learning_rate, policy_kwargs=dict(activation_fn=eval("th.nn." + activation_fn), net_arch=net_arch))
        low_level_model.learn(total_timesteps=low_level_timesteps)
        low_level_models.append(low_level_model)

    high_level_env = DummyVecEnv([lambda: HighLevelEnv(env, low_level_models)])
    high_level_env.seed(seed)

    high_level_model = PPO("MlpPolicy", high_level_env, verbose=1, seed=seed, learning_rate=learning_rate, policy_kwargs=dict(activation_fn=eval("th.nn." + activation_fn), net_arch=net_arch))

    logger = configure(LOG_PATH + "Hierarchical_" + str(seed), ["stdout", "csv", "log", "tensorboard", "json"])
    high_level_model.set_logger(logger)

    high_level_model.learn(total_timesteps=high_level_timesteps)

    return HierarchicalPPO(high_level_model, low_level_models)


class HierarchicalPPO:
    def __init__(self, high_level_model, low_level_models):
        self.high_level_model = high_level_model
        self.low_level_models = low_level_models

    def predict(self, obs, deterministic=True):
        piece_action, _ = self.high_level_model.predict(obs, deterministic=deterministic)
        piece_action = int(piece_action)  # Ensure piece_action is an integer
        low_level_model = self.low_level_models[piece_action]
        low_level_obs = obs  # Assuming the low-level observation is the same
        move_action, _ = low_level_model.predict(low_level_obs, deterministic=deterministic)
        move_action = int(move_action)  # Ensure move_action is an integer
        return [piece_action, move_action], None


def forwards(algo, num_timesteps, seed, learning_rate=0.0003, net_arch=[64, 64], activation_fn="Tanh"):
    env = create_unity_env()
    num_pieces = 4
    high_level_timesteps = num_timesteps
    low_level_timesteps = num_timesteps // 10

    hierarchical_model = train_hierarchical_model(env, num_pieces, high_level_timesteps, low_level_timesteps, seed, learning_rate, net_arch, activation_fn)
    env.close()

    return hierarchical_model


def playback_hierarchical_model(hierarchical_model, num_episodes):
    env = create_unity_env()
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            action, _ = hierarchical_model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1

        print(f"Episode {episode + 1}: Total Reward = {total_reward}, Steps = {step_count}")

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('algo', help='Algorithm to be used for execution: PPO or TRPO currently')
    parser.add_argument('num_timesteps', type=int, help='Number of time steps for execution')
    parser.add_argument('seed', type=int, help='Seed for the RL Algo. Example: 100, 200, 300, 400, 500')
    parser.add_argument('--playback', action='store_true', help='Playback mode to run a pretrained model')

    args = parser.parse_args()

    if args.playback:
        hierarchical_model = forwards(eval(args.algo), args.num_timesteps, args.seed)
        playback_hierarchical_model(hierarchical_model, num_episodes=10)
    else:
        forwards(eval(args.algo), args.num_timesteps, args.seed)