import os
import gym
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from haco.DIDrive_core.haco_env import HACOEnv
from haco.utils.train_utils import get_train_parser
from ray.rllib.agents.callbacks import DefaultCallbacks
import datetime

# Set up environment for avoiding certain threading issues
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Define the CARLA callback from HACO
class CARLACallBack(DefaultCallbacks):
    def on_episode_start(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        episode.user_data["velocity"] = []
        episode.user_data["steering"] = []
        episode.user_data["step_reward"] = []
        episode.user_data["acceleration"] = []
        episode.user_data["takeover"] = 0
        episode.user_data["raw_episode_reward"] = 0
        episode.user_data["episode_crash_rate"] = 0
        episode.user_data["episode_out_of_road_rate"] = 0
        episode.user_data["total_takeover_cost"] = 0
        episode.user_data["total_native_cost"] = 0
        episode.user_data["cost"] = 0

    def on_episode_step(self, *, worker, base_env, episode, env_index, **kwargs):
        info = episode.last_info_for()
        if info is not None:
            episode.user_data["velocity"].append(info["velocity"])
            episode.user_data["steering"].append(info["steering"])
            episode.user_data["acceleration"].append(info["acceleration"])
            episode.user_data["step_reward"].append(info["step_reward"])
            episode.user_data["takeover"] += 1 if info["takeover"] else 0
            episode.user_data["raw_episode_reward"] += info["step_reward"]
            episode.user_data["episode_crash_rate"] += 1 if info["crash"] else 0
            episode.user_data["episode_out_of_road_rate"] += 1 if info["out_of_road"] else 0
            episode.user_data["total_takeover_cost"] += info["takeover_cost"]
            episode.user_data["total_native_cost"] += info["native_cost"]
            episode.user_data["cost"] += info["cost"] if "cost" in info else info["native_cost"]

    def on_episode_end(self, worker, base_env, policies, episode, **kwargs):
        arrive_dest = episode.last_info_for()["arrive_dest"]
        crash = episode.last_info_for()["crash"]
        out_of_road = episode.last_info_for()["out_of_road"]
        max_step_rate = not (arrive_dest or crash or out_of_road)
        episode.custom_metrics["success_rate"] = float(arrive_dest)
        episode.custom_metrics["crash_rate"] = float(crash)
        episode.custom_metrics["out_of_road_rate"] = float(out_of_road)
        episode.custom_metrics["max_step_rate"] = float(max_step_rate)
        episode.custom_metrics["velocity_max"] = float(np.max(episode.user_data["velocity"]))
        episode.custom_metrics["velocity_mean"] = float(np.mean(episode.user_data["velocity"]))
        episode.custom_metrics["velocity_min"] = float(np.min(episode.user_data["velocity"]))
        episode.custom_metrics["steering_max"] = float(np.max(episode.user_data["steering"]))
        episode.custom_metrics["steering_mean"] = float(np.mean(episode.user_data["steering"]))
        episode.custom_metrics["steering_min"] = float(np.min(episode.user_data["steering"]))
        episode.custom_metrics["acceleration_min"] = float(np.min(episode.user_data["acceleration"]))
        episode.custom_metrics["acceleration_mean"] = float(np.mean(episode.user_data["acceleration"]))
        episode.custom_metrics["acceleration_max"] = float(np.max(episode.user_data["acceleration"]))
        episode.custom_metrics["step_reward_max"] = float(np.max(episode.user_data["step_reward"]))
        episode.custom_metrics["step_reward_mean"] = float(np.mean(episode.user_data["step_reward"]))
        episode.custom_metrics["step_reward_min"] = float(np.min(episode.user_data["step_reward"]))
        episode.custom_metrics["takeover_rate"] = float(episode.user_data["takeover"] / episode.length)
        episode.custom_metrics["takeover_count"] = float(episode.user_data["takeover"])
        episode.custom_metrics["raw_episode_reward"] = float(episode.user_data["raw_episode_reward"])
        episode.custom_metrics["episode_crash_num"] = float(episode.user_data["episode_crash_rate"])
        episode.custom_metrics["episode_out_of_road_num"] = float(episode.user_data["episode_out_of_road_rate"])

        episode.custom_metrics["total_takeover_cost"] = float(episode.user_data["total_takeover_cost"])
        episode.custom_metrics["total_native_cost"] = float(episode.user_data["total_native_cost"])

        episode.custom_metrics["cost"] = float(episode.user_data["cost"])

# Define custom HACO environment for SB3 compatibility
class SB3HACOEnv(gym.Env):
    def __init__(self, env_config):
        super(SB3HACOEnv, self).__init__()
        self.haco_env = HACOEnv(env_config)
        
        # Define observation and action space
        self.observation_space = self.haco_env.observation_space
        self.action_space = self.haco_env.action_space
    
    def reset(self):
        return self.haco_env.reset()
    
    def step(self, action):
        obs, reward, done, info = self.haco_env.step(action)
        return obs, reward, done, info
    
    def render(self, mode="human"):
        self.haco_env.render(mode)
    
    def close(self):
        self.haco_env.close()

# Define the SB3-compatible callback for CARLA metrics
class CARLACallBackSB3(BaseCallback):
    def __init__(self, verbose=0):
        super(CARLACallBackSB3, self).__init__(verbose)
        self.carla_callback = CARLACallBack()

    def _on_step(self) -> bool:
        if self.locals['dones']:
            episode_info = self.locals['infos']
            self.carla_callback.on_episode_end(
                worker=None,
                base_env=self.training_env,
                policies=None,
                episode=None,
                **episode_info
            )
        return True  # Continue training

# Function to get current timestamp
def get_time_str():
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")

if __name__ == '__main__':
    # Set up the environment configuration for HACO
    env_config = {
        "keyboard_control": True  # Custom configuration as needed
    }

    # Create HACO environment compatible with SB3
    env = SB3HACOEnv(env_config)

    # Wrap the environment in SB3's vectorized environment for better performance
    env = make_vec_env(lambda: env, n_envs=1)

    # Configure the logger
    log_dir = './logs/'
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

    # Define the RL algorithm (TD3 in this case)
    model = TD3('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)

    # Set up the checkpoint callback
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./models/', name_prefix='haco_model')

    # Set up the custom callback for CARLA metrics
    carla_callback_sb3 = CARLACallBackSB3()

    # Train the model
    model.learn(total_timesteps=100000, callback=[checkpoint_callback, carla_callback_sb3])
