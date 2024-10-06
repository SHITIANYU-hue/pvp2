import argparse
import os
from pathlib import Path
import time

from pvp.pvp_td3 import PVPTD3
from pvp.sb3.common.callbacks import CallbackList, CheckpointCallback
from pvp.sb3.common.monitor import Monitor
from pvp.sb3.common.wandb_callback import WandbCallback
from pvp.utils.shared_control_monitor import SharedControlMonitor
from pvp.utils.utils import get_time_str

# Import CarlaRouteEnv
from carla_env.envs.carla_route_env import CarlaRouteEnv
from carla_env.state_commons import create_encode_state_fn, load_vae
from carla_env.rewards import reward_functions
from vae.utils.misc import LSIZE
from utils import parse_wrapper_class
from config import CONFIG

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trains a PVP agent in the CARLA environment using TD3")
    parser.add_argument("--exp_name", default="pvp_carla", type=str, help="The name for this batch of experiments.")
    parser.add_argument("--seed", default=0, type=int, help="The random seed.")
    parser.add_argument("--wandb", action="store_true", help="Set to True to upload stats to wandb.")
    parser.add_argument("--wandb_project", type=str, default="", help="The project name for wandb.")
    parser.add_argument("--wandb_team", type=str, default="", help="The team name for wandb.")
    parser.add_argument("--obs_mode", default="birdview", choices=["birdview", "first", "birdview42", "firststack"], help="The observation mode.")
    parser.add_argument("--port", default=9000, type=int, help="Carla server port.")
    parser.add_argument("--total_timesteps", type=int, default=50_000, help="Total timesteps to train for")
    parser.add_argument("--reload_model", type=str, default="", help="Path to a model to reload")
    parser.add_argument("--config", type=str, default="1", help="Config to use (default: 1)")

    args = parser.parse_args()

    # Setup experiment directory
    experiment_batch_name = "{}_{}".format(args.exp_name, args.obs_mode)
    trial_name = "{}_{}".format(experiment_batch_name, get_time_str())
    trial_dir = Path("runs") / experiment_batch_name / trial_name
    os.makedirs(trial_dir, exist_ok=True)

    # Load environment config from CarlaRouteEnv
    config.set_config(args.config)

    vae = None
    if CONFIG["vae_model"]:
        vae = load_vae(f'./vae/log_dir/{CONFIG["vae_model"]}', LSIZE)

    observation_space, encode_state_fn, decode_vae_fn = create_encode_state_fn(vae, CONFIG["state"])

    env = CarlaRouteEnv(
        obs_res=CONFIG["obs_res"], 
        host="localhost", 
        port=args.port,
        reward_fn=reward_functions[CONFIG["reward_fn"]],
        observation_space=observation_space,
        encode_state_fn=encode_state_fn, 
        decode_vae_fn=decode_vae_fn,
        fps=15, 
        action_smoothing=CONFIG["action_smoothing"],
        action_space_type='continuous', 
        activate_spectator=False, 
        activate_render=True
    )

    # Apply wrappers if needed
    for wrapper_class_str in CONFIG["wrappers"]:
        wrap_class, wrap_params = parse_wrapper_class(wrapper_class_str)
        env = wrap_class(env, *wrap_params)

    # Setup callbacks
    save_freq = args.total_timesteps // 10  # Saving model every 10% of total timesteps
    callbacks = [
        CheckpointCallback(save_freq=save_freq, save_path=str(trial_dir / "models"), name_prefix="rl_model")
    ]
    if args.wandb:
        callbacks.append(
            WandbCallback(
                trial_name=trial_name,
                exp_name=experiment_batch_name,
                team_name=args.wandb_team,
                project_name=args.wandb_project,
                config=CONFIG
            )
        )
    callbacks = CallbackList(callbacks)

    # Setup the training algorithm (using PVPTD3)
    model = PVPTD3(
        policy=TD3Policy,
        env=env,
        learning_rate=1e-4,
        q_value_bound=1,
        optimize_memory_usage=True,
        buffer_size=50_000,  # We only conduct experiment less than 50K steps
        learning_starts=100,
        batch_size=128,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        action_noise=None,
        tensorboard_log=str(trial_dir),
        verbose=2,
        seed=args.seed,
        device="auto"
    )

    # Reload model if specified
    if args.reload_model:
        model.load(args.reload_model, env=env)

    # Train the model
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        reset_num_timesteps=True
    )
