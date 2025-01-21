import optuna
import airsim
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import json
from env import AirSimEnv


def make_env():
    """Creates a single environment instance."""
    env = AirSimEnv()
    env.seed(0)
    return env

def make_vec_env(n_envs):
    """Creates a vectorized environment."""
    if n_envs <= 4:
        return DummyVecEnv([make_env for _ in range(n_envs)])
    return SubprocVecEnv([make_env for _ in range(n_envs)])

def load_hyperparameters(file_path):
    """Loads hyperparameters from a JSON file."""
    try:
        with open(file_path, "r") as f:
            params = json.load(f)  # Load the JSON content as a dictionary
        return params
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in {file_path}: {e}")
        return {}
    except Exception as e:
        print(f"Error loading hyperparameters from {file_path}: {e}")
        return {}

def save_hyperparameters(params, file_path):
    """Saves hyperparameters to a JSON file."""
    try:
        with open(file_path, "w") as f:
            json.dump(params, f, indent=4)
        print(f"Hyperparameters saved to {file_path}")
    except Exception as e:
        print(f"Error saving hyperparameters to {file_path}: {e}")

def objective(trial):
    """Objective function for Optuna hyperparameter optimization."""
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3)
    n_steps = trial.suggest_int('n_steps', 1024, 4096, step=512)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    n_epochs = trial.suggest_int('n_epochs', 3, 20)
    gamma = trial.suggest_float('gamma', 0.9, 0.999)
    gae_lambda = trial.suggest_float('gae_lambda', 0.8, 0.99)
    ent_coef = trial.suggest_float('ent_coef', 1e-8, 1e-2)
    vf_coef = trial.suggest_float('vf_coef', 0.1, 1.0)

    env = make_vec_env(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        device=device,
        verbose=1,
    )

    model.learn(total_timesteps=50000)

    # Evaluate the model
    rewards = []
    for _ in range(5):
        obs = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            with torch.no_grad():
                action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        rewards.append(total_reward)

    mean_reward = sum(rewards) / len(rewards)

    del model
    torch.cuda.empty_cache()
    return mean_reward

def train(best_params):
    """Trains the PPO model with the best hyperparameters."""
    print("Starting final training with parameters:", best_params)
    n_envs = best_params.get('n_envs', 1)
    env = make_vec_env(n_envs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=best_params['learning_rate'],
        n_steps=best_params['n_steps'],
        batch_size=best_params['batch_size'],
        n_epochs=best_params['n_epochs'],
        gamma=best_params['gamma'],
        gae_lambda=best_params['gae_lambda'],
        ent_coef=best_params['ent_coef'],
        vf_coef=best_params['vf_coef'],
        device=device,
        verbose=1,
    )

    model.learn(total_timesteps=200000)
    model.save("ppo_drone_navigation")
    print("Final model saved as 'ppo_drone_navigation'.")

if __name__ == "__main__":
    hyperparam_file = "hyperparameters.txt"

    if os.path.exists(hyperparam_file):
        print("Hyperparameters file found. Skipping Optuna trials.")
        best_params = load_hyperparameters(hyperparam_file)
        if not best_params:
            print("Failed to load hyperparameters. Exiting.")
            exit(1)
    else:
        print("Hyperparameters file not found. Running Optuna trials.")

        sampler = optuna.samplers.TPESampler()
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials=10)

        best_params = study.best_params
        save_hyperparameters(best_params, hyperparam_file)

    train(best_params)
