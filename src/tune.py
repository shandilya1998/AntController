import optuna
import gym
import numpy as np
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from torch import nn as nn
from simulations.gym.ant import AntEnvV7
from typing import Any, Dict
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import sys

step = 0

def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for PPO hyperparams.

    :param trial:
    :return:
    """
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform("lr", 1e-5, 1)
    lr_schedule = "constant"
    # Uncomment to enable learning rate schedule
    # lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
    ent_coef = trial.suggest_loguniform("ent_coef", 0.00000001, 0.1)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5]) 
    vf_coef = trial.suggest_uniform("vf_coef", 0, 1)
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])
    # Uncomment for gSDE (continuous actions)
    log_std_init = trial.suggest_uniform("log_std_init", -4, 1)
    sde_sample_freq = trial.suggest_categorical("sde_sample_freq", [-1, 8, 16, 32, 64, 128, 256])
    # Orthogonal initialization
    #ortho_init = False
    ortho_init = trial.suggest_categorical('ortho_init', [False, True])
    activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])

    # TODO: account when using multiple envs
    if batch_size > n_steps:
        batch_size = n_steps

    if lr_schedule == "linear":
        learning_rate = linear_schedule(learning_rate)

    # Independent networks usually work best
    # when not working with images
    net_arch = {
        "small": [dict(pi=[64, 64], vf=[64, 64])],
        "medium": [dict(pi=[256, 256], vf=[256, 256])],
        "large" : [dict(pi=[512, 512], vf=[512, 512])]
    }[net_arch]

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU, 'prelu': nn.PReLU}[activation_fn]
    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
        # "sde_sample_freq": sde_sample_freq,
        "policy_kwargs": dict(
            # log_std_init=log_std_init,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
        ),
    }

def optimize_env(trial):
    "Learning Hyperparameters for the env"
    return np.array([
        trial.suggest_loguniform('w1', 0.1, 1),
        trial.suggest_loguniform('w2', 0.1, 1),
        trial.suggest_loguniform('w3', 0.1, 1),
        trial.suggest_loguniform('w4', 0.1, 1),
        trial.suggest_loguniform('w5', 0.1, 1)
    ])


def optimize_agent(trial):
    """ Train the model and optimize
        Optuna maximises the negative log likelihood, so we
        need to negate the reward here
    """
    logging.info('Starting Trial {}'.format(step))
    model_params = sample_ppo_params(trial)
    w = optimize_env(trial)
    def get_env():
        _env = AntEnvV7()
        _env.set_w(w)
        return _env
    env = make_vec_env(get_env, n_envs=16, seed=0)
    model = PPO('MultiInputPolicy', env, verbose=0, **model_params)
    model.learn(50000)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
    logging.info('Train Mean reward: {}'.format(mean_reward))
    logging.info('Trial Over')
    return -1 * mean_reward


if __name__ == '__main__':
    optuna.logging.set_verbosity(optuna.logging.DEBUG)
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "quadruped"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists = True)
    try:
        study.optimize(optimize_agent, n_trials=100, n_jobs=8)
    except KeyboardInterrupt:
        print('Interrupted by keyboard.')
    trial = study.best_trial
    print("Best hyperparameters: {}".format(trial.params))
