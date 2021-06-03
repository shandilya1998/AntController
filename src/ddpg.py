# -*- coding: utf-8 -*-
import os
import argparse
import gym
import numpy as np
import matplotlib.pyplot as plt
import stable_baselines3
from stable_baselines3 import TD3, HerReplayBuffer, PPO, SAC, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from rl.torch.sb3.policy import TD3Policy, MultiInputPolicy, MultiInputPolicyV2, MultiInputPolicyV3
import torch
from torchsummary import summary
from rl.torch.constants import params



info_kwargs = (
    'reward_velocity',
    'reward_rotation',
    'reward_position',
    'reward',
    'reward_ctrl',
    'reward_orientation',
    'reward_motion',
    'reward_contact',
    'reward_torque'
)

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, env,batch_size, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.batch_size = batch_size
        self.env = env
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    """
    def _on_rollout_start(self):
        self.env.envs[0].reset_state()
    """

    def _on_step(self) -> bool:
        if self.n_calls % 100 == 1:
            df = load_results(self.log_dir)
            if len(df) > 0:
                reward_position = np.mean(df.reward_position.values[-100:])
                reward_velocity = np.mean(df.reward_velocity.values[-100:])
                reward_orientation = np.mean(df.reward_orientation.values[-100:])
                reward_rotation = np.mean(df.reward_rotation.values[-100:])
                reward_ctrl = np.mean(df.reward_ctrl.values[-100:])
                reward_motion = np.mean(df.reward_motion.values[-100:])
                reward_contact = np.mean(df.reward_contact.values[-100:])
                reward_torque = np.mean(df.reward_torque.values[-100:])
                reward = np.mean(df.reward.values[-100:])
                self.logger.record('reward_position', reward_position)
                self.logger.record('reward_velocity', reward_velocity)
                self.logger.record('reward_orientation', reward_orientation)
                self.logger.record('reward_rotation', reward_rotation)
                self.logger.record('reward_ctrl', reward_ctrl)
                self.logger.record('reward', reward)
                self.logger.record('reward_motion', reward_motion)
                self.logger.record('reward_contact', reward_contact)
                self.logger.record('reward_torque', reward_torque)
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            df = load_results(self.log_dir)
            x, y = ts2xy(df, 'timesteps')
            if len(x) > 0:
            # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)
                    #print(stable_baselines3.common.evaluation.evaluate_policy(self.model, self.env, render=True))
        return True

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()
    fig.savefig(os.path.join(log_folder, 'learning_curve.png'))

if __name__ == '__main__':
    # Create log dir
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment',
        type = int,
        help = 'ID of experiment being performaed'
    )
    parser.add_argument(
        '--out_path',
        type = str,
        help = 'Path to output directory'
    )
    parser.add_argument(
        '--env',
        type = str,
        help = 'environment name'
    )
    parser.add_argument(
        '--env_version',
        type = str,
        help = 'environment version'
    )
    parser.add_argument(
        '--model_dir',
        type = str,
        help = 'path of dir of the model to be loaded'
    )
    parser.add_argument(
        '--test_env',
        nargs='?', type=int, const=1,
        help = 'choice to test env before loading'
    )
    parser.add_argument(
        '--her',
        nargs='?', type = int, const = 1,
        help = 'choice to use HER replay buffer'
    )
    parser.add_argument(
        '--ppo',
        nargs='?', type = int, const = 1,
        help = 'choice to use PPO instead of TD3'
    )
    parser.add_argument(
        '--sac',
        nargs='?', type = int, const = 1,
        help = 'choice to use SAC instead of TD3'
    )
    parser.add_argument(
        '--a2c',
        nargs='?', type = int, const = 1,
        help = 'choice to use A2C instead of TD3'
    )
    parser.add_argument(
        '--standard',
        nargs='?', type = int, const = 1,
        help = 'choice to use standard policies instead of custom policies for baselines'
    )
    parser.add_argument(
        '--evaluate',
        nargs='?', type = int, const = 1,
        help = 'choice to evaluate or train'
    )
    args = parser.parse_args()
    path = os.path.join(args.out_path, 'exp{}'.format(args.experiment))
    if not os.path.exists(path):
        os.mkdir(path)
    tensorboard_log = os.path.join(path, 'tensorboard')
    if not os.path.exists(tensorboard_log):
        os.mkdir(tensorboard_log)
    log_dir = path
    env_name = args.env + '-v' + args.env_version
    import_name = args.env + 'V' + args.env_version

    if env_name != 'Ant-v2':
        gym.envs.registration.register(
            id=env_name,
            entry_point='simulations.gym.ant:{}'.format(import_name),
            max_episode_steps = params['rnn_steps'] * params['max_steps']
        )

    if args.test_env is not None:
        # Create and wrap the environment
        env = gym.make(env_name)
        from stable_baselines3.common.env_checker import check_env
        print(check_env(env))
        #env = _AntEnv()
        # Logs will be saved in log_dir/monitor.csv


    env = stable_baselines3.common.env_util.make_vec_env(
        env_name, monitor_dir = log_dir, monitor_kwargs = {
            'info_keywords' : info_kwargs
        },
    )


    # Create action noise because TD3 and DDPG use a deterministic policy
    n_actions = env.action_space.sample().shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))
    # Create the callback: check every 1000 steps
    # Create RL model
    batch_size = 128
    # Train the agent
    callback = SaveOnBestTrainingRewardCallback(env, batch_size, check_freq=6000, log_dir=log_dir, verbose = 1)

    model = None
    if args.ppo is None and args.a2c is None and args.sac is None:
        policy = TD3Policy
        if args.standard is not None:
            policy = 'MlpPolicy'
        model_class = TD3
        if int(args.env_version) == 0:
            model = model_class(
                policy,
                env,
                action_noise = action_noise,
                verbose = 2,
                tensorboard_log = log_dir,
                batch_size = batch_size,
                learning_starts = 1000,
                train_freq = (5, 'step')
            )
        elif int(args.env_version) > 0:
            policy_kwargs = {}
            policy = MultiInputPolicy
            if int(args.env_version) >= 4 and args.sac is None:
                policy = MultiInputPolicyV2
                if int(args.env_version) == 4:
                    policy = MultiInputPolicyV2
                elif int(args.env_version) == 5:
                    policy = MultiInputPolicyV3
            if args.standard is not None or args.sac is not None:
                policy = 'MultiInputPolicy'
            print('[DDPG] MultiInputPolicy')
            if args.her is None:
                model = model_class(
                    policy,
                    env,
                    learning_starts=5000,
                    action_noise = action_noise,
                    verbose = 2,
                    tensorboard_log = log_dir,
                    batch_size = batch_size,
                    train_freq = (5, 'step'),
                )
            else:
                print('[DDPG] Using HER')
                model = model_class(
                    policy,
                    env,
                    replay_buffer_class = HerReplayBuffer,
                    replay_buffer_kwargs = dict(
                        n_sampled_goal=4,
                        goal_selection_strategy='future',
                        online_sampling=True,
                        max_episode_length = params['rnn_steps'] * params['max_steps'],
                    ),
                    learning_starts=5000,
                    action_noise = action_noise,
                    verbose = 2,
                    tensorboard_log = log_dir,
                    batch_size = batch_size,
                    train_freq = (5, 'step'),
                )

            if args.model_dir is not None:
                actor = torch.load(os.path.join(args.model_dir, 'actor.pth'))
                critic = torch.load(os.path.join(args.model_dir, 'critic.pth'))
                actor_state_dict = prev_model.policy.actor.state_dict()
                critic_state_dict = prev_model.policy.critic.state_dict()
                model.policy.actor.load_state_dict(actor_state_dict, strict = False)
                model.policy.actor_target.load_state_dict(actor_state_dict, strict = False)
                model.policy.critic.load_state_dict(critic_state_dict, strict = False)
                model.policy.critic_target.load_state_dict(critic_state_dict, strict = False)
    elif args.ppo is not None:
        model = PPO(
            policy = 'MultiInputPolicy',
            env = env,
            policy_kwargs = {
                'net_arch' : [dict(pi=[512, 512], vf=[512, 512])],
                'activation_fn' : torch.nn.PReLU,
                'log_std_init' : -1,
                'ortho_init' : False
            },
            tensorboard_log = log_dir,
            verbose = 2,
            batch_size = batch_size,
            use_sde = True,
            sde_sample_freq = 4,
            n_epochs = 20,
            n_steps = params['rnn_steps'] * params['max_steps'],
            gae_lambda = 0.9,
            clip_range = 0.4,

        )
    elif args.a2c is not None:
        model = A2C(
            policy = 'MultiInputPolicy',
            env = env,
            policy_kwargs = {
                'net_arch' : [dict(pi=[512, 512], vf=[512, 512])],
                'activation_fn' : torch.nn.Tanh,
                'log_std_init' : -2,
                'ortho_init' : False
            },
            tensorboard_log = log_dir,
            verbose = 2,
            n_steps = 8,
            gae_lambda = 0.9,
            vf_coef=0.4,
            learning_rate = 0.00096,
            use_sde = True,
            normalize_advantage=True,
        )
    elif args.sac is not None:
        if args.her is None:
            model = SAC(
                'MultiInputPolicy',
                env,
                learning_starts=10000,
                action_noise = action_noise,
                verbose = 2,
                tensorboard_log = log_dir,
                batch_size = batch_size,
                gamma = 0.98,
                tau = 0.02,
                train_freq = 64,
                gradient_steps = 64,
                use_sde = True
            )
        else:
            print('[DDPG] Using HER')
            model = SAC(
                'MultiInputPolicy',
                env,
                replay_buffer_class = HerReplayBuffer,
                replay_buffer_kwargs = dict(
                    n_sampled_goal=4,
                    goal_selection_strategy='future',
                    online_sampling=True,
                    max_episode_length = params['rnn_steps'] * params['max_steps'],
                ),
                learning_starts=10000,
                action_noise = action_noise,
                verbose = 2,
                tensorboard_log = log_dir,
                batch_size = batch_size,
                gamma = 0.98,
                tau = 0.02,
                train_freq = 64,
                gradient_steps = 64,
                use_sde = True
            )


    steps = 1e6
    if args.evaluate is None:
        model.learn(total_timesteps=int(steps), callback=callback)
        model.save(log_dir + '/Policy')
        if args.ppo is not None or args.a2c is not None:
            torch.save(model.policy.action_net, os.path.join(log_dir, 'action_net.pth'))
            torch.save(model.policy.value_net, os.path.join(log_dir, 'value_net.pth'))
        elif args.sac is not None:
            torch.save(model.actor, os.path.join(log_dir, 'actor.pth'))
            torch.save(model.critic, os.path.join(log_dir, 'critic.pth'))
            torch.save(model.critic_target, os.path.join(log_dir, 'critic_target.pth'))
        else:
            torch.save(model.actor, os.path.join(log_dir, 'actor.pth'))
            torch.save(model.critic, os.path.join(log_dir, 'critic.pth'))
            torch.save(model.critic_target, os.path.join(log_dir, 'critic_target.pth'))
            torch.save(model.actor_target, os.path.joint(log_dir, 'actor_target.pth'))
    else:
        model_class = None
        if args.ppo is not None:
            model_class = PPO
        elif args.sac is not None:
            model_class = SAC
        elif args.a2c is not None:
            model_class = A2C
        else:
            model_class = TD3
        model = model_class.load(log_dir + '/Policy', env = env, custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        })
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'gpu'
        if args.ppo is not None or args.a2c is not None:
            model.policy.action_net.load_state_dict(torch.load(os.path.join(log_dir, 'action_net.pth'), map_location=torch.device('cpu')))
            model.policy.value_net.load_state_dict(torch.load(os.path.join(log_dir, 'value_net.pth'), map_location=torch.device('cpu')))
        else:
            model.actor.load_state_dict(torch.load(os.path.join(log_dir, 'actor.pth'), map_location=torch.device('cpu')))
            model.critic.load_state_dict(torch.load(os.path.join(log_dir, 'critic.pth'), map_location=torch.device('cpu')))
            model.critic_target.load_state_dict(torch.load(os.path.join(log_dir, 'critic_target.pth'), map_location=torch.device('cpu')))
            if args.sac is None:
                model.actor_target.load_state_dict(torch.load(os.path.join(log_dir, 'actor_target.pth'), map_location=torch.device('cpu')))
        print(stable_baselines3.common.evaluation.evaluate_policy(model, env, render=True))
