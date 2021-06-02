import warnings
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch as th

from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.preprocessing import get_obs_shape
from stable_baselines3.common.type_aliases import DictReplayBufferSamples
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from stable_baselines3.her.goal_selection_strategy import KEY_TO_GOAL_STRATEGY, GoalSelectionStrategy
from stable_baselines3.common.type_aliases import ReplayBufferSamples

class RDPGDictReplayBufferSamples(ReplayBufferSamples):
    observations: TensorDict
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    input_state: th.Tensor
    output_state: th.Tensor


class RDPGReplayBuffer(HerReplayBuffer):
    def __init__(
        self,
        env: VecEnv,
        buffer_size: int,
        device: Union[th.device, str] = "cpu",
        replay_buffer: Optional[DictReplayBuffer] = None,
        max_episode_length: Optional[int] = None,
        trajectory_length: Optional[int] = 0,
        n_sampled_goal: int = 4,
        goal_selection_strategy: Union[GoalSelectionStrategy, str] = "future",
        online_sampling: bool = True,
        handle_timeout_termination: bool = True,
    ):
        super(RDPGReplayBuffer, self).__init__(
            env,
            buffer_size,
            device,
            replay_buffer,
            max_episode_length,
            n_sampled_goal,
            goal_selection_strategy
        )
        self.trajectory_length = trajectory_length
        self.replay_buffer = None
        input_shape = {
            "observation": (self.env.num_envs,) + self.obs_shape,
            "achieved_goal": (self.env.num_envs,) + self.goal_shape,
            "desired_goal": (self.env.num_envs,) + self.goal_shape,
            "state": (2 * self.action_dim + 2 * self.obs_shape[-1],)
            "action": (self.action_dim,),
            "reward": (1,),
            "next_obs": (self.env.num_envs,) + self.obs_shape,
            "next_achieved_goal": (self.env.num_envs,) + self.goal_shape,
            "next_desired_goal": (self.env.num_envs,) + self.goal_shape,
            "next_state": (2 * self.action_dim + 2 * self.obs_shape[-1],)
            "done": (1,),
        }
        self._observation_keys = ["observation", "achieved_goal", "desired_goal"]
        self._buffer = {
            key: np.zeros((self.max_episode_stored, self.max_episode_length, *dim), dtype=np.float32)
            for key, dim in input_shape.items()
        }

    def _sample_transitions(
        self,
        batch_size: Optional[int],
        maybe_vec_env: Optional[VecNormalize],
        online_sampling: bool,
        n_sampled_goal: Optional[int] = None,
    ) -> Union[DictReplayBufferSamples, Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray]]:
        """
        :param batch_size: Number of element to sample (only used for online sampling)
        :param env: associated gym VecEnv to normalize the observations/rewards
            Only valid when using online sampling
        :param online_sampling: Using online_sampling for HER or not.
        :param n_sampled_goal: Number of sampled goals for replay. (offline sampling)
        :return: Samples.
        """
        # Select which episodes to use
        if online_sampling:
            assert batch_size is not None, "No batch_size specified for online sampling of HER transitions"
            # Do not sample the episode with index `self.pos` as the episode is invalid
            if self.full:
                episode_indices = (
                    np.random.randint(1, self.n_episodes_stored, batch_size) + self.pos
                ) % self.n_episodes_stored
            else:
                episode_indices = np.random.randint(0, self.n_episodes_stored, batch_size)
            # A subset of the transitions will be relabeled using HER algorithm
            her_indices = np.arange(batch_size)[: int(self.her_ratio * batch_size)]
        else:
            assert maybe_vec_env is None, "Transitions must be stored unnormalized in the replay buffer"
            assert n_sampled_goal is not None, "No n_sampled_goal specified for offline sampling of HER transitions"
            # Offline sampling: there is only one episode stored
            episode_length = self.episode_lengths[0] - self.trajectory_length
            # we sample n_sampled_goal per timestep in the episode (only one is stored).
            episode_indices = np.tile(0, (episode_length * n_sampled_goal))
            # we only sample virtual transitions
            # as real transitions are already stored in the replay buffer
            her_indices = np.arange(len(episode_indices))

        ep_lengths = self.episode_lengths[episode_indices]

        # Special case when using the "future" goal sampling strategy
        # we cannot sample all transitions, we have to remove the last timestep
        if self.goal_selection_strategy == GoalSelectionStrategy.FUTURE:
            # restrict the sampling domain when ep_lengths > 1
            # otherwise filter out the indices
            her_indices = her_indices[ep_lengths[her_indices] > 1]
            ep_lengths[her_indices] -= 1

        if online_sampling:
            # Select which transitions to use
            lengths = ep_lengths - self.trajectory_length
            transitions_indices = np.random.randint(lengths)
        else:
            if her_indices.size == 0:
                # Episode of one timestep, not enough for using the "future" strategy
                # no virtual transitions are created in that case
                return {}, {}, np.zeros(0), np.zeros(0)
            else:
                # Repeat every transition index n_sampled_goals times
                # to sample n_sampled_goal per timestep in the episode (only one is stored).
                # Now with the corrected episode length when using "future" strategy
                transitions_indices = np.tile(np.arange(ep_lengths[0] - self.trajecory_length), n_sampled_goal)
                episode_indices = episode_indices[transitions_indices]
                her_indices = np.arange(len(episode_indices))



        out = []
        # get selected transitions
        for i in range(self.trajectory_length):
            transitions = {key: self._buffer[key][episode_indices, transitions_indices + i].copy() \
                for key in self._buffer.keys()}

            # sample new desired goals and relabel the transitions
            new_goals = self.sample_goals(episode_indices, her_indices, transitions_indices + i)
            transitions["desired_goal"][her_indices] = new_goals

            # Convert info buffer to numpy array
            transitions["info"] = np.array(
                [
                    self.info_buffer[episode_idx][transition_idx]
                    for episode_idx, transition_idx in zip(episode_indices, transitions_indices + i)
                ]
            )
            # Edge case: episode of one timesteps with the future strategy
            # no virtual transition can be created
            if len(her_indices) > 0:
                # Vectorized computation of the new reward
                for i in range(self.trajectory_length):
                    transitions["reward"][her_indices, 0] = self.env.env_method(
                        "compute_reward",
                        # the new state depends on the previous state and action
                        # s_{t+1} = f(s_t, a_t)
                        # so the next_achieved_goal depends also on the previous state and action
                        # because we are in a GoalEnv:
                        # r_t = reward(s_t, a_t) = reward(next_achieved_goal, desired_goal)
                        # therefore we have to use "next_achieved_goal" and not "achieved_goal"
                        transitions["next_achieved_goal"][her_indices, 0],
                        # here we use the new desired goal
                        transitions["desired_goal"][her_indices, 0],
                        transitions["info"][her_indices, 0],
                    )

            # concatenate observation with (desired) goal
            observations = self._normalize_obs(transitions, maybe_vec_env)

            # HACK to make normalize obs and `add()` work with the next observation
            next_observations = {
                "observation": transitions["next_obs"],
                "achieved_goal": transitions["next_achieved_goal"],
                # The desired goal for the next observation must be the same as the previous one
                "desired_goal": transitions["desired_goal"],
            }
            next_observations = self._normalize_obs(next_observations, maybe_vec_env)

            if online_sampling:
                next_obs = {key: self.to_torch(next_observations[key][:, 0, :]) for key in self._observation_keys}

                normalized_obs = {key: self.to_torch(observations[key][:, 0, :]) for key in self._observation_keys}

                out.append(DictReplayBufferSamples(
                    observations=normalized_obs,
                    actions=self.to_torch(transitions["action"]),
                    next_observations=next_obs,
                    dones=self.to_torch(transitions["done"]),
                    rewards=self.to_torch(self._normalize_reward(transitions["reward"], maybe_vec_env)),
                ))
            else:
                out.append(observations, next_observations, transitions["action"], transitions["reward"])
        return out

    def add(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:

        if self.current_idx == 0 and self.full:
            # Clear info buffer
            self.info_buffer[self.pos] = deque(maxlen=self.max_episode_length)

        # Remove termination signals due to timeout
        if self.handle_timeout_termination:
            done_ = done * (1 - np.array([info.get("TimeLimit.truncated", False) for info in infos]))
        else:
            done_ = done

        self._buffer["observation"][self.pos][self.current_idx] = obs["observation"]
        self._buffer["achieved_goal"][self.pos][self.current_idx] = obs["achieved_goal"]
        self._buffer["desired_goal"][self.pos][self.current_idx] = obs["desired_goal"]
        self._buffer["state"][self.pos][self.current)idx] = obs['state']
        self._buffer["action"][self.pos][self.current_idx] = action
        self._buffer["done"][self.pos][self.current_idx] = done_
        self._buffer["reward"][self.pos][self.current_idx] = reward
        self._buffer["next_obs"][self.pos][self.current_idx] = next_obs["observation"]
        self._buffer["next_achieved_goal"][self.pos][self.current_idx] = next_obs["achieved_goal"]
        self._buffer["next_desired_goal"][self.pos][self.current_idx] = next_obs["desired_goal"]
        self._buffer["next_state"][self.pos][self.current_idx] = obs['next_state']

        # When doing offline sampling
        # Add real transition to normal replay buffer
        if self.replay_buffer is not None:
            self.replay_buffer.add(
                obs,
                next_obs,
                action,
                reward,
                done,
                infos,
            )

        self.info_buffer[self.pos].append(infos)

        # update current pointer
        self.current_idx += 1

        self.episode_steps += 1

        if done or self.episode_steps >= self.max_episode_length:
            self.store_episode()
            if not self.online_sampling:
                # sample virtual transitions and store them in replay buffer
                self._sample_her_transitions()
                # clear storage for current episode
                self.reset()

            self.episode_steps = 0
