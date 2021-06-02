from stable_baselines3 import TD3

class RDPG(TD3):
    def __init__(
        self,
        policy: Union[str, Type[TD3Policy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = 1000000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 100,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
        gradient_steps: int = -1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Dict[str, Any] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super(RDPG, self).__init__(
            policy,
            env,
            learning_rate,
            buffer_size,  # 1e6
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class,
            replay_buffer_kwargs,
            optimize_memory_usage,
            policy_delay,
            target_policy_noise,
            target_noise_clip,
            tensorboard_log,
            create_eval_env,
            policy_kwargs,
            verbose,
            seed,
            device,
            _init_setup_model,
        )
        self.trajectory_length = replay_buffer_kwargs['trajectory_length']

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []

        for _ in range(gradient_steps):

            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            total_critic_loss = 0.0
            total_actor_loss = 0.0
            self.critic.reset_state(self.batch_size)
            self.critic_target.reset_state(self.batch_size)
            ob_size = 0
            for space in self.observation_spaces.spaces:
                ob_size += space.shape[0]
            critic_state = torch.zeros((self.batch_size, 2 * (self.action_space.shape[0] + ob_size)))
            for i in range(self.trajectory_length):
                with th.no_grad():
                    # Select action according to policy and add clipped noise
                    noise = replay_data[i].actions.clone().data.normal_(0, self.target_policy_noise)
                    noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                    next_actions = (self.actor_target(replay_data[i].next_observations) + noise).clamp(-1, 1)

                    # Compute the next Q-values: min over all critics targets
                    next_q_values = th.cat(self.critic_target(replay_data[i].next_observations, next_actions), dim=1)
                    next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                    target_q_values = replay_data[i].rewards + (1 - replay_data[i].dones) * self.gamma * next_q_values

                # Get current Q-values estimates for each critic network
                current_q_values = self.critic(replay_data[i].observations, replay_data[i].actions)

                # Compute critic loss
                critic_loss = sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])
                total_critic_loss += critic_loss

                if self._n_updates % self.policy_delay == 0:
                    q_val, critic_state = self.critic.q1_forward(replay_data[i].observations, self.actor(replay_data[i].observations), critic_state).mean()
                    actor_loss = -q_val.mean()
                    total_actor_loss += actor_loss

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            total_critic_loss.backward()
            self.critic.optimizer.step()
            critic_losses.append(total_critic_loss.item())
            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Optimize the actor
                actor_losses.append(total_actor_loss.item())
                self.actor.optimizer.zero_grad()
                total_actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)

        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            logger.record("train/actor_loss", np.mean(actor_losses))
        logger.record("train/critic_loss", np.mean(critic_losses))
