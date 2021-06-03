from collections import OrderedDict
import numpy as np
import gym
from gym.envs.mujoco import mujoco_env
from rl.torch.constants import params
import random
from gym import utils, error, spaces
import skinematics as skin

def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float('inf'), dtype=np.float32)
        high = np.full(observation.shape, float('inf'), dtype=np.float32)
        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space

class AntEnvV0(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, path = 'ant.xml'):
        mujoco_env.MujocoEnv.__init__(self, path, 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        penalty = 0.0
        info = {}
        if np.isnan(a).any():
            print('[DDPG] Action NaN')
            a = np.zeros(shape = a, dtype = action.dtype)
            penalty = -5.0
        xposbefore = self.get_body_com("torso")[0]
        yposbefore = self.get_body_com("torso")[1]
        self.do_simulation(a, self.frame_skip)
        #self.render()
        xposafter = self.get_body_com("torso")[0]
        yposafter = self.get_body_com("torso")[1]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        info['reward'] = reward
        info['reward_sideways'] = (yposafter - yposbefore)/self.dt
        info['reward_rotation'] = self.sim.data.qvel[5]
        info['reward_height'] = self.sim.data.qpos[2]
        info['penalty'] = penalty
        info['reward_forward'] = forward_reward
        info['reward_ctrl'] = -ctrl_cost
        info['reward_contact'] = -contact_cost
        info['reward_survive'] = survive_reward
        info['forward_vel'] = forward_reward
        return ob, reward, done, info

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

class AntEnvV1(gym.GoalEnv, mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, path = 'ant.xml'):
        mujoco_env.MujocoEnv.__init__(self, path, 5)
        utils.EzPickle.__init__(self)
        gym.GoalEnv.__init__(self)

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def _create_desired_goal_lst(self):
        self.w = np.array([0.375, 0.175, 0.175, 0.175, 0.1], dtype = np.float32)
        self.xvels = np.arange(0, 3, 0.01)
        self.yvels = np.arange(0, 2, 0.01)
        self.h = np.arange(0.4, 1.0, 0.01)
        self.yaws = np.arange(-0.5, 0.5, 0.01)
        xvels = np.random.choice(self.xvels, 300).tolist()
        yvels = np.random.choice(self.yvels, 200).tolist() + [0.0] * 100
        zvels = np.zeros((300,), dtype = np.float32)
        hs = np.random.choice(self.h, 300).tolist()
        rolls = np.zeros((300,), dtype = np.float32)
        pitches = np.zeros((300,), dtype = np.float32)
        yaws = np.random.choice(self.yaws, 200).tolist() + [0.0] * 100
        #self.desired_motions = [np.array([xvel, yvel, zvel, h, roll, pitch, yaw], dtype = np.float32) \
        #    for xvel, yvel, zvel, h, roll, pitch, yaw in zip(xvels, yvels, zvels, hs, rolls, pitches, yaws)]
        self.desired_motions = [np.array([xvel, 0, 0.75, 0], dtype = np.float32) \
            for xvel, yvel, h, yaw in zip(xvels, yvels, hs, yaws)]
        return self.desired_motions

    def _set_action_space(self):
        self.params = params
        self.desired_motions = self._create_desired_goal_lst()
        self.desired_goal = random.choice(self.desired_motions)
        self.achieved_goal = np.zeros((self.params['motion_state_size'],), dtype = np.float32)
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def reset(self):
        self.desired_goal = random.choice(self.desired_motions)
        self.achieved_goal = np.zeros((self.params['motion_state_size'],), dtype = np.float32)
        self.sim.reset()
        self.ob = self.reset_model()
        return {
            'observation' : self.ob,
            'desired_goal' : self.desired_goal,
            'achieved_goal' : self.achieved_goal
        }

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = None
        reward_forward = None
        reward_sideways = None
        reward_rotation = None
        reward_height = None
        if len(desired_goal.shape) > 1:
            reward_forward = np.exp(-np.square(desired_goal[:, 0] - achieved_goal[:, 0])) * self.w[0]
            reward_sideways = np.exp(-np.square(desired_goal[:, 1] - achieved_goal[:, 1])) * self.w[0]
            reward_rotation = np.exp(-np.square(desired_goal[:, 3] - achieved_goal[:, 3])) * self.w[1]
            reward_height = np.exp(-np.square(desired_goal[:, 2] - achieved_goal[:, 2])) * self.w[2]
        else:
            reward_forward = np.exp(-np.square(desired_goal[0] - achieved_goal[0])) * self.w[0]
            reward_sideways = np.exp(-np.square(desired_goal[1] - achieved_goal[1])) * self.w[0]
            reward_rotation = np.exp(-np.sum(np.square(desired_goal[3] - achieved_goal[3]), -1)) * self.w[1]
            reward_height = np.exp(-np.square(desired_goal[2] - achieved_goal[2])) * self.w[2]
        if isinstance(info, np.ndarray):
            out = {}
            for item in info:
                for key in item.keys():
                    if key in out.keys():
                        out[key].append(item[key])
                    else:
                        out[key] = [item[key]]
            out = {
                key : np.array(out[key], dtype = np.float32) \
                    for key in out.keys()
            }
            info = out
            reward = np.concatenate([
                np.expand_dims(reward_forward, -1),
                np.expand_dims(info['penalty'], -1),
                np.expand_dims(info['reward_ctrl'], -1),
                np.expand_dims(info['forward_vel'], -1),
                np.expand_dims(info['reward_contact'], -1),
                np.expand_dims(info['reward_survive'], -1),
            ], -1)
        else:
            reward = np.array([
                reward_forward,
                info['penalty'],
                info['reward_ctrl'],
                info['forward_vel'],
                info['reward_contact'],
                info['reward_survive'],
            ], dtype = np.float32)
        reward = np.round(np.sum(reward, -1).astype(np.float32), 6)
        return reward

    def step(self, action):
        penalty = 0.0
        if np.isnan(action).any():
            print('[DDPG] Action NaN')
            action = np.zeros(shape = action, dtype = action.dtype)
            penalty = -5.0
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore)/self.dt
        self.ob = self._get_obs()
        info = {}
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        #self.render()

        self.achieved_goal = np.array([
            self.sim.data.qvel[0],
            self.sim.data.qvel[1],
            self.sim.data.qpos[2],
            self.sim.data.qvel[5],
        ], dtype = np.float32)

        ctrl_cost = .5 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        info['reward_contact'] = -contact_cost
        info['reward_ctrl'] = -ctrl_cost
        info['reward_survive'] = survive_reward
        info['forward_vel'] = forward_reward
        info['penalty'] = penalty
        reward_forward = np.exp(-np.square(self.desired_goal[0] - self.achieved_goal[0])) * self.w[0]
        reward_sideways = np.exp(-np.square(self.desired_goal[1] - self.achieved_goal[1])) * self.w[0]
        reward_rotation = np.exp(-np.sum(np.square(self.desired_goal[3] - self.achieved_goal[3]), -1)) * self.w[1]
        reward_height = np.exp(-np.square(self.desired_goal[2] - self.achieved_goal[2])) * self.w[2]

        reward = info['reward_contact'] + \
            info['reward_survive'] + info['forward_vel'] + info['penalty'] + \
            reward_forward + info['reward_ctrl']
        info['reward'] = reward
        info['reward_forward'] = reward_forward
        info['reward_sideways'] = reward_sideways
        info['reward_rotation'] = reward_rotation
        info['reward_height'] = reward_height
        return {'observation' : self.ob, 'desired_goal' : self.desired_goal, 'achieved_goal' : self.achieved_goal}, reward, done, info

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5


class AntEnvV2(AntEnvV1):
    def __init__(self, path = 'ant.xml'):
        super(AntEnvV2, self).__init__(path)

    def _create_desired_goal_lst(self):
        self.h = 0.75
        self.xpos = np.arange(1, 6, 0.1)
        self.ypos = np.arange(1, 6, 0.1)
        self.desired_motions = []
        for i in range(50):
            self.desired_motions.append(np.array([self.xpos[i], self.ypos[i], self.h], dtype = np.float32))

        return self.desired_motions

    def step(self, action):
        penalty = 0.0
        if np.isnan(action).any():
            print('[DDPG] Action NaN')
            action = np.zeros(shape = action, dtype = action.dtype)
            penalty = -5.0
        posbefore = self.get_body_com("torso").copy()
        self.do_simulation(action, self.frame_skip)
        posafter = self.get_body_com("torso").copy()
        forward_reward = np.sqrt(np.sum(np.square((posafter - posbefore)/self.dt)[:2]))
        self.ob = self._get_obs()
        info = {}
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        #self.render()

        self.achieved_goal = posafter

        ctrl_cost = .5 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        info['reward_contact'] = -contact_cost
        info['reward_ctrl'] = -ctrl_cost
        info['reward_survive'] = survive_reward
        info['forward_vel'] = min(forward_reward, 2.5)
        info['penalty'] = penalty
        info['reward_position'] = np.exp(-np.sum(np.square(self.achieved_goal - self.desired_goal)[:2], -1))

        reward = info['reward_contact'] + \
            info['reward_survive'] + info['penalty'] + \
            info['reward_position'] + info['reward_ctrl'] + info['forward_vel']
        info['reward'] = reward
        return {'observation' : self.ob, 'desired_goal' : self.desired_goal, 'achieved_goal' : self.achieved_goal}, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = None
        reward_position = np.exp(-np.sum(np.square(achieved_goal - desired_goal), -1))
        if isinstance(info, np.ndarray):
            out = {}
            for item in info:
                for key in item.keys():
                    if key in out.keys():
                        out[key].append(item[key])
                    else:
                        out[key] = [item[key]]
            out = {
                key : np.array(out[key], dtype = np.float32) \
                    for key in out.keys()
            }
            info = out
            reward = np.concatenate([
                np.expand_dims(reward_position, -1),
                np.expand_dims(info['penalty'], -1),
                np.expand_dims(info['forward_vel'], -1),
                np.expand_dims(info['reward_ctrl'], -1),
                np.expand_dims(info['reward_contact'], -1),
                np.expand_dims(info['reward_survive'], -1),
            ], -1)
        else:
            reward = np.array([
                reward_position,
                info['penalty'],
                info['forward_vel'],
                info['reward_ctrl'],
                info['reward_contact'],
                info['reward_survive'],
            ], dtype = np.float32)
        reward = np.round(np.sum(reward, -1).astype(np.float32), 6)
        return reward

class AntEnvV3(AntEnvV1):
    def __init__(self, path = 'ant.xml'):
        super(AntEnvV3, self).__init__(path)

    def _set_action_space(self):
        self.params = params
        self.desired_motions = self._create_desired_goal_lst()
        self.desired_command = random.choice(self.desired_motions)
        self.desired_goal = np.zeros((self.params['motion_state_size'],), dtype = np.float32)
        self.desired_goal[:2] = self.init_qpos[:2]
        self.desired_goal[2:] = self.desired_command
        self.achieved_goal = np.zeros((self.params['motion_state_size'],), dtype = np.float32)
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space


    def _create_desired_goal_lst(self):
        self.xvels = np.arange(0, 3, 0.001)
        self.yvels = np.arange(0, 2, 0.001)
        self.h = np.arange(0.5, 1.0, 0.001)
        self.yaws = np.arange(-0.5, 0.5, 0.001)
        xvels = np.random.choice(self.xvels, 3000).tolist()
        yvels = np.random.choice(self.yvels, 2000).tolist() + [0.0] * 1000
        zvels = np.zeros((3000,), dtype = np.float32)
        hs = np.random.choice(self.h, 3000).tolist()
        rolls = np.zeros((3000,), dtype = np.float32)
        pitches = np.zeros((3000,), dtype = np.float32)
        yaws = np.random.choice(self.yaws, 2000).tolist() + [0.0] * 1000
        #self.desired_motions = [np.array([xvel, yvel, zvel, h, roll, pitch, yaw], dtype = np.float32) \
        #    for xvel, yvel, zvel, h, roll, pitch, yaw in zip(xvels, yvels, zvels, hs, rolls, pitches, yaws)]
        self.desired_motions = [np.array([xvel, yvel, h, yaw], dtype = np.float32) \
            for xvel, yvel, h, yaw in zip(xvels, yvels, hs, yaws)]
        return self.desired_motions

    def reset(self):
        self.desired_command = random.choice(self.desired_motions)
        self.desired_goal = np.zeros((self.params['motion_state_size'],), dtype = np.float32)
        self.desired_goal[:2] = self.init_qpos[:2]
        self.desired_goal[2:] = self.desired_command
        self.achieved_goal = np.zeros((self.params['motion_state_size'],), dtype = np.float32)
        self.sim.reset()
        self.ob = self.reset_model()
        return {
            'observation' : self.ob,
            'desired_goal' : self.desired_goal,
            'achieved_goal' : self.achieved_goal
        }


    def step(self, action):
        penalty = 0.0
        if np.isnan(action).any():
            print('[DDPG] Action NaN')
            action = np.zeros(shape = action, dtype = action.dtype)
            penalty = -5.0
        self.do_simulation(action, self.frame_skip)
        self.ob = self._get_obs()
        info = {}
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        #self.render()

        self.achieved_goal = np.array([
            self.sim.data.qpos[0],
            self.sim.data.qpos[1],
            self.sim.data.qvel[0],
            self.sim.data.qvel[1],
            self.sim.data.qpos[2],
            self.sim.data.qvel[5],
        ], dtype = np.float32)
        self.desired_goal[:2] += self.desired_goal[2:4] * self.dt

        ctrl_cost = .5 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        info['reward_contact'] = -contact_cost
        info['reward_ctrl'] = -ctrl_cost
        info['reward_survive'] = survive_reward + min(0.6, np.sum(np.square(self.sim.data.qvel[:2])))
        info['penalty'] = penalty
        info['reward_position'] = np.exp(-np.sum(np.square(self.achieved_goal[:2] - self.desired_goal[:2]), -1)) * 0.9
        info['reward_velocity'] = np.exp(-np.sum(np.square(self.achieved_goal[2:4] - self.desired_goal[2:4]), -1)) * 0.9
        info['reward_height'] = np.exp(-np.sum(np.square(self.achieved_goal[4] - self.desired_goal[4]), -1)) * 0.1
        info['reward_yaw'] = np.exp(-np.sum(np.square(self.achieved_goal[5] - self.desired_goal[5]), -1)) * 0.1

        reward = info['reward_contact'] + info['reward_velocity'] + \
            info['reward_survive'] + info['penalty'] + \
            info['reward_position'] + info['reward_ctrl'] + \
            info['reward_height'] + info['reward_yaw']
        info['reward'] = reward
        return {'observation' : self.ob, 'desired_goal' : self.desired_goal, 'achieved_goal' : self.achieved_goal}, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = None
        reward_position = None
        reward_velocity = None
        reward_height = None
        reward_yaw = None
        if len(desired_goal.shape) > 1:
            reward_position = np.exp(-np.sum(np.square(desired_goal[:, :2] - achieved_goal[:, :2]), -1)) * 0.9
            reward_velocity = np.exp(-np.sum(np.square(desired_goal[:, 2:4] - achieved_goal[:, 2:4]), -1)) * 0.9
            reward_height = np.exp(-np.sum(np.square(desired_goal[:, 4] - achieved_goal[:, 4]), -1)) * 0.1
            reward_yaw = np.exp(-np.sum(np.square(desired_goal[:, 5] - achieved_goal[:, 5]), -1)) * 0.1
        else:
            reward_position = np.exp(-np.sum(np.square(achieved_goal[:2] - desired_goal[:2]), -1)) * 0.9
            reward_velocity = np.exp(-np.sum(np.square(achieved_goal[2:4] - desired_goal[2:4]), -1)) * 0.9
            reward_height = np.exp(-np.sum(np.square(achieved_goal[4] - desired_goal[4]), -1)) * 0.1
            reward_yaw = np.exp(-np.sum(np.square(achieved_goal[5] - desired_goal[5]), -1)) * 0.1
        if isinstance(info, np.ndarray):
            out = {}
            for item in info:
                for key in item.keys():
                    if key in out.keys():
                        out[key].append(item[key])
                    else:
                        out[key] = [item[key]]
            out = {
                key : np.array(out[key], dtype = np.float32) \
                    for key in out.keys()
            }
            info = out
        reward = info['reward_contact'] + info['reward_velocity'] + \
            info['reward_survive'] + info['penalty'] + \
            info['reward_position'] + info['reward_ctrl'] + \
            info['reward_height'] + info['reward_yaw']
        return reward

class AntEnvV4(AntEnvV1):
    def __init__(self, path = '/home/ubuntu/AntController/src/simulations/gym/ant.xml'):
        super(AntEnvV4, self).__init__(path)

    def _set_action_space(self):
        self.params = params
        self.kc = 0.1
        self.kd = 0.997
        self.a = np.array([0.5235, 0.9599, 0.5235, 0.9599, 0.5235, 0.9599, 0.5235, 0.9599], dtype = np.float32)
        self.b = np.array([0.0, 0.8289, 0.0, -0.8289, 0.0, -0.8289, 0.0, 0.8289], dtype = np.float32)
        self._step_num = 0
        self._update = 0
        self.q = np.zeros((4,), dtype = np.float32)
        self.pos = self.init_qpos[:3]
        self.vel = np.zeros((3,), dtype = np.float32)
        self.omega = np.zeros((3,), dtype = np.float32)
        self.acc = np.zeros((3,), dtype = np.float32)
        self.w = np.array([0.324, 0.264, 0.174, 0.154, 0.084], dtype = np.float32)
        self.goal_keys = ['command', 'ctrl', 'position', 'orientation']
        self.commands = self._create_command_lst()
        self.command = random.choice(self.commands)
        self.last_action = self.init_qpos[-8:]
        self.desired_goal = np.concatenate([
            self.command,
            np.zeros((8,), dtype = np.float32),
            np.zeros((3,), dtype = np.float32),
            np.zeros((4,), dtype = np.float32),
        ], -1)
        self.achieved_goal = np.concatenate([
            np.zeros(shape = self.command.shape, dtype = self.command.dtype),
            np.zeros((8,), dtype = np.float32),
            np.zeros((3, ), dtype = np.float32),
            np.zeros((4,), dtype =np.float32)
        ], -1)
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        low_osc = -np.ones((2 * self.params['units_osc'],), dtype = np.float32)
        high_osc = np.ones((2 * self.params['units_osc'],), dtype = np.float32)
        low = np.concatenate([low, low_osc], -1)
        high = np.concatenate([high, high_osc], -1)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def reset(self):
        self.command = random.choice(self.commands)
        self.omega = []
        self.acc = []
        self._update = 0
        """
        self.omega.append(self.sim.data.sensordata[:3])
        self.acc.append(self.sim.data.sensordata[3:6])
        self.q = np.zeros((4,), dtype = np.float32)
        self.pos = self.init_qpos[:3]
        self.vel = np.zeros((3,), dtype = np.float32)
        """
        self.pos = self.sim.data.qpos[:3]
        self.q = self.sim.data.qpos[3:7]
        self.vel = self.sim.data.qvel[:3]
        self.omega = self.sim.data.qvel[3:6]
        self.acc = self.sim.data.qacc[:3]
        self.last_action = self.init_qpos[-8:]
        self.last_torque = np.zeros_like(self.sim.data.actuator_force / 150)
        self.desired_goal = np.concatenate([
            self.command,
            np.zeros((8,), dtype = np.float32),
            self.pos,
            np.zeros((4,), dtype = np.float32)
        ], -1)
        self.achieved_goal = np.concatenate([
            np.concatenate([
                self.vel,
                self.omega
            ], -1),
            np.zeros((8,), dtype = np.float32),
            self.pos,
            self.q
        ], -1)
        self.osc = np.concatenate([
            np.zeros((self.params['units_osc'],), dtype = np.float32),
            np.ones((self.params['units_osc'],), dtype = np.float32),
        ], -1)
        self.sim.reset()
        self.ob = self.reset_model()
        return {
            'observation' : self.ob,
            'desired_goal' : self.desired_goal,
            'achieved_goal' : self.achieved_goal
        }

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_goal_error(self):
        err = self.achieved_goal - self.desired_goal
        return err

    def _get_vec_from_dct(self, dct):
        lst = []
        for key in self.goal_keys:
            lst.append(dct[key])
        return np.concatenate(lst, -1)

    def _get_obs(self):
        acc = self.sim.data.qacc[:3]
        return np.concatenate([self.osc, acc, self.last_action], -1)

    def _create_command_lst(self):
        self.commands = []
        delta = 0.01
        forward_vel = np.arange(-0.4, 0.4, delta).tolist()
        lateral_vel = np.arange(-0.4, 0.4, delta).tolist()
        yaw = np.arange(-0.3, 0.3, delta).tolist()
        for xvel in forward_vel:
            for yvel in lateral_vel:
                for yw in yaw:
                    self.commands.append(np.array([
                        xvel, yvel, 0, 0, 0, yw
                    ], dtype = np.float32))
        return self.commands

    def perform_action(self, action):
        penalty = 0.0
        ac = action[:self.params['action_dim']]
        #print(np.round(ac, 4))
        if np.isnan(action[:self.params['action_dim']]).any():
            print('[DDPG] Action NaN')
            ac = np.zeros(shape = ac.shape, dtype = action.dtype)
            penalty = -5.0
        #print(action[:self.params['action_dim']])
        self.desired_goal[6:14] = (ac - self.last_action) / self.dt
        self.do_simulation(ac, self.frame_skip)
        self.last_action = ac
        self.osc = action[self.params['action_dim']:]
        return penalty

    def K(self, x):
        return np.exp(-x)

    def step(self, action):
        if self._step_num % 100 == 0 or self._update == 0:
            scale = np.ones((self.params['motion_state_size'],), dtype = np.float32)
            if self._update == 0:
                scale[1:] = 0.0
            self._update += 1
            self.desired_goal[:6] = self.command * scale
        self._step_num += 1
        #print(action)
        posbefore = self.get_body_com("torso").copy()
        jointposbefore = self.sim.data.qpos[-8:].copy()
        penalty = self.perform_action(action)
        posafter = self.get_body_com("torso").copy()
        jointposafter = self.sim.data.qpos[-8:].copy()
        info = {}
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        #self.render()
        self.pos = posafter
        self.q = self.sim.data.qpos[3:7].copy()
        self.vel = (posafter - posbefore) / self.dt
        self.omega = self.sim.data.sensordata[:3]
        self.acc = self.sim.data.qacc[:3].copy()
        self.joint_vel = (jointposafter - jointposbefore)/self.dt
        self.achieved_goal = np.concatenate([
            np.concatenate([
                self.vel,
                self.omega
            ], -1),
            self.joint_vel,
            self.pos,
            self.q
        ], -1)

        self.ob = self._get_obs()
        err = self._get_goal_error()
        geod_dist = 1 - np.square(np.sum(self.achieved_goal[17:21] * self.desired_goal[17:21], -1))
        info['reward_velocity'] = self.K(np.sum(np.abs(err[0]), -1)) + self.K(np.sum(np.abs(err[1]), -1)) + self.K(np.sum(np.abs(err[2]), -1))
        info['reward_rotation'] = self.K(np.sum(np.abs(err[3]), -1)) + self.K(np.sum(np.abs(err[4]), -1)) + self.K(np.sum(np.abs(err[5]), -1))
        info['reward_torque'] = -0.005 * self.dt * self.kc * np.square(np.linalg.norm(self.sim.data.actuator_force.copy() / 150))
        info['reward_ctrl'] = -0.03 * self.dt * self.kc * np.square(np.linalg.norm(self.achieved_goal[6:14]))
        info['reward_position'] = -0.1 * self.dt * np.square(np.linalg.norm(err[14:17]))
        info['reward_orientation'] = -0.4 * self.dt * self.kc * np.square(geod_dist)
        info['reward_motion'] = np.linalg.norm(self.vel) if np.linalg.norm(self.vel) < 0.6 else -0.1
        info['reward_contact'] = -self.kc * 2.0 * self.dt * np.square(np.linalg.norm(np.clip(self.sim.data.cfrc_ext, -1, 1).flat))
        """
        reward = info['reward_velocity'] + info['reward_rotation'] + \
            info['reward_ctrl'] + info['reward_position'] + \
            info['reward_orientation'] + info['reward_motion'] + \
            info['reward_contact'] + info['reward_torque']
        """
        reward = info['reward_position']
        info['reward'] = reward
        self.desired_goal[14:17] += self.desired_goal[:3] * self.dt
        self.desired_goal[17:21] = skin.quat.unit_q(skin.quat.deg2quat(
            skin.quat.quat2deg(
                self.desired_goal[17:21]
            ) + self.desired_goal[3:6] * self.dt
        ))
        self.kc = self.kc ** self.kd
        #print({'observation' : self.ob, 'desired_goal' : self.desired_goal, 'achieved_goal' : self.achieved_goal})
        return {'observation' : self.ob, 'desired_goal' : self.desired_goal, 'achieved_goal' : self.achieved_goal}, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        out = {}
        for item in info:
            for key in item.keys():
                if key in out.keys():
                    out[key].append(item[key])
                else:
                    out[key] = [item[key]]
        info = {
            key : np.array(out[key], dtype = np.float32) \
                for key in out.keys()
        }
        err = achieved_goal - desired_goal
        geod_dist = 1 - np.square(np.sum(achieved_goal[:, 17:21] * desired_goal[:, 17:21], -1))
        info['reward_velocity'] = self.K(np.sum(np.abs(err[:, 0]), -1)) + self.K(np.sum(np.abs(err[:, 1]), -1)) + self.K(np.sum(np.abs(err[:, 2]), -1))
        info['reward_rotation'] = self.K(np.sum(np.abs(err[:, 3]), -1)) + self.K(np.sum(np.abs(err[:, 4]), -1)) + self.K(np.sum(np.abs(err[:, 5]), -1))
        info['reward_ctrl'] = -0.03 * self.dt * self.kc * np.square(np.linalg.norm(achieved_goal[:, 6:14]))
        info['reward_position'] = -0.1 * self.dt * np.square(np.linalg.norm(err[:, 14:17], -1)) + np.linalg.norm(achieved_goal[:, 14:16], -1)
        info['reward_orientation'] = -0.4 * self.dt * self.kc * np.square(geod_dist)
        reward = info['reward_velocity'] + info['reward_rotation'] + \
            info['reward_ctrl'] + info['reward_position'] + \
            info['reward_orientation'] + info['reward_motion'] + \
            info['reward_contact'] + info['reward_torque']
        return reward

class AntEnvV5(AntEnvV4):
    def __init__(self, path = '/home/ubuntu/AntController/src/simulations/gym/ant.xml'):
        super(AntEnvV5, self).__init__(path)

    def _set_action_space(self):
        self.kc = 0.075
        self.kd = 0.999
        self._step_num = 0
        self._update = 0
        self.params = params
        self.q = np.zeros((4,), dtype = np.float32)
        self.pos = self.init_qpos[:3]
        self.vel = np.zeros((3,), dtype = np.float32)
        self.omega = np.zeros((3,), dtype = np.float32)
        self.acc = np.zeros((3,), dtype = np.float32)
        self.w = np.array([0.3,0.24,0.15,0.13,0.06,0.06,0.06], dtype = np.float32)
        self.goal_keys = ['command', 'ctrl', 'position', 'orientation']
        self.commands = self._create_command_lst()
        self.command = random.choice(self.commands)
        self.last_action = self.init_qpos[-8:]
        self.last_torque = self.sim.data.actuator_force / 150
        self.desired_goal = np.concatenate([
            self.command,
            np.zeros((8,), dtype = np.float32),
            np.zeros((3,), dtype = np.float32),
            np.zeros((4,), dtype = np.float32),
        ], -1)
        self.achieved_goal = np.concatenate([
            np.zeros(shape = self.command.shape, dtype = self.command.dtype),
            np.zeros((8,), dtype = np.float32),
            np.zeros((3, ), dtype = np.float32),
            np.zeros((4,), dtype =np.float32)
        ], -1)
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.sensordata[3:6],
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            self.sim.data.actuator_force.copy() / 150,
            self.last_action
        ], -1)

    def perform_action(self, action):
        penalty = 0.0
        #print(np.round(action, 4))
        if np.isnan(action).any():
            print('[DDPG] Action NaN')
            action = np.zeros(shape = action.shape, dtype = action.dtype)
            penalty = -5.0
        #print(action[:self.params['action_dim']])
        self.desired_goal[6:14] = (action - self.last_action) / self.dt
        self.do_simulation(action, self.frame_skip)
        self.last_action = action
        return penalty

class AntEnvV6(AntEnvV5):
    def reset(self):
        self.command = random.choice(self.commands)
        self.omega = []
        self.acc = []
        self._update = 0
        self.pos = self.sim.data.qpos[:3]
        self.q = self.sim.data.qpos[3:7]
        self.vel = self.sim.data.qvel[:3]
        self.omega = self.sim.data.qvel[3:6]
        self.acc = self.sim.data.qacc[:3]
        self.last_action = self.init_qpos[-8:]
        self.last_torque = np.zeros_like(self.sim.data.actuator_force / 150)
        self.desired_goal = np.concatenate([
            self.command,
            np.zeros((8,), dtype = np.float32),
            self.pos,
            np.zeros((4,), dtype = np.float32)
        ], -1)
        self.achieved_goal = np.concatenate([
            np.concatenate([
                self.vel,
                self.omega
            ], -1),
            np.zeros((8,), dtype = np.float32),
            self.pos,
            self.q
        ], -1)
        self.osc = np.concatenate([
            np.zeros((self.params['units_osc'],), dtype = np.float32),
            np.ones((self.params['units_osc'],), dtype = np.float32),
        ], -1)
        self.sim.reset()
        self.ob = self.reset_model()
        return {
            'observation' : self.ob,
            'desired_goal' : self.desired_goal,
            'achieved_goal' : self.achieved_goal
        }

    def step(self, action):
        if self._step_num % 100 == 0 or self._update == 0:
            scale = np.ones((self.params['motion_state_size'],), dtype = np.float32)
            if self._update == 0:
                scale[1:] = 0.0
            self._update += 1
            self.desired_goal[:6] = self.command * scale
        self._step_num += 1
        #print(action)
        posbefore = self.get_body_com("torso").copy()
        penalty = self.perform_action(action)
        posafter = self.get_body_com("torso").copy()
        info = {}
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        #self.render()
        self.pos = posafter
        self.q = self.sim.data.qpos[3:7].copy()
        self.vel = (posafter - posbefore) / self.dt
        self.omega = self.sim.data.sensordata[:3]
        self.acc = self.sim.data.qacc[:3].copy()
        self.achieved_goal = np.concatenate([
            np.concatenate([
                self.vel,
                self.omega
            ], -1),
            self.sim.data.qpos[-8:].copy(),
            self.pos,
            self.q
        ], -1)

        self.ob = self._get_obs()
        err = self._get_goal_error()
        geod_dist = 1 - np.square(np.sum(self.achieved_goal[17:21] * self.desired_goal[17:21], -1))
        info['reward_velocity'] = np.exp(-np.square(np.linalg.norm(err[:3]))) * self.w[0]
        info['reward_rotation']= np.exp(-np.square(np.linalg.norm(err[3:6]))) * self.w[1]
        info['reward_ctrl'] = self.kc * np.exp(-np.square(np.linalg.norm(err[6:14]))) * self.w[2]
        info['reward_position'] = self.kc * np.exp(-np.square(np.linalg.norm(err[14:17]))) * self.w[3]
        info['reward_orientation'] = self.kc * np.exp(-np.square(geod_dist)) * self.w[4]
        info['reward_motion'] = np.linalg.norm(self.vel) if np.linalg.norm(self.vel) < 0.6 else -0.1
        info['reward_contact'] = self.kc * np.exp(-np.square(np.linalg.norm(np.clip(self.sim.data.cfrc_ext, -1, 1).flat))) * self.w[6]
        info['reward_torque'] = self.kc * np.exp(-np.square(np.linalg.norm(self.sim.data.actuator_force / 150))) * self.w[5]
        reward = info['reward_velocity'] + info['reward_rotation'] + \
            info['reward_ctrl'] + info['reward_position'] + \
            info['reward_orientation'] + info['reward_motion'] + \
            info['reward_contact'] + info['reward_torque']
        info['reward'] = reward
        self.desired_goal[14:17] += self.desired_goal[:3] * self.dt
        self.desired_goal[17:21] = skin.quat.unit_q(skin.quat.deg2quat(
            skin.quat.quat2deg(
                self.desired_goal[17:21]
            ) + self.desired_goal[3:6] * self.dt
        ))
        self.kc = self.kc ** self.kd
        #print({'observation' : self.ob, 'desired_goal' : self.desired_goal, 'achieved_goal' : self.achieved_goal})
        return {'observation' : self.ob, 'desired_goal' : self.desired_goal, 'achieved_goal' : self.achieved_goal}, reward, done, info

    def perform_action(self, action):
        penalty = 0.0
        #print(np.round(ac, 4))
        if np.isnan(action[:self.params['action_dim']]).any():
            print('[DDPG] Action NaN')
            action = np.zeros(shape = action.shape, dtype = action.dtype)
            penalty = -5.0
        #print(action[:self.params['action_dim']])
        self.desired_goal[6:14] = action
        self.do_simulation(action, self.frame_skip)
        return penalty

    def compute_reward(self, achieved_goal, desired_goal, info):
        out = {}
        for item in info:
            for key in item.keys():
                if key in out.keys():
                    out[key].append(item[key])
                else:
                    out[key] = [item[key]]
        info = {
            key : np.array(out[key], dtype = np.float32) \
                for key in out.keys()
        }
        err = achieved_goal - desired_goal
        geod_dist = 1 - np.square(np.sum(achieved_goal[:, 17:21] * desired_goal[:, 17:21], -1))
        info['reward_velocity'] = np.exp(-np.square(np.linalg.norm(err[:,:3], axis = -1))) * self.w[0]
        info['reward_rotation'] = np.exp(-np.square(np.linalg.norm(err[:, 3:6], axis = -1))) * self.w[1]
        info['reward_ctrl'] = np.exp(-np.square(np.linalg.norm(err[:,6:14], axis = -1))) * self.w[2]
        info['reward_position'] = np.exp(-np.square(np.linalg.norm(err[:,14:17], axis = -1))) * self.w[3]
        info['reward_orientation'] = np.exp(-np.square(geod_dist)) * self.w[4]
        reward = info['reward_velocity'] + info['reward_rotation'] + \
            info['reward_ctrl'] + info['reward_position'] + \
            info['reward_orientation'] + info['reward_motion'] + \
            info['reward_contact'] + info['reward_torque']
        return reward
