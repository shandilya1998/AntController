import torch
#torch.cuda.current_device()
from collections import OrderedDict
import numpy as np
from rl.torch.util import *
from rl.torch.constants import params

USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
DEVICE = 'cpu'
if USE_CUDA:
    DEVICE = 'gpu'

def complex_relu(input):
    size = input.shape[-1]//2
    x, y = torch.split(input, size, -1)
    out = torch.cat(
        [
            torch.nn.functional.relu(x),
            torch.nn.functional.relu(y)
        ], - 1
    )
    return out

def complex_elu(input):
    size = input.shape[-1]//2
    x, y = torch.split(input, size, -1)
    out = torch.cat(
        [
            torch.nn.functional.elu(x),
            torch.nn.functional.elu(y)
        ], - 1
    )
    return out



def complex_tanh(input):
    size = input.shape[-1]//2
    x, y = torch.split(input, size, -1)
    denominator = torch.cosh(2*x) + torch.cos(2*y)
    x = torch.sin(2*x) / denominator
    y = torch.sinh(2*y) / denominator
    out = torch.cat([x, y], -1)
    return out

def apply_complex(fr, fi, input, dtype = torch.float32):
    size = input.shape[-1]//2
    x, y = torch.split(input, size, -1)
    out = torch.cat(
        [
            fr(x) - fi(y),
            fr(y) + fi(x)
        ], -1
    )
    return out

class ComplexReLU(torch.nn.Module):

     def forward(self,input):
         return complex_relu(input)

class ComplexELU(torch.nn.Module):

     def forward(self,input):
         return complex_elu(input)

class ComplexTanh(torch.nn.Module):

    def forward(self, input):
        return complex_tanh(input)

class ComplexPReLU(torch.nn.Module):
    def __init__(self):
        super(ComplexPReLU, self).__init__()
        self.prelu_r = torch.nn.PReLU()
        self.prelu_i = torch.nn.PReLU()

    def forward(self, input):
        size = input.shape[-1]//2
        x, y = torch.split(input, size, -1)
        out = torch.cat([self.prelu_r(x), self.prelu_i(y)], -1)
        return out

class ComplexLinear(torch.nn.Module):

    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.fc_r = torch.nn.Linear(in_features, out_features)
        self.fc_i = torch.nn.Linear(in_features, out_features)

    def forward(self, input):
        return apply_complex(self.fc_r, self.fc_i, input)

class ActorV0(torch.nn.Module):
    def __init__(self):
        super(ActorV0, self).__init__()
        self.params = params
        robot_state_enc_seq = OrderedDict()
        input_size = self.params['robot_state_size']
        for i, units in enumerate(self.params['units_robot_state']):
            robot_state_enc_seq['fc{i}'.format(i = i)] = torch.nn.Linear(
                input_size,
                units
            )
            robot_state_enc_seq['ac{i}'.format(i = i)] = torch.nn.ELU()
            input_size = units
        self.out_rs = torch.nn.Tanh()
        self.robot_state_enc = torch.nn.Sequential(robot_state_enc_seq)

        complex_seq = OrderedDict()
        input_size = self.params['units_osc']
        for i, units in enumerate(self.params['units_output_mlp']):
            complex_seq['fc{i}'.format(i = i)] = ComplexLinear(
                input_size,
                units
            )
            complex_seq['ac{i}'.format(i = i)] =  ComplexELU()
            input_size = units

        complex_seq['out_fc'] = ComplexLinear(
            input_size,
            self.params['action_dim']
        )
        complex_seq['out_ac'] = ComplexELU()
        self.complex_mlp = torch.nn.Sequential(
            complex_seq
        )
        self.action_dim = self.params['action_dim']

    def forward(self, robot_state):
        z_r = self.out_rs(self.robot_state_enc(robot_state))
        z_i = torch.zeros_like(z_r)
        z = self.complex_mlp(torch.cat([z_r, z_i], -1))
        x, y = torch.split(z, self.action_dim, -1)
        phi = torch.atan2(y, x)
        out = self.params['action_scale_factor'] * torch.sin(phi)
        return out

class ActorV1(torch.nn.Module):
    def __init__(self):
        super(ActorV1, self).__init__()
        self.params = params
        robot_state_enc_seq = OrderedDict()
        input_size = 2 * params['motion_state_size']
        output_size_motion_state_enc = None
        motion_seq = []
        for i, units in enumerate(params['units_motion_state']):
            motion_seq.append(torch.nn.Linear(
                input_size,
                units
            ))
            motion_seq.append(torch.nn.ELU())
            input_size = units
            output_size_motion_state_enc = units
        motion_seq.extend([
            torch.nn.Linear(
                output_size_motion_state_enc,
                self.params['units_osc']
            ),
            torch.nn.ELU()
        ])
        self.desired_motion_seq = torch.nn.Sequential(
            *motion_seq
        )
        input_size = self.params['robot_state_size']
        for i, units in enumerate(self.params['units_robot_state']):
            robot_state_enc_seq['fc{i}'.format(i = i)] = torch.nn.Linear(
                input_size,
                units
            )
            robot_state_enc_seq['ac{i}'.format(i = i)] = torch.nn.ELU()
            input_size = units
        self.out_rs = torch.nn.Tanh()
        self.robot_state_enc = torch.nn.Sequential(robot_state_enc_seq)

        complex_seq = OrderedDict()
        input_size = self.params['units_osc']
        for i, units in enumerate(self.params['units_output_mlp']):
            complex_seq['fc{i}'.format(i = i)] = ComplexLinear(
                input_size,
                units
            )
            complex_seq['ac{i}'.format(i = i)] =  ComplexELU()
            input_size = units

        complex_seq['out_fc'] = ComplexLinear(
            input_size,
            self.params['action_dim']
        )
        self.action_dim = self.params['action_dim']
        complex_seq['out_ac'] = ComplexELU()
        self.complex_mlp = torch.nn.Sequential(
            complex_seq
        )


    def forward(self, observation):
        robot_state, motion_state = observation
        z_r = self.out_rs(self.robot_state_enc(robot_state)) + \
            self.desired_motion_seq(motion_state)
        z_i = torch.zeros_like(z_r)
        z = torch.cat([z_r, z_i], -1)
        z = self.complex_mlp(z)
        x, y = torch.split(z, self.action_dim, -1)
        phi = torch.atan2(y, x)
        out = self.params['action_scale_factor'] * torch.sin(phi)
        return out

class ParamNet(torch.nn.Module):
    def __init__(self,
        params,
    ):
        super(ParamNet, self).__init__()
        self.params = params
        motion_seq = []
        input_size =  params['motion_state_size']
        output_size_motion_state_enc = None
        for i, units in enumerate(params['units_motion_state']):
            motion_seq.append(torch.nn.Linear(
                input_size,
                units
            ))
            motion_seq.append(torch.nn.ELU())
            input_size = units
            output_size_motion_state_enc = units
        self.motion_state_enc = torch.nn.Sequential(
            *motion_seq
        )

        omega_seq = []
        for i, units in enumerate(params['units_omega']):
            omega_seq.append(torch.nn.Linear(
                input_size,
                units
            ))
            omega_seq.append(torch.nn.ELU())
            input_size = units

        omega_seq.append(torch.nn.Linear(
            input_size,
            params['units_osc']
        ))
        omega_seq.append(torch.nn.ReLU())

        self.omega_dense_seq = torch.nn.Sequential(
            *omega_seq
        )

        mu_seq = []
        for i, units in enumerate(params['units_mu']):
            mu_seq.append(torch.nn.Linear(
                input_size,
                units
            ))
            mu_seq.append(torch.nn.ELU())
            input_size = units

        mu_seq.append(torch.nn.Linear(
            input_size,
            params['units_osc']
        ))
        self.out_relu = torch.nn.ReLU()

        self.mu_dense_seq = torch.nn.Sequential(
            *mu_seq
        )

    def forward(self, desired_motion):
        x = self.motion_state_enc(desired_motion)
        omega = self.omega_dense_seq(x)
        mu = self.out_relu(torch.cos(self.mu_dense_seq(x)))
        return omega, mu

class Hopf(torch.nn.Module):
    def __init__(self, params):
        super(Hopf, self).__init__()
        self.params = params
        self.dt = torch.Tensor([self.params['dt']]).type(FLOAT)
        self.arange = torch.arange(0, self.params['units_osc'], 1.0).type(FLOAT)

    def forward(self, z, omega, mu):
        units_osc = z.shape[-1]
        x, y = torch.split(z, units_osc // 2, -1)
        r = torch.sqrt(x ** 2 + y ** 2)
        phi = torch.atan2(y,x)
        delta_phi = self.dt * omega
        phi = phi + delta_phi
        r = r + self.dt * (mu - r ** 2) * r
        z = torch.cat([x, y], -1)
        return z

class ActorV2(torch.nn.Module):
    def __init__(self):
        super(ActorV2, self).__init__()
        self.params = params

        self.param_net = ParamNet(self.params)
        self.hopf = Hopf(self.params)

        robot_state_enc_seq = OrderedDict()
        input_size = self.params['robot_state_size']
        for i, units in enumerate(self.params['units_robot_state']):
            robot_state_enc_seq['fc{i}'.format(i = i)] = torch.nn.Linear(
                input_size,
                units
            )
            robot_state_enc_seq['ac{i}'.format(i = i)] = torch.nn.PReLU()
            input_size = units
        self.robot_state_enc = torch.nn.Sequential(robot_state_enc_seq)

        self.action_dim = self.params['action_dim']

    def forward(self, features):
        state, desired_goal, z = features
        omega, mu = self.param_net(desired_goal)
        z = self.hopf(z, omega, mu)
        phase = self.robot_state_enc(state)
        z_i = torch.sin(phase)
        z_r = torch.cos(phase)
        out = (1 - self.params['scale_factor_1']) * z + self.params['scale_factor_1']*torch.cat([z_r, z_i], -1)
        x, y = torch.split(out, self.action_dim, -1)
        out = x * self.params['action_scale_factor']
        return torch.cat([x, z], -1)

class ActorV3(torch.nn.Module):
    def __init__(self):
        super(ActorV3, self).__init__()
        self.params = params

        motion_seq = []
        input_size =  params['motion_state_size']
        output_size_motion_state_enc = None
        for i, units in enumerate(params['units_motion_state']):
            motion_seq.append(torch.nn.Linear(
                input_size,
                units
            ))
            motion_seq.append(torch.nn.PReLU())
            input_size = units
            output_size_motion_state_enc = units
        motion_seq.append(torch.nn.Linear(
            input_size,
            params['units_osc']
        ))
        motion_seq.append(torch.nn.PReLU())

        self.motion_state_enc = torch.nn.Sequential(
            *motion_seq
        )

        robot_state_enc_seq = OrderedDict()
        input_size = self.params['robot_state_size']
        for i, units in enumerate(self.params['units_robot_state']):
            robot_state_enc_seq['fc{i}'.format(i = i)] = torch.nn.Linear(
                input_size,
                units
            )
            robot_state_enc_seq['ac{i}'.format(i = i)] = torch.nn.PReLU()
            input_size = units
        self.robot_state_enc = torch.nn.Sequential(robot_state_enc_seq)
        self.action_dim = self.params['action_dim']

    def forward(self, features):
        state, desired_goal = features
        x1 = torch.cos(self.motion_state_enc(desired_goal))
        x2 = torch.cos(self.robot_state_enc(state))
        out = (1 - self.params['scale_factor_1'])*x1 + x2*self.params['scale_factor_1']
        return self.params['action_scale_factor'] * out
