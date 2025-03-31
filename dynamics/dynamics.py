from abc import ABC, abstractmethod
from utils import diff_operators, quaternion

import math
import torch
import numpy as np
from multiprocessing import Pool
import torch.nn as nn
import scipy.io as spio
# during training, states will be sampled uniformly by each state dimension from the model-unit -1 to 1 range (for training stability),
# which may or may not correspond to proper test ranges
# note that coord refers to [time, *state], and input refers to whatever is fed directly to the model (often [time, *state, params])
# in the future, code will need to be fixed to correctly handle parametrized models


class Dynamics(ABC):
    def __init__(self,
                 name: str, loss_type: str, set_mode: str,
                 state_dim: int, input_dim: int,
                 control_dim: int, disturbance_dim: int,
                 state_mean: list, state_var: list,
                 value_mean: float, value_var: float, value_normto: float,
                 deepReach_model: bool):
        self.name= name 
        self.loss_type = loss_type
        self.set_mode = set_mode
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.control_dim = control_dim
        self.disturbance_dim = disturbance_dim
        self.state_mean = torch.tensor(state_mean)
        self.state_var = torch.tensor(state_var)
        self.value_mean = value_mean
        self.value_var = value_var
        self.value_normto = value_normto
        self.deepReach_model = deepReach_model

        assert self.loss_type in [
            'brt_hjivi', 'brat_hjivi'], f'loss type {self.loss_type} not recognized'
        if self.loss_type == 'brat_hjivi':
            assert callable(self.reach_fn) and callable(self.avoid_fn)
        assert self.set_mode in [
            'reach', 'avoid', 'reach_avoid'], f'set mode {self.set_mode} not recognized'
        for state_descriptor in [self.state_mean, self.state_var]:
            assert len(state_descriptor) == self.state_dim, 'state descriptor dimension does not equal state dimension, ' + \
                str(len(state_descriptor)) + ' != ' + str(self.state_dim)

    # ALL METHODS ARE BATCH COMPATIBLE

    # set deepreach model. choices: "vanilla" (vanilla DeepReach V=NN(x,t)), diff (diff model V=NN(x,t) + l(x)), exact ( V=NN(x,t) + l(x))
    def set_model(self, deepreach_model):
        self.deepReach_model = deepreach_model

    # MODEL-UNIT CONVERSIONS 
    # convert model input (normalized) to real coord
    def input_to_coord(self, input):
        coord = input.clone()
        coord[..., 1:] = (input[..., 1:] * self.state_var.to(device=input.device)
                          ) + self.state_mean.to(device=input.device)
        return coord

    # convert real coord to model input
    def coord_to_input(self, coord):
        input = coord*1.0
        input[..., 1:] = (coord[..., 1:] - self.state_mean.to(device=coord.device)
                          ) / self.state_var.to(device=coord.device)
        return input

    # convert model io to real value
    def io_to_value(self, input, output):
        if self.deepReach_model == 'diff':
            return (output * self.value_var / self.value_normto) + self.boundary_fn(self.input_to_coord(input)[..., 1:])
        elif self.deepReach_model == 'exact':
            return (output * input[..., 0] * self.value_var / self.value_normto) + self.boundary_fn(self.input_to_coord(input)[..., 1:])
        elif self.deepReach_model == 'exact_diff':
            # V(x,t)= l(x) + NN(x,t) - NN(x,0)
            # print(input.shape, output.shape)
            output0 = output[0].squeeze(dim=-1)
            output1 = output[1].squeeze(dim=-1)
            return (output0 - output1) * self.value_var / self.value_normto + self.boundary_fn(self.input_to_coord(input[0].detach())[..., 1:])
        else:
            return (output * self.value_var / self.value_normto) + self.value_mean

    # convert model io to real dv
    def io_to_dv(self, input, output):
        if self.deepReach_model == 'exact_diff':

            dodi1 = diff_operators.jacobian(
                output[0], input[0])[0].squeeze(dim=-2)
            dodi2 = diff_operators.jacobian(
                output[1], input[1])[0].squeeze(dim=-2)

            dvdt = (self.value_var / self.value_normto) * dodi1[..., 0]

            dvds_term1 = (self.value_var / self.value_normto /
                          self.state_var.to(device=dodi1.device)) * (dodi1[..., 1:]-dodi2[..., 1:])

            state = self.input_to_coord(input[0])[..., 1:]
            dvds_term2 = diff_operators.jacobian(self.boundary_fn(
                state).unsqueeze(dim=-1), state)[0].squeeze(dim=-2)
            dvds = dvds_term1 + dvds_term2
            return torch.cat((dvdt.unsqueeze(dim=-1), dvds), dim=-1)

        dodi = diff_operators.jacobian(
            output.unsqueeze(dim=-1), input)[0].squeeze(dim=-2)

        if self.deepReach_model == 'diff':
            dvdt = (self.value_var / self.value_normto) * dodi[..., 0]

            dvds_term1 = (self.value_var / self.value_normto /
                          self.state_var.to(device=dodi.device)) * dodi[..., 1:]
            state = self.input_to_coord(input)[..., 1:]
            dvds_term2 = diff_operators.jacobian(self.boundary_fn(
                state).unsqueeze(dim=-1), state)[0].squeeze(dim=-2)
            dvds = dvds_term1 + dvds_term2

        elif self.deepReach_model == 'exact':

            dvdt = (self.value_var / self.value_normto) * \
                (input[..., 0]*dodi[..., 0] + output)

            dvds_term1 = (self.value_var / self.value_normto /
                          self.state_var.to(device=dodi.device)) * dodi[..., 1:] * input[..., 0].unsqueeze(-1)
            state = self.input_to_coord(input)[..., 1:]
            dvds_term2 = diff_operators.jacobian(self.boundary_fn(
                state).unsqueeze(dim=-1), state)[0].squeeze(dim=-2)
            dvds = dvds_term1 + dvds_term2
        else:
            dvdt = (self.value_var / self.value_normto) * dodi[..., 0]
            dvds = (self.value_var / self.value_normto /
                    self.state_var.to(device=dodi.device)) * dodi[..., 1:]

        return torch.cat((dvdt.unsqueeze(dim=-1), dvds), dim=-1)

    # convert model io to real dv
    def io_to_2nd_derivative(self, input, output):
        hes = diff_operators.batchHessian(
            output.unsqueeze(dim=-1), input)[0].squeeze(dim=-2)

        if self.deepReach_model == 'diff':
            vis_term1 = (self.value_var / self.value_normto /
                         self.state_var.to(device=hes.device))**2 * hes[..., 1:]
            state = self.input_to_coord(input)[..., 1:]
            vis_term2 = diff_operators.batchHessian(self.boundary_fn(
                state).unsqueeze(dim=-1), state)[0].squeeze(dim=-2)
            hes = vis_term1 + vis_term2

        else:
            hes = (self.value_var / self.value_normto /
                   self.state_var.to(device=hes.device))**2 * hes[..., 1:]

        return hes

    def clamp_control(self, state, control):
        return control
    
    def clamp_state_input(self, state_input):
        return state_input
    
    def clamp_verification_state(self, state):
        return state
    # ALL FOLLOWING METHODS USE REAL UNITS
    @abstractmethod
    def periodic_transform_fn(self, input):
        raise NotImplementedError
    
    @abstractmethod
    def state_test_range(self):
        raise NotImplementedError

    @abstractmethod
    def equivalent_wrapped_state(self, state):
        raise NotImplementedError

    @abstractmethod
    def dsdt(self, state, control, disturbance):
        raise NotImplementedError

    @abstractmethod
    def boundary_fn(self, state):
        raise NotImplementedError

    @abstractmethod
    def sample_target_state(self, num_samples):
        raise NotImplementedError

    @abstractmethod
    def cost_fn(self, state_traj):
        raise NotImplementedError

    @abstractmethod
    def hamiltonian(self, state, dvds):
        raise NotImplementedError

    @abstractmethod
    def optimal_control(self, state, dvds):
        raise NotImplementedError

    @abstractmethod
    def optimal_disturbance(self, state, dvds):
        raise NotImplementedError

    @abstractmethod
    def plot_config(self):
        raise NotImplementedError

class VertDrone2D(Dynamics):
    def __init__(self):
        self.gravity = 9.8                             # g
        self.input_multiplier = 12.0   # K
        self.input_magnitude_max = 1.0     # u_max
        self.state_range_ = torch.tensor([[-4, 4],[-0.5, 3.5]]).cuda() # v, z, k
        self.control_range_ =torch.tensor([[-self.input_magnitude_max, self.input_magnitude_max]]).cuda()
        self.eps_var=torch.tensor([2]).cuda()
        self.control_init= torch.ones(1).cuda()*self.gravity/self.input_multiplier 


        state_mean_=(self.state_range_[:,0]+self.state_range_[:,1])/2.0
        state_var_=(self.state_range_[:,1]-self.state_range_[:,0])/2.0

        super().__init__(
            name='VertDrone2D', loss_type='brt_hjivi', set_mode='avoid',
            state_dim=2, input_dim=3, # input_dim of the NN = state_dim + 1 (time dim)
            control_dim=1, disturbance_dim=0,
            state_mean=state_mean_.cpu().tolist(),
            state_var=state_var_.cpu().tolist(),    
            value_mean=0.5, # we estimate the ground-truth value function to be within [-0.5, 1.5] w.r.t. the state_range_ we used
            value_var=1,    # Then value_mean = 0.5*(-0.5 + 1.5) and value_max = 0.5*(1.5 - -0.5)
            value_normto=0.02,  # Don't need any changes
            deepReach_model='exact',  # chioce ['vanilla', 'exact'],
        )

    def control_range(self, state):
        return [[-self.input_magnitude_max, self.input_magnitude_max]]

    def state_test_range(self):
        return self.state_range_.cpu().tolist()
    
    def state_verification_range(self):
        return self.state_range_.cpu().tolist() 
        # Here we verify the training results using the training range itself, we can verify on a smaller range for "stiff" systems

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        return wrapped_state 

    def periodic_transform_fn(self, input):
        return input.cuda()
    
    # ParameterizedVertDrone2D dynamics
    # \dot v = k*u - g
    # \dot z = v
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = self.input_multiplier * control[..., 0] - self.gravity
        dsdt[..., 1] = state[..., 0]
        return dsdt

    def boundary_fn(self, state):
        return -torch.abs(state[..., 1] - 1.5) + 1.5 # distance to ground (0m) and ceiling (3m)

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        return  torch.abs(self.input_multiplier *dvds[..., 0]) * self.input_magnitude_max \
            - dvds[..., 0] * self.gravity \
            + dvds[..., 1] * state[..., 0]

    def optimal_control(self, state, dvds):
        return torch.sign(dvds[..., 0])[..., None]

    def optimal_disturbance(self, state, dvds):
        return torch.tensor([0])

    def plot_config(self):
        return {
            'state_slices': [0, 1.5],
            'state_labels': ['v', 'z'],
            'x_axis_idx': 0, # which dim you want it to be the 
            'y_axis_idx': 1,
            'z_axis_idx': -1, # because there is only 2D
        }
    
class ParameterizedVertDrone2D(Dynamics):
    def __init__(self, gravity: float, input_multiplier: float, input_magnitude_max: float):
        self.gravity = gravity                             # g
        self.input_multiplier = input_multiplier   # k_max
        self.input_magnitude_max = input_magnitude_max     # u_max
        self.state_range_ = torch.tensor([[-4, 4],[-0.5, 3.5],[0, self.input_multiplier]]).cuda() # v, z, k
        self.control_range_ =torch.tensor([[-self.input_magnitude_max, self.input_magnitude_max]]).cuda()
        self.eps_var=torch.tensor([2]).cuda()
        self.control_init= torch.ones(1).cuda()*gravity/input_multiplier 


        state_mean_=(self.state_range_[:,0]+self.state_range_[:,1])/2.0
        state_var_=(self.state_range_[:,1]-self.state_range_[:,0])/2.0

        super().__init__(
            name='ParameterizedVertDrone2D', loss_type='brt_hjivi', set_mode='avoid',
            state_dim=3, input_dim=4, control_dim=1, disturbance_dim=0,
            state_mean=state_mean_.cpu().tolist(),
            state_var=state_var_.cpu().tolist(),    
            value_mean=0.5,
            value_var=1,
            value_normto=0.02,
            deepReach_model='exact',  # chioce ['vanilla', 'exact'],
        )

    def control_range(self, state):
        return [[-self.input_magnitude_max, self.input_magnitude_max]]

    def state_test_range(self):
        return self.state_range_.cpu().tolist()
    
    def state_verification_range(self):
        return self.state_range_.cpu().tolist()

    
    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        return wrapped_state

    def periodic_transform_fn(self, input):
        return input.cuda()
    
    # ParameterizedVertDrone2D dynamics
    # \dot v = k*u - g
    # \dot z = v
    # \dot k = 0
    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = state[..., 2] * control[..., 0] - self.gravity
        dsdt[..., 1] = state[..., 0]
        dsdt[..., 2] = 0
        return dsdt

    def boundary_fn(self, state):
        return -torch.abs(state[..., 1] - 1.5) + 1.5

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        return  torch.abs(state[..., 2] *dvds[..., 0]) * self.input_magnitude_max \
            - dvds[..., 0] * self.gravity \
            + dvds[..., 1] * state[..., 0]

    def optimal_control(self, state, dvds):
        return torch.sign(dvds[..., 0])[..., None]

    def optimal_disturbance(self, state, dvds):
        return torch.tensor([0])

    def plot_config(self):
        return {
            'state_slices': [0, 1.5, self.input_multiplier],
            'state_labels': ['v', 'z', 'k'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 2,
        }

class Dubins3D(Dynamics):
    def __init__(self, set_mode: str):
        self.goalR = 0.5
        self.velocity = 1.
        self.omega_max = 1.2
        self.state_range_ = torch.tensor([[-1, 1],[-1, 1],[-math.pi, math.pi]]).cuda()
        self.control_range_ =torch.tensor([[-self.omega_max, self.omega_max]]).cuda()
        self.eps_var=torch.tensor([1]).cuda()
        self.control_init= torch.zeros(1).cuda()
        self.set_mode=set_mode

        state_mean_=(self.state_range_[:,0]+self.state_range_[:,1])/2.0
        state_var_=(self.state_range_[:,1]-self.state_range_[:,0])/2.0
        super().__init__(
            name="Dubins3D", loss_type='brt_hjivi', set_mode=set_mode,
            state_dim=3, input_dim=5, control_dim=1, disturbance_dim=0,
            state_mean=state_mean_.cpu().tolist(),
            state_var=state_var_.cpu().tolist(),    
            value_mean=0.5,
            value_var=1,
            value_normto=0.02,
            deepReach_model='exact'
        )

    def control_range(self, state):
        return [[-self.omega_max, self.omega_max]]

    def state_test_range(self):
        return self.state_range_.cpu().tolist()
    
    def state_verification_range(self):
        return self.state_range_.cpu().tolist()

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 2] = (
            wrapped_state[..., 2] + math.pi) % (2 * math.pi) - math.pi
        return wrapped_state

    def periodic_transform_fn(self, input):
        output_shape = list(input.shape)
        output_shape[-1] = output_shape[-1]+1
        transformed_input = torch.zeros(output_shape)
        transformed_input[..., :3] = input[..., :3]
        transformed_input[..., 3] = torch.sin(input[..., 3]*self.state_var[-1])
        transformed_input[..., 4] = torch.cos(input[..., 3]*self.state_var[-1])
        return transformed_input.cuda()
    
    # Dubins3D dynamics
    # \dot x    = v \cos \theta
    # \dot y    = v \sin \theta
    # \dot \theta = u

    def dsdt(self, state, control, disturbance):
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = self.velocity * torch.cos(state[..., 2])
        dsdt[..., 1] = self.velocity * torch.sin(state[..., 2])
        dsdt[..., 2] = control[..., 0]
        return dsdt

    def boundary_fn(self, state):
        return torch.norm(state[..., :2], dim=-1) - 0.5

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        if self.set_mode =="avoid":
            return self.velocity * (torch.cos(state[..., 2]) * dvds[..., 0] + torch.sin(state[..., 2]) * dvds[..., 1]) + self.omega_max * torch.abs(dvds[..., 2]) 
        elif self.set_mode =="reach":
            return self.velocity * (torch.cos(state[..., 2]) * dvds[..., 0] + torch.sin(state[..., 2]) * dvds[..., 1]) - self.omega_max * torch.abs(dvds[..., 2]) 
        else:
            raise NotImplementedError
        
    def optimal_control(self, state, dvds):
        if self.set_mode =="avoid":
            return (self.omega_max * torch.sign(dvds[..., 2]))[..., None]
        elif self.set_mode =="reach":
            return -(self.omega_max * torch.sign(dvds[..., 2]))[..., None]
        else:
            raise NotImplementedError

    def optimal_disturbance(self, state, dvds):
        return 0

    def plot_config(self):
        return {
            'state_slices': [0, 0, 0],
            'state_labels': ['x', 'y', r'$\theta$'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 2,
        }
    
class Quadrotor(Dynamics):
    def __init__(self, collisionR: float, collective_thrust_max: float,  set_mode: str):  # simpler quadrotor
        self.collective_thrust_max = collective_thrust_max
        # self.body_rate_acc_max = body_rate_acc_max
        self.m = 1  # mass
        self.arm_l = 0.17
        self.CT = 1
        self.CM = 0.016
        self.Gz = -9.8

        self.dwx_max = 8
        self.dwy_max = 8
        self.dwz_max = 4
        self.dist_dwx_max = 0
        self.dist_dwy_max = 0
        self.dist_dwz_max = 0
        self.dist_f = 0

        self.collisionR = collisionR
        self.reach_fn_weight = 1.
        self.avoid_fn_weight = 0.3
        self.state_range_ = torch.tensor([
            [-3.0, 3.0],
            [-3.0, 3.0],
            [-3.0, 3.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-5.0, 5.0],
            [-5.0, 5.0],
            [-5.0, 5.0],
            [-5.0, 5.0],
            [-5.0, 5.0],
            [-5.0, 5.0],
            ]).cuda()
        self.control_range_ =torch.tensor([[-self.collective_thrust_max, self.collective_thrust_max],
                [-self.dwx_max, self.dwx_max],
                [-self.dwy_max, self.dwy_max],
                [-self.dwz_max, self.dwz_max]]).cuda()
        self.eps_var=torch.tensor([20,8,8,4]).cuda()
        self.control_init= torch.tensor([-self.Gz*0.0,0,0,0]).cuda() 

        state_mean_=(self.state_range_[:,0]+self.state_range_[:,1])/2.0
        state_var_=(self.state_range_[:,1]-self.state_range_[:,0])/2.0
        if set_mode=='reach_avoid':
            l_type='brat_hjivi'
        else:
            l_type='brt_hjivi'
        super().__init__(
            name='Quadrotor', loss_type=l_type, set_mode=set_mode,
            state_dim=13, input_dim=14, control_dim=4, disturbance_dim=0,
            state_mean=state_mean_.cpu().tolist(),
            state_var=state_var_.cpu().tolist(),    
            value_mean=(math.sqrt(3.0**2 + 3.0**2) -
                        2 * self.collisionR) / 2,
            value_var=math.sqrt(3.0**2 + 3.0**2)/2,
            value_normto=0.02,
            deepReach_model='exact',
        )
    def normalize_q(self, x):
        # normalize quaternion
        normalized_x = x*1.0
        q_tensor = x[..., 3:7]
        q_tensor = torch.nn.functional.normalize(
            q_tensor, p=2,dim=-1)  # normalize quaternion
        normalized_x[..., 3:7] = q_tensor
        return normalized_x
    
    def clamp_state_input(self, state_input):
        return self.normalize_q(state_input)

    def control_range(self, state):
        return [[-self.collective_thrust_max, self.collective_thrust_max],
                [-self.dwx_max, self.dwx_max],
                [-self.dwy_max, self.dwy_max],
                [-self.dwz_max, self.dwz_max]]

    def state_test_range(self):
        return self.state_range_.cpu().tolist()
    
    def state_verification_range(self):
        return self.state_range_.cpu().tolist()

    def periodic_transform_fn(self, input):
        return input.cuda()
    
    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        # return wrapped_state
        return self.normalize_q(wrapped_state)

    def dsdt(self, state, control, disturbance):
        qw = state[..., 3] * 1.0
        qx = state[..., 4] * 1.0
        qy = state[..., 5] * 1.0
        qz = state[..., 6] * 1.0
        vx = state[..., 7] * 1.0
        vy = state[..., 8] * 1.0
        vz = state[..., 9] * 1.0
        wx = state[..., 10] * 1.0
        wy = state[..., 11] * 1.0
        wz = state[..., 12] * 1.0
        f = (control[..., 0]) * 1.0

        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = vx
        dsdt[..., 1] = vy
        dsdt[..., 2] = vz
        dsdt[..., 3] = -(wx * qx + wy * qy + wz * qz) / 2.0
        dsdt[..., 4] = (wx * qw + wz * qy - wy * qz) / 2.0
        dsdt[..., 5] = (wy * qw - wz * qx + wx * qz) / 2.0
        dsdt[..., 6] = (wz * qw + wy * qx - wx * qy) / 2.0
        dsdt[..., 7] = 2 * (qw * qy + qx * qz) * self.CT / \
            self.m * f
        dsdt[..., 8] = 2 * (-qw * qx + qy * qz) * self.CT / \
            self.m * f
        dsdt[..., 9] = self.Gz + (1 - 2 * torch.pow(qx, 2) - 2 *
                                  torch.pow(qy, 2)) * self.CT / self.m * f
        dsdt[..., 10] = (control[..., 1]
                         ) * 1.0 - 5 * wy * wz / 9.0
        dsdt[..., 11] = (control[..., 2]
                         ) * 1.0 + 5 * wx * wz / 9.0
        dsdt[..., 12] = (control[..., 3]) * 1.0

        return dsdt
    
    def dist_to_cylinder(self, state, a, b):
        '''for cylinder with full body collision'''
        state_=state*1.0
        state_[...,0]=state_[...,0]- a
        state_[...,1]=state_[...,1]- b

        # create normal vector
        v = torch.zeros_like(state_[..., 4:7])
        v[..., 2] = 1
        v = quaternion.quaternion_apply(state_[..., 3:7], v)
        vx = v[..., 0]
        vy = v[..., 1]
        vz = v[..., 2]
        # compute vector from center of quadrotor to the center of cylinder
        px = state_[..., 0]
        py = state_[..., 1]

        # get full body distance
        dist = torch.norm(state_[..., :2], dim=-1)
        # return dist- self.collisionR
        dist = dist- torch.sqrt((self.arm_l**2*px**2*vz**2)/(px**2*vx**2 + px**2*vz**2 + 2*px*py*vx*vy + py**2*vy**2 + py**2*vz**2)
                           + (self.arm_l**2*py**2*vz**2)/(px**2*vx**2 + px**2*vz**2 + 2*px*py*vx*vy + py**2*vy**2 + py**2*vz**2))
        return torch.maximum(dist, torch.zeros_like(dist)) - self.collisionR
    
    def reach_fn(self, state):
        state_=state*1.0
        state_[...,0]=state_[...,0] - 0.
        state_[...,1]=state_[...,1]
        return (torch.norm(state[..., :2], dim=-1)-0.3)*self.reach_fn_weight

    def avoid_fn(self, state):
        return self.avoid_fn_weight*torch.minimum(self.dist_to_cylinder(state,0.0,0.75), self.dist_to_cylinder(state,0.0,-0.75))

    def boundary_fn(self, state):
        if self.set_mode=='avoid':
            return self.dist_to_cylinder(state,0.0,0.0)
        else:
            return torch.maximum(self.reach_fn(state), -self.avoid_fn(state))


    def sample_target_state(self, num_samples):
        target_state_range = self.state_test_range()
        target_state_range[0] = [-1, 1]
        target_state_range[1] = [-0.25, 0.25]
        target_state_range = torch.tensor(target_state_range)
        return target_state_range[:, 0] + torch.rand(num_samples, self.state_dim)*(target_state_range[:, 1] - target_state_range[:, 0])
    
    def cost_fn(self, state_traj):
        if self.set_mode=='avoid':
            return torch.min(self.boundary_fn(state_traj), dim=-1).values
        else:
            # return min_t max{l(x(t)), max_k_up_to_t{-g(x(k))}}, where l(x) is reach_fn, g(x) is avoid_fn
            reach_values = self.reach_fn(state_traj)
            avoid_values = self.avoid_fn(state_traj)
            return torch.min(torch.clamp(reach_values, min=torch.max(-avoid_values, dim=-1).values.unsqueeze(-1)),dim=-1).values

    def hamiltonian(self, state, dvds):
        if self.set_mode in ['reach', 'reach_avoid']:
            qw = state[..., 3] * 1.0
            qx = state[..., 4] * 1.0
            qy = state[..., 5] * 1.0
            qz = state[..., 6] * 1.0
            vx = state[..., 7] * 1.0
            vy = state[..., 8] * 1.0
            vz = state[..., 9] * 1.0
            wx = state[..., 10] * 1.0
            wy = state[..., 11] * 1.0
            wz = state[..., 12] * 1.0

            c1 = 2 * (qw * qy + qx * qz) * self.CT / self.m
            c2 = 2 * (-qw * qx + qy * qz) * self.CT / self.m
            c3 = (1 - 2 * torch.pow(qx, 2) - 2 *
                  torch.pow(qy, 2)) * self.CT / self.m

            # Compute the hamiltonian for the quadrotor
            ham = dvds[..., 0] * vx + dvds[..., 1] * vy + dvds[..., 2] * vz
            ham += -dvds[..., 3] * (wx * qx + wy * qy + wz * qz) / 2.0
            ham += dvds[..., 4] * (wx * qw + wz * qy - wy * qz) / 2.0
            ham += dvds[..., 5] * (wy * qw - wz * qx + wx * qz) / 2.0
            ham += dvds[..., 6] * (wz * qw + wy * qx - wx * qy) / 2.0
            ham += dvds[..., 9] * self.Gz
            ham += -dvds[..., 10] * 5 * wy * wz / \
                9.0 + dvds[..., 11] * 5 * wx * wz / 9.0

            ham -= torch.abs(dvds[..., 7] * c1 + dvds[..., 8] *
                             c2 + dvds[..., 9] * c3) * self.collective_thrust_max

            ham -= torch.abs(dvds[..., 10]) * self.dwx_max + torch.abs(
                dvds[..., 11]) * self.dwy_max + torch.abs(dvds[..., 12]) * self.dwz_max

        elif self.set_mode == 'avoid':
            qw = state[..., 3] * 1.0
            qx = state[..., 4] * 1.0
            qy = state[..., 5] * 1.0
            qz = state[..., 6] * 1.0
            vx = state[..., 7] * 1.0
            vy = state[..., 8] * 1.0
            vz = state[..., 9] * 1.0
            wx = state[..., 10] * 1.0
            wy = state[..., 11] * 1.0
            wz = state[..., 12] * 1.0

            c1 = 2 * (qw * qy + qx * qz) * self.CT / self.m
            c2 = 2 * (-qw * qx + qy * qz) * self.CT / self.m
            c3 = (1 - 2 * torch.pow(qx, 2) - 2 *
                  torch.pow(qy, 2)) * self.CT / self.m

            # Compute the hamiltonian for the quadrotor
            ham = dvds[..., 0] * vx + dvds[..., 1] * vy + dvds[..., 2] * vz
            ham += -dvds[..., 3] * (wx * qx + wy * qy + wz * qz) / 2.0
            ham += dvds[..., 4] * (wx * qw + wz * qy - wy * qz) / 2.0
            ham += dvds[..., 5] * (wy * qw - wz * qx + wx * qz) / 2.0
            ham += dvds[..., 6] * (wz * qw + wy * qx - wx * qy) / 2.0
            ham += dvds[..., 9] * self.Gz
            ham += -dvds[..., 10] * 5 * wy * wz / \
                9.0 + dvds[..., 11] * 5 * wx * wz / 9.0

            ham += torch.abs(dvds[..., 7] * c1 + dvds[..., 8] *
                             c2 + dvds[..., 9] * c3) * self.collective_thrust_max

            ham += torch.abs(dvds[..., 10]) * self.dwx_max + torch.abs(
                dvds[..., 11]) * self.dwy_max + torch.abs(dvds[..., 12]) * self.dwz_max

        else:
            raise NotImplementedError

        return ham

    def optimal_control(self, state, dvds):
        if self.set_mode in ['reach', 'reach_avoid']:
            qw = state[..., 3] * 1.0
            qx = state[..., 4] * 1.0
            qy = state[..., 5] * 1.0
            qz = state[..., 6] * 1.0

            c1 = 2 * (qw * qy + qx * qz) * self.CT / self.m
            c2 = 2 * (-qw * qx + qy * qz) * self.CT / self.m
            c3 = (1 - 2 * torch.pow(qx, 2) - 2 *
                  torch.pow(qy, 2)) * self.CT / self.m

            u1 = -self.collective_thrust_max * \
                torch.sign(dvds[..., 7] * c1 + dvds[..., 8] *
                           c2 + dvds[..., 9] * c3)
            u2 = -self.dwx_max * torch.sign(dvds[..., 10])
            u3 = -self.dwy_max * torch.sign(dvds[..., 11])
            u4 = -self.dwz_max * torch.sign(dvds[..., 12])
        elif self.set_mode == 'avoid':
            qw = state[..., 3] * 1.0
            qx = state[..., 4] * 1.0
            qy = state[..., 5] * 1.0
            qz = state[..., 6] * 1.0

            c1 = 2 * (qw * qy + qx * qz) * self.CT / self.m
            c2 = 2 * (-qw * qx + qy * qz) * self.CT / self.m
            c3 = (1 - 2 * torch.pow(qx, 2) - 2 *
                  torch.pow(qy, 2)) * self.CT / self.m

            u1 = self.collective_thrust_max * \
                torch.sign(dvds[..., 7] * c1 + dvds[..., 8] *
                           c2 + dvds[..., 9] * c3)
            u2 = self.dwx_max * torch.sign(dvds[..., 10])
            u3 = self.dwy_max * torch.sign(dvds[..., 11])
            u4 = self.dwz_max * torch.sign(dvds[..., 12])

        return torch.cat((u1[..., None], u2[..., None], u3[..., None], u4[..., None]), dim=-1)

    def optimal_disturbance(self, state, dvds):
        return torch.zeros(1)


    def plot_config(self):
        return {
            'state_slices': [0.96,  1.18,  0.54,  0.44, -0.45,  0.27, -0.73, -2.83, -1.07, -3.34, 3.19, -2.80,  3.43],
            'state_labels': ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 7,
        }
    


class F1tenth(Dynamics):
    def __init__(self):
        # variable for dynamics
        self.mu = 1.0489
        self.C_Sf = 4.718
        self.C_Sr = 5.4562
        self.lf = 0.15875
        self.lr = 0.17145
        self.h = 0.074
        self.m = 3.74
        self.I = 0.04712
        self.s_min = -0.4189
        self.s_max = 0.4189
        self.sv_min = -3.2
        self.sv_max = 3.2
        self.v_switch = 7.319
        self.a_max = 9.51
        self.v_min = 0.1
        self.v_max = 10.0
        self.omega_max= 6.0
        self.delta_t = 0.01
        self.g = 9.81
        self.lwb = self.lf + self.lr

        self.v_mean = (self.v_min + self.v_max) / 2
        self.v_var = (self.v_max - self.v_min) / 2

        # map info
        # self.dt = np.load(map_path)
        self.origin = [-78.21853769831466, -44.37590462453829]
        self.resolution = 0.062500
        self.width = 1600
        self.height = 1600


        # control constraints
        self.input_steering_v_max = self.sv_max
        self.input_acceleration_max = self.a_max

        self.xmean=62.5/2
        self.xvar=62.5/2
        self.ymean=25
        self.yvar=25


        self.x_min=self.xmean-self.xvar
        self.x_max=self.xmean+self.xvar
        self.y_min=self.ymean-self.yvar
        self.y_max=self.ymean+self.yvar

        self.state_range_ = torch.tensor([[self.x_min, self.x_max], [self.y_min, self.y_max], [-0.4189, 0.4189], [self.v_min, self.v_max], [-math.pi, math.pi], [-self.omega_max, self.omega_max], [-1, 1]]).cuda()
        self.control_range_ = torch.tensor([[self.sv_min, self.sv_max], [-self.a_max, self.a_max]]).cuda()
        self.eps_var = torch.tensor([self.sv_max**2, self.a_max**2]).cuda()
        self.control_init = torch.tensor([0.0, 0.0]).cuda()

        # for the track
        self.obstaclemap_file = 'dynamics/F1_map_obstaclemap.mat'
        self.pixel2world = 0.0625
        self.obstacle_map = spio.loadmat(self.obstaclemap_file)
        self.obstacle_map = self.obstacle_map['obs_map']
        self.obstacle_map[self.obstacle_map == -0.] = 1
        self.obstacle_map = self.obstacle_map[int(self.y_min/self.pixel2world):int(self.y_max/self.pixel2world)+1,
                                            int(self.x_min/self.pixel2world):int(self.x_max/self.pixel2world)+1]
        self.obstacle_map = torch.tensor(self.obstacle_map) 


        self.world_range = self.state_range_.cpu().numpy()
        self.x_rangearray = torch.arange(self.obstacle_map.shape[0])
        self.y_rangearray = torch.arange(self.obstacle_map.shape[1])

        
        state_mean_=(self.state_range_[:,0]+self.state_range_[:,1])/2.0
        state_var_=(self.state_range_[:,1]-self.state_range_[:,0])/2.0

        super().__init__(
            name='F1tenth', loss_type='brt_hjivi', set_mode='avoid',
            state_dim=7, input_dim=9, control_dim=2, disturbance_dim=0,
            
            state_mean=state_mean_.cpu().tolist(),
            state_var=state_var_.cpu().tolist(),    
            value_mean=0.5, # mean of expected value function
            value_var=1.5, # (max - min)/2.0 of expected value function
            value_normto=0.02,
            deepReach_model='exact'
        )

    def state_test_range(self):
        return self.state_range_.cpu().tolist()
    
    def state_verification_range(self):
        return [
            [self.x_min, self.x_max], 
            [self.y_min, self.y_max],                      # y
            [-0.4189, 0.4189],               # steering angle
            [0.1, 8.0],                   # velocity
            [-math.pi, math.pi],                  # pose theta
            [-4.5, 4.5],                        # pose theta rate
            [-0.8, 0.8],                       # slip angle
        ]
    
    def periodic_transform_fn(self, input):
        output_shape = list(input.shape)
        output_shape[-1] = output_shape[-1]+1
        transformed_input = torch.zeros(output_shape)
        transformed_input[..., :5] = input[..., :5]
        transformed_input[..., 5] = torch.sin(input[..., 5]*self.state_var[4])
        transformed_input[..., 6] = torch.cos(input[..., 5]*self.state_var[4])
        transformed_input[..., 7:] = input[..., 6:]
        return transformed_input.cuda()
    
    def dsdt(self, state, control, disturbance):
        # here the control is steering angle v and acceleration
        f = torch.zeros_like(state)
        current_vel = state[..., 3] # [1, 65000]
        kinematic_mask = torch.abs(current_vel) < 0.5
        # switch to kinematic model for small velocities
        if torch.any(kinematic_mask):
            # print(f"kinematic_mask is {kinematic_mask.shape}")
            if len(kinematic_mask.shape)==1:
                sample_idx = kinematic_mask.nonzero(as_tuple=True)[0]
                x_ks = state[kinematic_mask][..., 0:5]
                u_ks = control[kinematic_mask]
                f_ks = torch.zeros_like(x_ks)
                f_ks[..., 0] = x_ks[..., 3]*torch.cos(x_ks[..., 4])
                f_ks[..., 1] = x_ks[..., 3]*torch.sin(x_ks[..., 4])
                f_ks[..., 2] = u_ks[..., 0]
                f_ks[..., 3] = u_ks[..., 1]
                f_ks[..., 4] = x_ks[..., 3]/self.lwb*torch.tan(x_ks[..., 2])
                f[sample_idx, :5] = f_ks
                f[sample_idx, 5] = u_ks[..., 1]/self.lwb*torch.tan(state[kinematic_mask][..., 2])+state[kinematic_mask][..., 3]/(self.lwb*torch.cos(state[kinematic_mask][..., 2])**2)*u_ks[..., 0]
                f[sample_idx, 6] = 0.
            else:
                batch_idx, sample_idx = kinematic_mask.nonzero(as_tuple=True)
                x_ks = state[kinematic_mask][..., 0:5]
                u_ks = control[kinematic_mask]
                f_ks = torch.zeros_like(x_ks)
                f_ks[..., 0] = x_ks[..., 3]*torch.cos(x_ks[..., 4])
                f_ks[..., 1] = x_ks[..., 3]*torch.sin(x_ks[..., 4])
                f_ks[..., 2] = u_ks[..., 0]
                f_ks[..., 3] = u_ks[..., 1]
                f_ks[..., 4] = x_ks[..., 3]/self.lwb*torch.tan(x_ks[..., 2])
                f[batch_idx, sample_idx, :5] = f_ks
                f[batch_idx, sample_idx, 5] = u_ks[..., 1]/self.lwb*torch.tan(state[kinematic_mask][..., 2])+state[kinematic_mask][..., 3]/(self.lwb*torch.cos(state[kinematic_mask][..., 2])**2)*u_ks[..., 0]
                f[batch_idx, sample_idx, 6] = 0.

        dynamic_mask = ~kinematic_mask
        if torch.any(dynamic_mask):
            if len(kinematic_mask.shape)==1:
                sample_idx = dynamic_mask.nonzero(as_tuple=True)[0]
                f[sample_idx, 0] = state[dynamic_mask][..., 3]*torch.cos(state[dynamic_mask][..., 6] + state[dynamic_mask][..., 4])
                f[sample_idx, 1] = state[dynamic_mask][..., 3]*torch.sin(state[dynamic_mask][..., 6] + state[dynamic_mask][..., 4])
                f[sample_idx, 2] = control[dynamic_mask][..., 0]
                f[sample_idx, 3] = control[dynamic_mask][..., 1]
                f[sample_idx, 4] = state[dynamic_mask][..., 5]
                f[sample_idx, 5] = -self.mu*self.m/(state[dynamic_mask][..., 3]*self.I*(self.lr+self.lf))*(self.lf**2*self.C_Sf*(self.g*self.lr-control[dynamic_mask][..., 1]*self.h) + self.lr**2*self.C_Sr*(self.g*self.lf + control[dynamic_mask][..., 1]*self.h))*state[dynamic_mask][..., 5] \
                        +self.mu*self.m/(self.I*(self.lr+self.lf))*(self.lr*self.C_Sr*(self.g*self.lf + control[dynamic_mask][..., 1]*self.h) - self.lf*self.C_Sf*(self.g*self.lr - control[dynamic_mask][..., 1]*self.h))*state[dynamic_mask][..., 6] \
                        +self.mu*self.m/(self.I*(self.lr+self.lf))*self.lf*self.C_Sf*(self.g*self.lr - control[dynamic_mask][..., 1]*self.h)*state[dynamic_mask][..., 2]
                f[sample_idx, 6] = (self.mu/(state[dynamic_mask][..., 3]**2*(self.lr+self.lf))*(self.C_Sr*(self.g*self.lf + control[dynamic_mask][..., 1]*self.h)*self.lr - self.C_Sf*(self.g*self.lr - control[dynamic_mask][..., 1]*self.h)*self.lf)-1)*state[dynamic_mask][..., 5] \
                        -self.mu/(state[dynamic_mask][..., 3]*(self.lr+self.lf))*(self.C_Sr*(self.g*self.lf + control[dynamic_mask][..., 1]*self.h) + self.C_Sf*(self.g*self.lr-control[dynamic_mask][..., 1]*self.h))*state[dynamic_mask][..., 6] \
                        +self.mu/(state[dynamic_mask][..., 3]*(self.lr+self.lf))*(self.C_Sf*(self.g*self.lr-control[dynamic_mask][..., 1]*self.h))*state[dynamic_mask][..., 2]
            else:
                batch_idx, sample_idx = dynamic_mask.nonzero(as_tuple=True)
                f[batch_idx, sample_idx, 0] = state[dynamic_mask][..., 3]*torch.cos(state[dynamic_mask][..., 6] + state[dynamic_mask][..., 4])
                f[batch_idx, sample_idx, 1] = state[dynamic_mask][..., 3]*torch.sin(state[dynamic_mask][..., 6] + state[dynamic_mask][..., 4])
                f[batch_idx, sample_idx, 2] = control[dynamic_mask][..., 0]
                f[batch_idx, sample_idx, 3] = control[dynamic_mask][..., 1]
                f[batch_idx, sample_idx, 4] = state[dynamic_mask][..., 5]
                f[batch_idx, sample_idx, 5] = -self.mu*self.m/(state[dynamic_mask][..., 3]*self.I*(self.lr+self.lf))*(self.lf**2*self.C_Sf*(self.g*self.lr-control[dynamic_mask][..., 1]*self.h) + self.lr**2*self.C_Sr*(self.g*self.lf + control[dynamic_mask][..., 1]*self.h))*state[dynamic_mask][..., 5] \
                        +self.mu*self.m/(self.I*(self.lr+self.lf))*(self.lr*self.C_Sr*(self.g*self.lf + control[dynamic_mask][..., 1]*self.h) - self.lf*self.C_Sf*(self.g*self.lr - control[dynamic_mask][..., 1]*self.h))*state[dynamic_mask][..., 6] \
                        +self.mu*self.m/(self.I*(self.lr+self.lf))*self.lf*self.C_Sf*(self.g*self.lr - control[dynamic_mask][..., 1]*self.h)*state[dynamic_mask][..., 2]
                f[batch_idx, sample_idx, 6] = (self.mu/(state[dynamic_mask][..., 3]**2*(self.lr+self.lf))*(self.C_Sr*(self.g*self.lf + control[dynamic_mask][..., 1]*self.h)*self.lr - self.C_Sf*(self.g*self.lr - control[dynamic_mask][..., 1]*self.h)*self.lf)-1)*state[dynamic_mask][..., 5] \
                        -self.mu/(state[dynamic_mask][..., 3]*(self.lr+self.lf))*(self.C_Sr*(self.g*self.lf + control[dynamic_mask][..., 1]*self.h) + self.C_Sf*(self.g*self.lr-control[dynamic_mask][..., 1]*self.h))*state[dynamic_mask][..., 6] \
                        +self.mu/(state[dynamic_mask][..., 3]*(self.lr+self.lf))*(self.C_Sf*(self.g*self.lr-control[dynamic_mask][..., 1]*self.h))*state[dynamic_mask][..., 2]
        #------------------------------OPT CTRL--------------------------------

        return f

    def clamp_state_input(self, state_input):
        full_input=torch.cat((torch.ones(state_input.shape[0],1).to(state_input),state_input),dim=-1)
        state=self.input_to_coord(full_input)[...,1:]
        lx=self.boundary_fn(state)
        return state_input[lx>=-1]

    def clamp_verification_state(self, state):
        lx=self.boundary_fn(state)
        return state[lx>=0]

    def clamp_control(self, state, control):
        control_clamped=control*1.0
        
        smax_mask = state[...,2]>self.s_max-0.01
        smin_mask = state[...,2]<-self.s_max+0.01
        if len(smax_mask.shape)==1:
            max_sample_idx = smax_mask.nonzero(as_tuple=True)[0]
            control_clamped[max_sample_idx, 0] = torch.clamp(control_clamped[max_sample_idx, 0],max=0)
            min_sample_idx = smin_mask.nonzero(as_tuple=True)[0]
            control_clamped[min_sample_idx, 0] = torch.clamp(control_clamped[min_sample_idx, 0],min=0)
            
        else:
            max_batch_idx, max_sample_idx = smax_mask.nonzero(as_tuple=True)
            control_clamped[max_batch_idx, max_sample_idx, 0] = torch.clamp(control_clamped[max_batch_idx, max_sample_idx, 0],max=0)
            min_batch_idx, min_sample_idx = smin_mask.nonzero(as_tuple=True)
            control_clamped[min_batch_idx, min_sample_idx, 0] = torch.clamp(control_clamped[min_batch_idx, min_sample_idx, 0],min=0)

        accelerate_upper = torch.ones(state.shape[:-1], device=state.device) * self.input_acceleration_max
        accelerate_upper[state[..., 3] > self.v_switch] = self.input_acceleration_max * self.v_switch / state[state[..., 3] > self.v_switch][..., 3]
        
        acc_mask=control_clamped[...,1]>accelerate_upper
        if len(acc_mask.shape)==1:
            sample_idx = acc_mask.nonzero(as_tuple=True)[0]
            control_clamped[sample_idx,1]=accelerate_upper[sample_idx]
        else:
            batch_idx, sample_idx = acc_mask.nonzero(as_tuple=True)
            control_clamped[batch_idx, sample_idx,1]=accelerate_upper[batch_idx, sample_idx]

        assert ((accelerate_upper-control_clamped[...,1])>=0.0).all()
        return control_clamped
    
    def interpolation(self, state_pixel_coords):
        self.obstacle_map=self.obstacle_map.to(state_pixel_coords)
        # Find the indices surrounding the query points
        x0 = torch.floor(state_pixel_coords[..., 0]).long()
        x1 = x0 + 1
        y0 = torch.floor(state_pixel_coords[..., 1]).long()
        y1 = y0 + 1
        # Ensure indices are within bounds
        x0 = torch.clamp(x0, 0, self.x_rangearray.size(0) - 1)
        x1 = torch.clamp(x1, 0, self.x_rangearray.size(0) - 1)
        y0 = torch.clamp(y0, 0, self.y_rangearray.size(0) - 1)
        y1 = torch.clamp(y1, 0, self.y_rangearray.size(0) - 1)

        # Gather the values at the corner points for each query point
        v00 = self.obstacle_map[x0, y0]
        v01 = self.obstacle_map[x0, y1]
        v10 = self.obstacle_map[x1, y0]
        v11 = self.obstacle_map[x1, y1]
        # Compute the fractional part for each query point
        x_frac = state_pixel_coords[..., 0] - x0.float()
        y_frac = state_pixel_coords[..., 1] - y0.float()
        # Bilinear interpolation for each query point
        v0 = v00 * (1 - x_frac) + v10 * x_frac
        v1 = v01 * (1 - x_frac) + v11 * x_frac
        # Interpolated value
        interp_values = v0 * (1 - y_frac) + v1 * y_frac
        return interp_values
    
    def boundary_fn(self, state): 
        # MPC: state = B * N * H * 7
        # DeepReach: state = B * 7
        # Takes the cordinates in the real world and returns the lx for the obstacles at those coords
        # shift the origin so that the min is 0
        shiftedCoords = state - torch.tensor(self.world_range[...,0].reshape(self.state_dim,), device = state.device) # num states involve time as well
        # extract and flip the x and y pos for image space query
        if shiftedCoords.shape[0] == 1:
            shiftedCoords_pos_world = np.squeeze(shiftedCoords[...,0:2])
        else:
            shiftedCoords_pos_world = shiftedCoords[...,0:2]
        if len(shiftedCoords_pos_world.shape)==2:
            shiftedCoords_pos_image =  torch.fliplr(shiftedCoords_pos_world) # B*2 for deepreach, B*N*H*2 for MPC 
        else:
            shiftedCoords_pos_image=torch.flip(shiftedCoords_pos_world, [-1])
        # convert the world coordinates to pixel coordinates
        shiftedCoords_pos_pixel = shiftedCoords_pos_image/self.pixel2world # note this does not have to be integers due to the regularGridInterpolator
        # query the generator
        obstacle_value = self.interpolation(shiftedCoords_pos_pixel)  # obstacle value only depends on pos
        # obstacle_value = obstacle_value.reshape([obstacle_value.shape[0],1]) # this should be the lx
        return obstacle_value
    
    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values

    def hamiltonian(self, state, dvds):
        if self.set_mode == 'reach':
            raise NotImplementedError

        elif self.set_mode == 'avoid':
            opt_control=self.optimal_control(state,dvds)
            dsdt_=self.dsdt(state,opt_control,None)
            ham=torch.sum(dvds*dsdt_,dim=-1)
        return ham

    def optimal_control(self, state, dvds):

        if self.set_mode == 'reach':
            raise NotImplementedError
        elif self.set_mode == 'avoid':
            unsqueeze_u=False
            if state.shape[0]==1:
                state=state.squeeze(0)
                dvds=dvds.squeeze(0)
                unsqueeze_u=True
            batch_dims = state.shape[:-1]
            u = torch.zeros(*batch_dims, 2, device=state.device)

            
            kinematic_mask = torch.abs(state[..., 3]) < 0.5
            # if the number of kinematic_mask's dimensional is greater than 1, print
            
            if torch.any(kinematic_mask):
                if len(kinematic_mask.shape)==1:
                    sample_idx = kinematic_mask.nonzero(as_tuple=True)[0]
                    u[sample_idx, 0] = self.input_steering_v_max * torch.sign(dvds[kinematic_mask][..., 2] + dvds[kinematic_mask][..., 5] * state[kinematic_mask][..., 3] / (self.lwb * torch.cos(state[kinematic_mask][..., 2])**2))
                    u[sample_idx, 1] = self.input_acceleration_max * torch.sign(dvds[kinematic_mask][..., 3] + dvds[kinematic_mask][..., 5] / self.lwb * torch.tan(state[kinematic_mask][..., 2]))
                else:
                    batch_idx, sample_idx = kinematic_mask.nonzero(as_tuple=True)
                    u[batch_idx, sample_idx, 0] = self.input_steering_v_max * torch.sign(dvds[kinematic_mask][..., 2] + dvds[kinematic_mask][..., 5] * state[kinematic_mask][..., 3] / (self.lwb * torch.cos(state[kinematic_mask][..., 2])**2))
                    u[batch_idx, sample_idx, 1] = self.input_acceleration_max * torch.sign(dvds[kinematic_mask][..., 3] + dvds[kinematic_mask][..., 5] / self.lwb * torch.tan(state[kinematic_mask][..., 2]))

            dynamic_mask = ~kinematic_mask
            if torch.any(dynamic_mask):
                if len(kinematic_mask.shape)==1:
                    sample_idx = dynamic_mask.nonzero(as_tuple=True)[0]
                    u[sample_idx, 0] = self.input_steering_v_max * torch.sign(dvds[dynamic_mask][..., 2])

                    u[sample_idx, 1] = self.input_acceleration_max * torch.sign(
                        dvds[dynamic_mask][..., 3] \
                        + dvds[dynamic_mask][..., 5] * ((-self.mu * self.m / (state[dynamic_mask][..., 3] * self.I * (self.lr + self.lf))*(-self.lf**2*self.C_Sf*self.h + self.lr**2*self.C_Sr*self.h))*state[dynamic_mask][..., 5] \
                            + self.mu * self.m / (self.I * (self.lr + self.lf)) * (self.lr*self.C_Sr*self.h + self.lf*self.C_Sf*self.h)*state[dynamic_mask][..., 6] \
                            - self.mu * self.m / (self.I * (self.lr + self.lf)) * self.lf*self.C_Sf*self.h*state[dynamic_mask][..., 2])\
                        + dvds[dynamic_mask][..., 6] * ((self.mu / (state[dynamic_mask][..., 3]**2 * (self.lr + self.lf)) * (self.C_Sr*self.h*self.lr + self.C_Sf*self.h*self.lf))*state[dynamic_mask][..., 5]
                            - self.mu / (state[dynamic_mask][..., 3] * (self.lr + self.lf)) * (self.C_Sr*self.h - self.C_Sf*self.h)*state[dynamic_mask][..., 6]
                            - self.mu / (state[dynamic_mask][..., 3] * (self.lr + self.lf)) * self.C_Sf*self.h*state[dynamic_mask][..., 2])
                    )
                else:
                    batch_idx, sample_idx = dynamic_mask.nonzero(as_tuple=True)

                    u[batch_idx, sample_idx, 0] = self.input_steering_v_max * torch.sign(dvds[dynamic_mask][..., 2])

                    u[batch_idx, sample_idx, 1] = self.input_acceleration_max * torch.sign(
                        dvds[dynamic_mask][..., 3] \
                        + dvds[dynamic_mask][..., 5] * ((-self.mu * self.m / (state[dynamic_mask][..., 3] * self.I * (self.lr + self.lf))*(-self.lf**2*self.C_Sf*self.h + self.lr**2*self.C_Sr*self.h))*state[dynamic_mask][..., 5] \
                            + self.mu * self.m / (self.I * (self.lr + self.lf)) * (self.lr*self.C_Sr*self.h + self.lf*self.C_Sf*self.h)*state[dynamic_mask][..., 6] \
                            - self.mu * self.m / (self.I * (self.lr + self.lf)) * self.lf*self.C_Sf*self.h*state[dynamic_mask][..., 2])\
                        + dvds[dynamic_mask][..., 6] * ((self.mu / (state[dynamic_mask][..., 3]**2 * (self.lr + self.lf)) * (self.C_Sr*self.h*self.lr + self.C_Sf*self.h*self.lf))*state[dynamic_mask][..., 5]
                            - self.mu / (state[dynamic_mask][..., 3] * (self.lr + self.lf)) * (self.C_Sr*self.h - self.C_Sf*self.h)*state[dynamic_mask][..., 6]
                            - self.mu / (state[dynamic_mask][..., 3] * (self.lr + self.lf)) * self.C_Sf*self.h*state[dynamic_mask][..., 2])
                    )
            u=self.clamp_control(state,u)
            if unsqueeze_u:
                u=u[None,...]
        return u

    def sample_target_state(self, num_samples):
        raise NotImplementedError

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 4] = (
            wrapped_state[..., 4] + math.pi) % (2 * math.pi) - math.pi
        return wrapped_state
    
    def optimal_disturbance(self, state, dvds):
        return torch.tensor([0])
    
    def plot_config(self):
        return {
            'state_slices': [0, 0, 0, 8.0, 0, 0, 0],
            'state_labels': ['x', 'y', 'sangle', 'v', 'posetheta', 'poserate', 'slipangle'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 4,
        }
    
class LessLinearND(Dynamics):
    def __init__(self, N:int, gamma:float, mu:float, alpha:float, goalR:float):
        u_max, set_mode = 0.5, "reach" # TODO: unfix

        self.N = N 
        self.u_max = u_max
        self.input_center = torch.zeros(N-1)
        self.input_shape = "box"
        self.game = set_mode
        
        self.A = (-0.5 * torch.eye(N) - torch.cat((torch.cat((torch.zeros(1,1),torch.ones(N-1,1)),0),torch.zeros(N,N-1)),1)).cuda()
        self.B = torch.cat((torch.zeros(1,N-1), 0.4*torch.eye(N-1)), 0).cuda()
        self.Bumax = u_max * torch.matmul(self.B, torch.ones(self.N-1).cuda()).unsqueeze(0).unsqueeze(0).cuda()
        self.C = torch.cat((torch.zeros(1,N-1), 0.1*torch.eye(N-1)), 0)
        self.gamma, self.mu, self.alpha = gamma, mu, alpha
        self.gamma_orig, self.mu_orig, self.alpha_orig = gamma, mu, alpha

        self.goalR_2d = goalR
        self.goalR = ((N-1) ** 0.5) * self.goalR_2d # accounts for N-dimensional combination
        self.ellipse_params = torch.cat((((N-1) ** 0.5) * torch.ones(1), torch.ones(N-1) / 1.), 0) # accounts for N-dimensional combination

        self.state_range_ = torch.tensor([[-1, 1] for _ in range(self.N)]).cuda()
        self.control_range_ =torch.tensor([[-u_max, u_max] for _ in range(self.N-1)]).cuda()
        self.eps_var=torch.tensor([u_max for _ in range(self.N-1)]).cuda()
        self.control_init= torch.tensor([0.0 for _ in range(self.N-1)]).cuda() 

        super().__init__(
            name='50D system', loss_type='brt_hjivi', set_mode=set_mode,
            state_dim=N, input_dim=N+1, control_dim=N-1, disturbance_dim=N-1,
            state_mean=[0 for _ in range(N)], 
            state_var=[1 for _ in range(N)],
            value_mean=0.25, 
            value_var=0.5, 
            value_normto=0.02,
            deepReach_model="exact",
        )

    def vary_nonlinearity(self, epsilon):
        self.gamma = epsilon * self.gamma_orig
        self.mu = epsilon * self.mu_orig
        # self.alpha = epsilon * self.alpha_orig #shouldn't be varied since its not a scalar (1-\lambda) l(\cdot) +  \lambda f(\cdot)

    def state_test_range(self):
        return [[-1, 1] for _ in range(self.N)]
    
    def state_verification_range(self):
        return [[-1, 1] for _ in range(self.N)]
    
    def control_range(self, state):
        return self.control_range_.cpu().tolist()

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        return wrapped_state
        
    # LessLinear dynamics
    # \dot xN    = (aN \cdot x) + (no ctrl or dist) + mu * sin(alpha * xN) * xN^2
    # \dot xi    = (ai \cdot x) + bi * ui + ci * di - gamma * xi * xN^2
    # i.e.
    # \dot x = Ax + Bu + Cd + NLterm(x, gamma, mu, alpha)
    # def dsdt(self, state, control, disturbance):
    #     dsdt = torch.zeros_like(state)
    #     nl_term_N = self.mu * torch.sin(self.alpha * state[..., 0]) * state[..., 0] * state[..., 0]
    #     nl_term_i = torch.multiply(-self.gamma * state[..., 0] * state[..., 0], state[..., 1:])
    #     dsdt[..., :] = torch.matmul(self.A, state[..., :]) + torch.matmul(self.B, control[..., :]) + torch.cat((nl_term_N, nl_term_i), 0)
    #     return dsdt
    def dsdt(self, state, control, disturbance):
        x0 = state[..., 0]  # shape: (...)
        x_rest = state[..., 1:]  # shape: (..., n-1)

        # Nonlinear terms
        nl_term_N = self.mu * torch.sin(self.alpha * x0) * x0 * x0  # shape: (...)
        nl_term_N = nl_term_N.unsqueeze(-1)  # shape: (..., 1)

        x0_squared = (x0 ** 2).unsqueeze(-1)  # shape: (..., 1)
        nl_term_i = -self.gamma * x0_squared * x_rest  # broadcasted: (..., n-1)

        nl_term = torch.cat([nl_term_N, nl_term_i], dim=-1)  # shape: (..., n)

        # Linear terms
        linear_term = torch.matmul(state, self.A.T) + torch.matmul(control, self.B.T)

        return linear_term + nl_term

    
    def periodic_transform_fn(self, input):
        return input.cuda()
    
    def boundary_fn(self, state):
        if self.ellipse_params.device != state.device: # FIXME: Patch to cover de/attached state bug
            if state.device.type == 'cuda':
                self.ellipse_params = self.ellipse_params.cuda()
            else:
                self.ellipse_params = self.ellipse_params.cpu()
        return 0.5 * (torch.square(torch.norm(self.ellipse_params * state[..., :], dim=-1)) - (self.goalR ** 2))
        # return 0.5 * (torch.square(torch.norm(torch.cat((((self.N-1)**0.5)*torch.ones(1),torch.ones(self.N-1)),0) * state[..., :], dim=-1)) - (self.goalR ** 2))

    def sample_target_state(self, num_samples):
        raise NotImplementedError
    
    def cost_fn(self, state_traj):
        return torch.min(self.boundary_fn(state_traj), dim=-1).values
    
    def hamiltonian(self, state, dvds):

        nl_term_N = (self.mu * torch.sin(self.alpha * state[..., 0]) * state[..., 0] * state[..., 0]).unsqueeze(-1)
        nl_term_i = (-self.gamma * state[..., 0] * state[..., 0]).t() * state[..., 1:]
        pAx = (dvds * (torch.matmul(state, self.A.t()) + torch.cat((nl_term_N, nl_term_i), 2))).sum(2)
        pBumax = (torch.abs(dvds) * self.Bumax).sum(2)

        if self.set_mode == 'reach':
            return pAx - pBumax
        elif self.set_mode == 'avoid':
            return pAx + pBumax

    def optimal_control(self, state, dvds):
        if self.set_mode == 'reach':
            return -self.u_max * torch.sign(dvds[..., 1:])
        elif self.set_mode == 'avoid':
            return self.u_max * torch.sign(dvds[..., 1:])

    def optimal_disturbance(self, state, dvds):
        return 0.0
    
    def plot_config(self): # FIXME
        return {
            'state_slices': [0 for _ in range(self.N)],
            'state_labels': ['xN'] + ['x' + str(i) for i in range(1, self.N)],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 2,
        }

   