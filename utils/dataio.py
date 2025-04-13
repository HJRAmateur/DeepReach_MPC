import torch
from torch.utils.data import Dataset
import numpy as np
import math
from utils import MPC, MPC_FRS
import os
from tqdm import tqdm 
import random
from bisect import bisect_right
import gc

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(
            np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(
            np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords


# uses model input and real boundary fn
class ReachabilityDataset(Dataset):
    def __init__(self, dynamics, numpoints, pretrain, pretrain_iters, tMin, tMax, counter_start, counter_end, num_src_samples, num_target_samples, 
                 use_MPC, time_curr, MPC_data_path, num_MPC_perturbation_samples, MPC_dt, MPC_mode, MPC_sample_mode, MPC_style,
                 MPC_lambda_, MPC_batch_size, MPC_receding_horizon, num_MPC_data_samples, num_iterative_refinement, time_till_refinement, num_MPC_batches=1, 
                 aug_with_MPC_data=0, policy=None, refine_dataset=True, PDE_curr=False):
        self.dynamics = dynamics
        self.numpoints = numpoints
        self.pretrain = pretrain
        self.pretrain_counter = 0
        self.pretrain_iters = pretrain_iters
        self.tMin = tMin
        self.tMax = tMax
        self.counter = counter_start
        self.counter_end = counter_end
        self.num_src_samples = num_src_samples
        self.num_target_samples = num_target_samples



        self.use_MPC=use_MPC
        self.PDE_curr=PDE_curr
        self.time_curr=time_curr
        self.num_MPC_data_samples=num_MPC_data_samples
        self.aug_with_MPC_data=aug_with_MPC_data
        self.MPC_batch_size=MPC_batch_size
        self.MPC_receding_horizon=MPC_receding_horizon
        self.MPC_dt=MPC_dt
        self.policy=policy
        self.num_MPC_batches=num_MPC_batches
        self.time_till_refinement=time_till_refinement
        self.MPC_data_path=MPC_data_path
        self.num_MPC_perturbation_samples=num_MPC_perturbation_samples
        self.MPC_mode=MPC_mode
        self.MPC_sample_mode=MPC_sample_mode

        self.epoch_till_refinement=20000
        self.refine_dataset=refine_dataset
        self.MPC_lambda_=MPC_lambda_
        self.MPC_style=MPC_style
        self.num_iterative_refinement=num_iterative_refinement
         
        if use_MPC:

            if self.dynamics.loss_type=='frs_hjivi':
                # MPC for importance sampling in dataset generation
                self.mpc_IS = MPC_FRS.MPC_FRS(horizon=int(self.tMax/self.MPC_dt), receding_horizon=1, dT=MPC_dt, num_samples=2, 
                    dynamics_=dynamics, num_iterative_refinement=10, device='cuda', style='receding', time_varying_control=True)
            if MPC_data_path=='none':
                #initialize MPC dataset
                if refine_dataset:
                    self.generate_MPC_dataset(self.time_till_refinement, 0.0)
                else:
                    self.generate_MPC_dataset(self.tMax, 0.0)

            else:
                # self.MPC_inputs=torch.load(os.path.join(MPC_data_path, 'inputs.pt'))
                # self.MPC_values=torch.load(os.path.join(MPC_data_path, 'value_labels.pt'))
                MPC_inputs_mmap = np.load(os.path.join(MPC_data_path, 'inputs.npy'), mmap_mode="r")
                MPC_values_mmap = np.load(os.path.join(MPC_data_path, 'value_labels.npy'), mmap_mode="r")
                self.MPC_inputs = torch.from_numpy(MPC_inputs_mmap).detach()
                self.MPC_values = torch.from_numpy(MPC_values_mmap).detach()
                
                print("loaded %d labels"%self.MPC_inputs.shape[0])
                    
                self.mpc_time_sorted_indices = torch.argsort(self.MPC_inputs[:, 0])

            

    def use_terminal_MPC(self):
        # self.MPC_dt can be different
        if self.dynamics.loss_type != 'frs_hjivi':
            self.MPC_dt=0.005
            self.num_iterative_refinement=-1
        # self.MPC_batch_size=50000


    def __len__(self):
        return 1

    def sample_init_state(self, T=None):
        MPC_states=torch.empty((0,self.dynamics.state_dim))
        while MPC_states.shape[0]<self.MPC_batch_size:

            MPC_states_ = torch.zeros(
                self.MPC_batch_size, self.dynamics.state_dim).uniform_(-1, 1)

            if self.dynamics.name=="Quadrotor":
                MPC_states_=self.dynamics.normalize_q(MPC_states_)

            if self.dynamics.loss_type=='frs_hjivi':
                # Make half of the MPC samples roughly reachable
                MPC_states_unnorm = self.dynamics.input_to_coord(torch.cat((torch.ones(MPC_states_.shape[0],1),MPC_states_),dim=-1))[..., 1:]
                self.mpc_IS.batch_size=int(self.MPC_batch_size*0.5)
                # rollout states traj 
                self.mpc_IS.init_control_tensors(nominal_control=MPC_states_unnorm[:self.mpc_IS.batch_size,5:7])
                initial_condition_tensor= MPC_states_unnorm[:self.mpc_IS.batch_size,...]*1.0
                initial_condition_tensor[:,7]+=1.0 ## TODO: FIX temporary code for Dubins10D
                initial_condition_tensor[:,8]+=1.0 ## TODO: FIX temporary code for Dubins10D
                initial_condition_tensor[:,11]+=1.0 ## TODO: FIX temporary code for Dubins13D
                initial_condition_tensor[:,12]+=1.0 ## TODO: FIX temporary code for Dubins13D
                state_trajs,_ = self.mpc_IS.rollout_dynamics(initial_condition_tensor, 0, int(T/0.1))
                state_trajs=state_trajs.detach().cpu()
                state_reached=state_trajs[:, 1, -1, :]*1.0
                state_reached[:,7:9]=MPC_states_unnorm[:self.mpc_IS.batch_size,7:9]*1.0 ## TODO: FIX temporary code for Dubins10D
                state_reached[:,11:13]=MPC_states_unnorm[:self.mpc_IS.batch_size,11:13]*1.0 ## TODO: FIX temporary code for Dubins10D
                # print(self.dynamics.coord_to_input(torch.cat((torch.ones(state_reached.shape[0],1),state_reached),dim=-1))[..., 1:].shape,
                #       state_reached.shape,
                #       MPC_states_[:self.mpc_IS.batch_size, :].shape)
                MPC_states_[:self.mpc_IS.batch_size, :]= self.dynamics.coord_to_input(torch.cat((torch.ones(state_reached.shape[0],1),state_reached),dim=-1))[..., 1:]
            
            MPC_states=torch.cat((MPC_states,self.dynamics.clamp_state_input(MPC_states_)),dim=0)

            
        MPC_states = MPC_states[:self.MPC_batch_size,...]*1.0
        MPC_states = self.dynamics.input_to_coord(torch.cat((torch.ones(MPC_states.shape[0],1),MPC_states),dim=-1))[..., 1:]

        if self.num_target_samples>0:
            num_mpc_target_samples = min(int(self.MPC_batch_size/5),self.num_target_samples)
            target_state_samples = self.dynamics.sample_target_state(
                num_mpc_target_samples)
            MPC_states[-num_mpc_target_samples:] = target_state_samples
        return MPC_states

    def generate_MPC_dataset(self, T, t, style="random"):
        print("Generating MPC dataset")
        if self.dynamics.loss_type=='frs_hjivi':
            self.mpc = MPC_FRS.MPC_FRS(horizon=None, receding_horizon=self.MPC_receding_horizon, dT=self.MPC_dt, num_samples=self.num_MPC_perturbation_samples, 
                dynamics_=self.dynamics, device='cuda', num_iterative_refinement = self.num_iterative_refinement, style='receding', time_varying_control=True)
        else:
            self.mpc = MPC.MPC(horizon=None, receding_horizon=self.MPC_receding_horizon, dT=self.MPC_dt, num_samples=self.num_MPC_perturbation_samples, 
                    dynamics_=self.dynamics, device='cuda', mode=self.MPC_mode,
                    sample_mode=self.MPC_sample_mode,lambda_=self.MPC_lambda_, style= self.MPC_style, num_iterative_refinement = self.num_iterative_refinement)
        device='cuda'
        MPC_states = self.sample_init_state(T)

        _, _, MPC_inputs, MPC_values= self.mpc.get_batch_data(
            MPC_states.cuda(), T, self.policy, t=t) # Make sure to generate at least one batch of data at T, so we have "look-ahead" MPC labels for deepreach
        for i in tqdm(range(self.num_MPC_batches-1)):
            

            if self.dynamics.loss_type == 'frs_hjivi':
                t_max=random.randint(1, math.floor((T)/self.MPC_dt))*self.MPC_dt
                style="terminal" # TODO: currently since we haven't implemented FRS dataset refinement. Therefore we always want more data on terminal time. Need to fix later 
                
            else:
                if self.mpc.style=="direct":
                    t_max=random.randint(1, math.floor((T*1.1)/self.MPC_dt))*self.MPC_dt
                elif self.mpc.style=="receding":
                    t_max=random.randint(1, int(T/self.MPC_dt/self.mpc.receding_horizon))*self.mpc.receding_horizon*self.MPC_dt
                    # t_max=T*1.0
                else:
                    raise NotImplementedError
            
            if style=="terminal" and i<self.num_MPC_batches/2:
                t_max=T*1.0 # more data on the terminal time
            # t_max=self.tMax
            MPC_states = self.sample_init_state(t_max)
            cost, state_traj, MPC_inputs_, MPC_values_= self.mpc.get_batch_data(
                MPC_states.to(device), t_max, self.policy, t=t)

            MPC_inputs=torch.cat([MPC_inputs,MPC_inputs_],dim=0)
            MPC_values=torch.cat([MPC_values,MPC_values_],dim=0)
        # print("Generated %d labels"%MPC_inputs.shape[0])
        if style=="terminal":
            # with more coords and values being at the terminal time
            terminal_MPC_inputs_=MPC_inputs[MPC_inputs[:,0]==T,...]*1.0
            terminal_MPC_values_=MPC_values[MPC_inputs[:,0]==T,...]*1.0

            terminal_MPC_inputs_repeated=terminal_MPC_inputs_.repeat(int(self.tMax/self.MPC_dt/2), 1)
            terminal_MPC_values_repeated=terminal_MPC_values_.repeat(int(self.tMax/self.MPC_dt/2))
            

            MPC_inputs=torch.cat([MPC_inputs, terminal_MPC_inputs_repeated],dim=0)
            MPC_values=torch.cat([MPC_values, terminal_MPC_values_repeated],dim=0)


        # convert to memory-mapped tensor for faster query
        if not os.path.exists("./data"):
            os.makedirs("./data")
        device_id = os.environ.get("CUDA_VISIBLE_DEVICES")
        np.save("./data/MPC_inputs_gpu%s.npy"%device_id, MPC_inputs.cpu().numpy(), allow_pickle=False)
        np.save("./data/MPC_values_gpu%s.npy"%device_id, MPC_values.cpu().numpy(), allow_pickle=False)
        MPC_inputs_mmap = np.load("./data/MPC_inputs_gpu%s.npy"%device_id, mmap_mode="r")
        MPC_values_mmap = np.load("./data/MPC_values_gpu%s.npy"%device_id, mmap_mode="r")
        self.MPC_inputs = torch.from_numpy(MPC_inputs_mmap).detach()
        self.MPC_values = torch.from_numpy(MPC_values_mmap).detach()
        
        print("Generated %d labels"%MPC_inputs.shape[0])
            
        self.mpc_time_sorted_indices = torch.argsort(self.MPC_inputs[:, 0])

        
        del self.mpc # free it to get some memory
        gc.collect()
        torch.cuda.empty_cache()


    def __getitem__(self, idx):
        # uniformly sample domain and include coordinates where source is non-zero
        model_states_normed=torch.empty((0,self.dynamics.state_dim))
        while model_states_normed.shape[0]<self.numpoints:
            model_states_normed_ = torch.zeros(
                self.numpoints, self.dynamics.state_dim).uniform_(-1, 1)
            model_states_normed=torch.cat((model_states_normed,self.dynamics.clamp_state_input(model_states_normed_)),dim=0)
        model_states_normed=model_states_normed[:self.numpoints,...]
        

        if self.pretrain: # if pretrain: use entire horizon, 
            # only sample in time around the initial condition
            times = torch.full((model_states_normed.shape[0], 1), self.tMin)
        elif self.PDE_curr:
            # use a PDE curriculum instead of time curr:
            times = self.tMin + torch.zeros(model_states_normed.shape[0], 1).uniform_(
                0, self.tMax)
            # make sure we always have training samples at the initial time
            if self.dynamics.deepReach_model in ['vanilla', 'diff']:
                times[-self.num_src_samples:, 0] = self.tMin
        else:
            # slowly grow time values from start time
            # make the curriculum slightly (1.1 times) longer to avoid innaccuracy on the boundary
            times = self.tMin + torch.zeros(model_states_normed.shape[0], 1).uniform_(
                0, (self.tMax*1.1-self.tMin) * min((self.counter+1) / self.counter_end, 1.0))
            
            # make sure we always have training samples at the initial time
            if self.dynamics.deepReach_model in ['vanilla', 'diff']:
                times[-self.num_src_samples:, 0] = self.tMin

        if self.num_target_samples > 0:
            target_state_samples = self.dynamics.sample_target_state(
                self.num_target_samples)
            pde_idxs = torch.randint(0, self.numpoints-self.num_src_samples, (int(self.num_target_samples*0.5),))

            model_states_normed[pde_idxs] = self.dynamics.coord_to_input(torch.cat((torch.zeros(
                int(self.num_target_samples*0.5), 1), target_state_samples[:int(self.num_target_samples*0.5),...]), dim=-1))[:, 1:self.dynamics.state_dim+1]
            if self.dynamics.deepReach_model in ['vanilla', 'diff']:
                num_aug_data=min(int(self.num_target_samples*0.5),int(self.num_src_samples*0.5))
                bc_idxs = -torch.randint(0, self.num_src_samples, (num_aug_data,))
                model_states_normed[bc_idxs] = self.dynamics.coord_to_input(torch.cat((torch.zeros(
                    num_aug_data, 1), target_state_samples[-num_aug_data:,...]), dim=-1))[:, 1:self.dynamics.state_dim+1]
            
        model_inputs = torch.cat((times, model_states_normed), dim=1)

        # generating MPC inputs
        if self.use_MPC:
            current_t = (self.tMax*1.0 - self.tMin) * min((self.counter) / self.counter_end, 1.0)
            if self.time_curr and not self.pretrain and (not self.PDE_curr):
                if current_t > self.MPC_dt:
                    # Find upper bound index using precomputed sorted indices
                    max_idx = bisect_right(self.MPC_inputs[self.mpc_time_sorted_indices][:, 0].cpu().numpy(), current_t)

                    if max_idx >= self.num_MPC_data_samples:
                        idxs = torch.randint(0, max_idx, (self.num_MPC_data_samples,))
                        MPC_inputs_ = self.MPC_inputs[self.mpc_time_sorted_indices[idxs]]
                        MPC_values_ = self.MPC_values[self.mpc_time_sorted_indices[idxs]]
                    else:
                        MPC_inputs_ = self.MPC_inputs[self.mpc_time_sorted_indices[:max_idx]]
                        MPC_values_ = self.MPC_values[self.mpc_time_sorted_indices[:max_idx]]

                    # Augment with MPC data (around MPC datapoints)
                    if self.aug_with_MPC_data > 0:
                        num_available = min(max_idx, self.aug_with_MPC_data)
                        idxs = torch.randint(0, num_available, (num_available,))
                        state_around_MPC_inputs=self.MPC_inputs[self.mpc_time_sorted_indices[idxs]]*1.0
                        state_around_MPC_inputs=torch.clip(state_around_MPC_inputs*(torch.randn_like(state_around_MPC_inputs)*0.01+1.0),min=-1.0,max=1.0)
                        # state_around_MPC_inputs[...,1:]=self.dynamics.clamp_state_input(state_around_MPC_inputs[...,1:])
                        model_inputs[-num_available:, ...] = state_around_MPC_inputs

                else:
                    MPC_inputs_ = torch.Tensor([0])
                    MPC_values_ = torch.Tensor([0])

            else: # if use entire horizon and a PDE curriculum
                idxs = torch.randint(0, len(self.MPC_inputs), (self.num_MPC_data_samples,))
                MPC_inputs_ = self.MPC_inputs[idxs]
                MPC_values_ = self.MPC_values[idxs]

                # Augment with MPC data
                if self.aug_with_MPC_data > 0 and not self.pretrain:
                    idxs = torch.randint(0, len(self.MPC_inputs), (self.aug_with_MPC_data,))
                    model_inputs[-self.aug_with_MPC_data:, ...] = self.MPC_inputs[idxs]

        else:
            MPC_inputs_ = torch.Tensor([0])
            MPC_values_ = torch.Tensor([0])

                    

        boundary_values = self.dynamics.boundary_fn(
            self.dynamics.input_to_coord(model_inputs)[..., 1:])
        if self.dynamics.loss_type == 'brat_hjivi':
            reach_values = self.dynamics.reach_fn(
                self.dynamics.input_to_coord(model_inputs)[..., 1:])
            avoid_values = self.dynamics.avoid_fn(
                self.dynamics.input_to_coord(model_inputs)[..., 1:])

        if self.pretrain:
            dirichlet_masks = torch.ones(model_inputs.shape[0]) > 0
        else:
            # only enforce initial conditions around self.tMin
            dirichlet_masks = (model_inputs[:, 0] == self.tMin)

        if self.pretrain:
            self.pretrain_counter += 1
        else:
            self.counter += 1

        if self.pretrain and self.pretrain_counter == self.pretrain_iters:
            self.pretrain = False

        if self.dynamics.loss_type == 'brt_hjivi' or self.dynamics.loss_type == 'frs_hjivi':
            return {'model_inputs': model_inputs, 'MPC_inputs': MPC_inputs_}, \
                {'boundary_values': boundary_values, 'dirichlet_masks': dirichlet_masks,'MPC_values': MPC_values_}
        elif self.dynamics.loss_type == 'brat_hjivi':
            return {'model_inputs': model_inputs, 'MPC_inputs': MPC_inputs_}, \
                {'boundary_values': boundary_values, 'reach_values': reach_values, 'avoid_values': avoid_values, 'dirichlet_masks': dirichlet_masks,
                 'MPC_values': MPC_values_}
        else:
            raise NotImplementedError
