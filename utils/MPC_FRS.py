import torch
from tqdm import tqdm
import math

class MPC_FRS:
    def __init__(self, dT, horizon, receding_horizon, num_samples, dynamics_, device, num_iterative_refinement=1, style='receding'):
        self.horizon = horizon
        self.num_samples = num_samples
        self.device = device
        self.receding_horizon = receding_horizon
        self.dynamics_=dynamics_
        self.dT = dT
        self.num_iterative_refinement=num_iterative_refinement
        self.num_effective_horizon_refinement = 0
        self.style=style
  
    def get_batch_data(self, final_state_tensor, T, policy=None, t=0.0):
        '''
        Generate MPC dataset in a batch manner
        Inputs: final_state_tensor A*D_N (Batch size * State dim)
                T: MPC total horizon
                t: MPC total horizon - MPC effective horizon 
                            (the current DeepReach curriculum length / time-to-go for MPC after optimizing on H_R)
                policy: Current DeepReach model
        Outputs: 
                costs: best cost function for all final_state_tensor (A)
                state_trajs: best trajs for all final_state_tensor (A * Horizon * state_dim)
                coords: bootstrapped MPC coords after normalization (coords=[time,state])  (? * (state_dim+1))
                value_labels: bootstrapped MPC value labels  (?)
        '''
        self.T=T*1.0
        self.batch_size=final_state_tensor.shape[0]

        state_trajs, costs, num_iters = self.get_opt_trajs(final_state_tensor, policy, t)
        
        # generating MPC dataset: {..., (t, x, J), ...} 
        # TODO: we can actually bootstrap the dataset by transforming the final state using current state!
        # TODO: which means the final_state_new will be the final state relative to the current state!

        # coords=torch.zeros(self.batch_size,self.dynamics_.state_dim+1).to(self.device)
        # coords[: ,0] = self.T
        # coords[:,1:] = final_state_tensor*1.0
        # value_labels=costs


        # bootstrap version
        coords=torch.empty(0, self.dynamics_.state_dim+1).to(self.device)
        value_labels=torch.empty(0).to(self.device)
        
        for i in range(num_iters):
            coord_i=torch.zeros(self.batch_size,self.dynamics_.state_dim+1).to(self.device)
            coord_i[: ,0] = self.T - i * self.dT
            coord_i[:,1:] = final_state_tensor*1.0
            coord_i[:,1:4] = self.get_rel_frame(final_state_tensor[:,:3], state_trajs[:, i, :3])

            value_labels_i= costs

            # add to data
            coords=torch.cat((coords,coord_i),dim=0)
            value_labels=torch.cat((value_labels,value_labels_i),dim=0)

        
        ##################### only use in range labels ###################################################
        output1 = torch.all(coords[...,1:] >= self.dynamics_.state_range_[
                            :, 0]-0.01, -1, keepdim=False)
        output2 = torch.all(coords[...,1:] <= self.dynamics_.state_range_[
                            :, 1]+0.01, -1, keepdim=False)
        in_range_index = torch.logical_and(torch.logical_and(output1, output2), ~torch.isnan(value_labels))


        coords=coords[in_range_index]
        value_labels=value_labels[in_range_index]
        ###################################################################################################
        coords=self.dynamics_.coord_to_input(coords)

        return costs, state_trajs, coords.detach().cpu().clone(), value_labels.detach().cpu().clone()
    
    def get_rel_frame(self, A, B):
        # A, B = (N*3)
        A_pos = A[:, :2]*1.0  # (N, 2)
        B_pos = B[:, :2]*1.0
        B_theta = B[:, 2]*1.0  # (N,)

        # Step 1: translation
        delta = A_pos - B_pos  # (N, 2)

        # Step 2: rotation matrix using -B_theta
        cos_theta = torch.cos(-B_theta)
        sin_theta = torch.sin(-B_theta)

        # Build rotation matrices (N, 2, 2)
        R = torch.stack([
            torch.stack([cos_theta, -sin_theta], dim=1),
            torch.stack([sin_theta,  cos_theta], dim=1)
        ], dim=1)  # shape (N, 2, 2)

        # Rotate delta into B's frame
        delta_rotated = torch.bmm(R, delta.unsqueeze(-1)).squeeze(-1)  # (N, 2)

        # Step 3: compute relative heading
        theta_rel = A[:, 2] - B[:, 2]  # (N,)
        theta_rel = (theta_rel + torch.pi) % (2 * torch.pi) - torch.pi  # wrap to [-pi, pi]

        # Final relative state
        relative_state = torch.cat([delta_rotated, theta_rel.unsqueeze(1)], dim=1)  # (N, 3)
        relative_state[..., 0] += self.dynamics_.init_state_[0]
        relative_state[..., 1] += self.dynamics_.init_state_[1]
        return relative_state
    
    def get_opt_trajs(self,final_state_tensor, policy=None, t=0.0):
        '''
        Generate optimal trajs in a batch manner
        Inputs: final_state_tensor A*D_N (Batch size * State dim)
                t: MPC total horizon - MPC effective horizon 
                            (the current DeepReach curriculum length / time-to-go for MPC after optimizing on H_R)
                policy: Current DeepReach model
        Outputs: 
                
                best_trajs: best trajs for all final_state_tensor (A * Horizon * state_dim)
                lxs: l(x) along best trajs (A*H)
                num_iters: H 
        '''
        num_iters = math.ceil((self.T)/self.dT)
        self.horizon = math.ceil((self.T)/self.dT)
        
        self.incremental_horizon =  math.ceil((self.T-t)/self.dT)
        if self.style == 'direct':
            self.init_control_tensors()
            if policy is not None:
                self.num_effective_horizon_refinement=int(self.num_iterative_refinement*0.4)
                for i in range(self.num_effective_horizon_refinement):
                    self.warm_start_with_policy(final_state_tensor, policy, t) # optimize on the effective horizon first
            # optimize on the entire horizon for stability (in case that the current learned value function is not accurate)
            best_controls, best_trajs = self.get_control(
                    final_state_tensor, self.num_iterative_refinement, policy, t_remaining=t) 
            
            costs = self.dynamics_.distance_fn(best_trajs[...,-1,:], final_state_tensor)   
            return best_trajs, costs, num_iters
        elif self.style == 'receding':
            state_trajs = torch.zeros(( self.batch_size, num_iters+1, self.dynamics_.state_dim)).to(self.device)  # A*H*D
            state_trajs[:, 0, :] = self.dynamics_.get_initial_state_tensor(final_state_tensor)
            self.init_control_tensors()
            self.receiding_start=0
            for i in tqdm(range(int(num_iters/self.receding_horizon))):
                best_controls,_ = self.get_control(
                        final_state_tensor,current_state=state_trajs[:,i, :])
                for k in range(self.receding_horizon):
                    
                    state_trajs[:,i*self.receding_horizon+1+k,:] = self.get_next_step_state(
                        state_trajs[:,i*self.receding_horizon+k,:], best_controls[:, k, :])
                    self.receiding_start+=1
            costs = self.dynamics_.distance_fn(state_trajs[...,-1,:], final_state_tensor)   
            return state_trajs, costs, num_iters
        else:
            return NotImplementedError


        
 
    def warm_start_with_policy(self, final_state_tensor, policy=None, t_remaining=None):
        '''
        Generate optimal trajs in a batch manner using the DeepReach value function as the terminal cost
        Inputs: final_state_tensor A*D_N (Batch size * State dim)
                t_remaining: MPC total horizon - MPC effective horizon 
                            (the current DeepReach curriculum length / time-to-go for MPC after optimizing on H_R)
                policy: Current DeepReach model
        Outputs: 
                None
                Internally update self.control_tensors (first H_R horizon with MPC and t_remaining with DeepReach policy)
                Internally update self.warm_start_traj (for debugging purpose)
        '''
        if self.incremental_horizon>0:
            # Rollout with the incremental horizon
            state_trajs_H, permuted_controls_H = self.rollout_dynamics(final_state_tensor,start_iter=0, rollout_horizon=self.incremental_horizon)
        
            costs = self.dynamics_.cost_fn(state_trajs_H, final_state_tensor.unsqueeze(1).repeat(1, self.num_samples, 1).to(self.device))  # A * N
            # Use the learned value function for terminal cost and compute the cost function
            if t_remaining>0.0:
                traj_times=torch.ones(self.batch_size,self.num_samples,1).to(self.device)*t_remaining
                state_trajs_clamped = torch.clamp(state_trajs_H[:, :, -1, :], torch.tensor(self.dynamics_.state_test_range(
                                )).to(self.device)[..., 0], torch.tensor(self.dynamics_.state_test_range()).to(self.device)[..., 1])

                traj_coords = torch.cat(
                    (traj_times, state_trajs_clamped), dim=-1)
                traj_policy_results = policy(
                    {'coords': self.dynamics_.coord_to_input(traj_coords.to(self.device))})
                terminal_values=self.dynamics_.io_to_value(traj_policy_results['model_in'].detach(
                    ), traj_policy_results['model_out'].squeeze(dim=-1).detach())
                
                costs=terminal_values*1.0 # for FRS

            # Pick the best control and correponding traj
            best_costs, best_idx=costs.min(1) # for FRS

            expanded_idx = best_idx[...,None, None, None].expand(-1, -1, permuted_controls_H.size(2), permuted_controls_H.size(3))  

            best_controls_H = torch.gather(permuted_controls_H, dim=1, index=expanded_idx).squeeze(1) # A * H * D_u
            expanded_idx_traj = best_idx[...,None, None, None].expand(-1, -1, state_trajs_H.size(2), state_trajs_H.size(3))  
            best_traj_H= torch.gather(state_trajs_H, dim=1, index=expanded_idx_traj).squeeze(1)

            # Rollout the remaining horizon with the learned policy and update the nominal control traj
            self.control_tensors[:,:self.incremental_horizon,:]=best_controls_H*1.0
            self.warm_start_traj = self.rollout_with_policy(best_traj_H[:,-1,:],policy,self.horizon-self.incremental_horizon,self.incremental_horizon)
            self.warm_start_traj = torch.cat([best_traj_H[:,:-1,:],self.warm_start_traj],dim=1)

        else:
            # Rollout using the learned policy and update the nominal control traj
            self.warm_start_traj = self.rollout_with_policy(final_state_tensor,policy,self.horizon)

    def get_control(self, final_state_tensor, num_iterative_refinement=1, policy=None, t_remaining=None, current_state=None):
        '''
        Update self.control_tensors using perturbations
        Inputs: final_state_tensor A*D_N (Batch size * State dim)
                num_iterative_refinement: number of iterative improvement steps (re-sampling steps) in MPC
                t_remaining: MPC total horizon - MPC effective horizon 
                            (the current DeepReach curriculum length / time-to-go for MPC after optimizing on H_R)
                policy: Current DeepReach model
        '''
        if self.style == 'direct':
            if num_iterative_refinement==-1: # rollout using the policy
                best_traj = self.rollout_with_policy(final_state_tensor,policy,self.horizon)
            for i in range(num_iterative_refinement+1 - self.num_effective_horizon_refinement):
                state_trajs, permuted_controls = self.rollout_dynamics(final_state_tensor,start_iter=0,rollout_horizon=self.horizon)
                self.all_state_trajs=state_trajs.detach().cpu()*1.0
                current_controls, best_traj, best_costs = self.update_control_tensor(
                    state_trajs, permuted_controls, final_state_tensor) 
            return self.control_tensors, best_traj
        elif self.style == 'receding':
            # initial_condition_tensor: A*D
            state_trajs, permuted_controls = self.rollout_dynamics(final_state_tensor,start_iter=self.receiding_start,
                                                            rollout_horizon=self.horizon-self.receiding_start, current_state=current_state)

            current_controls, best_traj, best_costs = self.update_control_tensor(
                state_trajs, permuted_controls, final_state_tensor) 
        
            return current_controls, best_traj
        else:
            raise NotImplementedError
      
    def rollout_with_policy(self, final_state_tensor, policy, policy_horizon, policy_start_iter=0):
        '''
        Rollout traj with policy and update self.control_tensors (nominal control)
        Inputs: final_state_tensor A*D_N (Batch size * State dim)
                policy: Current DeepReach model
                policy_horizon: num steps correpond to t_remaining
                policy_start_iter: step num correpond to H_R
        '''
        state_trajs = torch.zeros((self.batch_size, policy_horizon+1, self.dynamics_.state_dim))  # A * H * D
        state_trajs = state_trajs.to(self.device, non_blocking=True)  # Move to GPU only when needed
        state_trajs[:, 0, :] = self.dynamics_.get_initial_state_tensor(final_state_tensor)
        state_trajs_clamped=state_trajs*1.0
        traj_times=torch.ones(self.batch_size,1).to(self.device)*policy_horizon*self.dT
        # update control from policy_start_iter to policy_start_iter+ policy horizon
        for k in range(policy_horizon):
            
            traj_coords = torch.cat(
                (traj_times, state_trajs_clamped[:, k, :]), dim=-1)
            traj_policy_results = policy(
                {'coords': self.dynamics_.coord_to_input(traj_coords.to(self.device))})
            traj_dvs = self.dynamics_.io_to_dv(
                traj_policy_results['model_in'], traj_policy_results['model_out'].squeeze(dim=-1)).detach()
        
            self.control_tensors[:, k+policy_start_iter, :] = self.dynamics_.optimal_control(
                traj_coords[:, 1:].to(self.device), traj_dvs[..., 1:].to(self.device))
            self.control_tensors[:, k+policy_start_iter, :]=self.dynamics_.clamp_control(state_trajs[:, k, :], self.control_tensors[:, k+policy_start_iter, :])
            state_trajs[:, k+1,:] = self.get_next_step_state(
                state_trajs[:, k, :], self.control_tensors[:, k+policy_start_iter, :])

            state_trajs_clamped[:, k+1,:] = torch.clamp(state_trajs[:, k+1,:], torch.tensor(self.dynamics_.state_test_range(
                    )).to(self.device)[..., 0], torch.tensor(self.dynamics_.state_test_range()).to(self.device)[..., 1])
            traj_times=traj_times-self.dT
        return state_trajs
        
    def update_control_tensor(self, state_trajs, permuted_controls, final_state_tensor):   
        '''
        Determine nominal controls (self.control_tensors) using permuted_controls and corresponding state trajs
        Inputs: 
                state_trajs: A*N*H*D_N (Batch size * Num perturbation * Horizon * State dim)
                permuted_controls: A*N*H*D_U (Batch size * Num perturbation * Horizon * Control dim)
        '''
        costs = self.dynamics_.cost_fn(state_trajs, final_state_tensor.unsqueeze(1).repeat(1, self.num_samples, 1).to(self.device)) # A * N
        # just use the best control
        best_costs, best_idx=costs.min(1)

        expanded_idx = best_idx[..., None, None, None].expand(-1, -1, permuted_controls.size(2), permuted_controls.size(3))  

        best_controls = torch.gather(permuted_controls, dim=1, index=expanded_idx).squeeze(1) # A * H * D_u
        if self.style == 'receding':
            self.control_tensors[:,self.receiding_start:,:]=best_controls*1.0
        else:
            self.control_tensors = best_controls*1.0
        expanded_idx_traj = best_idx[...,None, None, None].expand(-1, -1, state_trajs.size(2), state_trajs.size(3))  
        best_traj= torch.gather(state_trajs, dim=1, index=expanded_idx_traj).squeeze(1)
        
        # update controls
        current_controls = self.control_tensors[:, :self.receding_horizon, :]
        if self.style == 'receding':
            current_controls=self.control_tensors[:, self.receiding_start:self.receiding_start+self.receding_horizon, :]*1.0
        #   self.control_tensors[:, :self.horizon-self.receding_horizon,
        #                       :] = self.control_tensors[:,self.receding_horizon:, :]
        #   self.control_tensors[:, self.horizon-self.receding_horizon:, :] = self.control_init.unsqueeze(1).repeat(1,self.receding_horizon,1) # A * H_r * D_u 
        return current_controls, best_traj, best_costs
    
    
   
    def rollout_dynamics(self, final_state_tensor, start_iter, rollout_horizon, eps_var_factor=1, current_state=None):
        '''
        Rollout trajs by generating perturbed controls
        Inputs: 
                final_state_tensor A*D_N (Batch size * State dim)
                start_iter: from which step we start rolling out
                rollout_horizon: rollout for how many steps
                eps_var_factor: scaling factor for the sample variance (not being used in the paper but can be tuned if needed)
        Outputs: 
                state_trajs: A*N*H*D_N (Batch size * Num perturbation * Horizon * State dim)
                permuted_controls: A*N*H*D_U (Batch size * Num perturbation * Horizon * Control dim)
        '''
        # returns the state trajectory list and swith collision
        epsilon_tensor = torch.randn(
            self.batch_size, self.num_samples, rollout_horizon, self.dynamics_.control_dim).to(self.device)*torch.sqrt(self.dynamics_.eps_var)*eps_var_factor # A * N * H * D_u

        epsilon_tensor[:, 0, ...] = 0.0  # always include the nominal trajectory
        
        permuted_controls = self.control_tensors[:,start_iter:start_iter+rollout_horizon,:].unsqueeze(1).repeat(1, 
            self.num_samples, 1, 1) + epsilon_tensor *1.0 # A * N * H * D_u

        # clamp control
        permuted_controls = torch.clamp(permuted_controls, self.dynamics_.control_range_[..., 0], self.dynamics_.control_range_[..., 1])

        # rollout trajs
        state_trajs = torch.zeros((self.batch_size, self.num_samples, rollout_horizon+1, self.dynamics_.state_dim)).to(self.device)  # A * N * H * D
        if current_state is not None:
            initial_state_tensor=current_state*1.0
        else:
            initial_state_tensor=self.dynamics_.get_initial_state_tensor(final_state_tensor) 
        state_trajs[:, :, 0, :] = initial_state_tensor.unsqueeze(1).repeat(1, self.num_samples, 1) # A * N * D
        
        for k in range(rollout_horizon):
            permuted_controls[:, :, k, :]=self.dynamics_.clamp_control(state_trajs[:, :, k, :], permuted_controls[:, :, k, :])
            state_trajs[:, :, k+1,:]= self.get_next_step_state(
                state_trajs[:, :, k, :], permuted_controls[:, :, k, :])

        return state_trajs, permuted_controls
    
    def init_control_tensors(self, nominal_control=None):
        if nominal_control is None:
            self.control_init =self.dynamics_.control_init.unsqueeze(0).repeat(self.batch_size,1)
        else:
            self.control_init = nominal_control.clone().to(self.device)
        self.control_tensors = self.control_init.unsqueeze(1).repeat(1,self.horizon,1) # A * H * D_u
    
    def get_next_step_state(self, state, controls):
        current_dsdt = self.dynamics_.dsdt(
            state, controls, None)
        next_states= self.dynamics_.equivalent_wrapped_state(state + current_dsdt*self.dT)
        # next_states = torch.clamp(next_states, self.dynamics_.state_range_[..., 0], self.dynamics_.state_range_[..., 1])
        return next_states
