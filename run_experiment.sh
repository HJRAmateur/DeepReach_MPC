############################# Before any training  ####################################
# specify your GPU
export CUDA_VISIBLE_DEVICES = 0 # or any gpu number
############################# To verify any experiment ##################################
# To run verification
python run_experiment.py --mode test --experiment_name EXPNAME --checkpoint_toload -1 --data_step run_basic_recovery
# to plot verification
python run_experiment.py --mode test --experiment_name EXPNAME --checkpoint_toload -1 --data_step run_basic_recovery

############################################# 2D Vertical Drone ##########################################
# To train (exp time: 1.5 min on RTX 4090)
python run_experiment.py --mode train --experiment_name VD --dynamics_class VertDrone2D --tMax 1.2 --pretrain --pretrain_iters 1000 --num_epochs 10000 --counter_end 6000 --num_nl 128 --lr 3e-5 --num_iterative_refinement 10 --MPC_batch_size 100 --num_MPC_batches 10 --num_MPC_data_samples 100 --numpoints 10000 --time_till_refinement 0.24 --use_wandb --wandb_project MPC --wandb_name VD --wandb_group VertDrone2D --wandb_entity YOUR_WANDB_NAME

# To run verification
python run_experiment.py --mode test --experiment_name VD --checkpoint_toload -1 --data_step run_basic_recovery
# To plot validation. Plot available under ./runs/VD/basic_BRTs.png
python run_experiment.py --mode test --experiment_name VD --checkpoint_toload -1 --data_step plot_basic_recovery

############################################# Parameterized Vertical Drone ##########################################
# To train (exp time: 3 min on RTX 4090)
python run_experiment.py --mode train --experiment_name PVD --dynamics_class ParameterizedVertDrone2D --tMax 1.2 --pretrain --pretrain_iters 1000 --num_epochs 13000 --counter_end 10000 --num_nl 128 --gravity 9.8 --input_multiplier 12 --input_magnitude_max 1 --lr 5e-5 --num_iterative_refinement 10 --MPC_batch_size 1000 --num_MPC_batches 5 --num_MPC_data_samples 500 --numpoints 20000 --epochs_til_ckpt 1000 --use_wandb --wandb_project MPC --wandb_name PVD --wandb_group ParameterizedVertDrone2D --wandb_entity YOUR_WANDB_NAME

# To train vanilla DeepReach, not working even with more epochs (specify --not_use_MPC)
python run_experiment.py --mode train --experiment_name PVD_vanilla --dynamics_class ParameterizedVertDrone2D --tMax 1.2 --pretrain --pretrain_iters 1000 --num_epochs 21000 --counter_end 20000 --num_nl 128 --gravity 9.8 --input_multiplier 12 --input_magnitude_max 1 --lr 3e-5 --numpoints 10000 --epochs_til_ckpt 1000 --use_wandb --wandb_project MPC --wandb_name PVD_vanilla --wandb_group ParameterizedVertDrone2D --wandb_entity YOUR_WANDB_NAME --not_use_MPC

# Training vanilla DeepReach with even more epochs does not work T_T (exp time: 10 min on RTX 4090)
python run_experiment.py --mode train --experiment_name PVD_vanilla --dynamics_class ParameterizedVertDrone2D --tMax 1.2 --pretrain --pretrain_iters 1000 --num_epochs 101000 --counter_end 100000 --num_nl 128 --gravity 9.8 --input_multiplier 12 --input_magnitude_max 1 --lr 3e-5 --numpoints 10000 --epochs_til_ckpt 1000 --use_wandb --wandb_project MPC --wandb_name PVD_vanilla --wandb_group ParameterizedVertDrone2D --wandb_entity YOUR_WANDB_NAME --not_use_MPC

############################################# Dubins3D Reach problem ##########################################
# to train (exp time: 3 min)
python run_experiment.py --mode train --experiment_name Dubins --dynamics_class Dubins3D --tMax 1 --pretrain --pretrain_iters 1000 --num_epochs 13000 --counter_end 10000 --num_nl 128 --set_mode reach --lr 5e-5 --num_iterative_refinement 10 --MPC_batch_size 1000 --num_MPC_batches 3 --num_MPC_data_samples 500 --numpoints 5000 --use_wandb --wandb_project MPC --wandb_name Dubins --wandb_group Dubins3D --wandb_entity YOUR_WANDB_NAME

# to train with VANILLA DeepReach (use --not_use_MPC flag)
python run_experiment.py --mode train --experiment_name Dubins_vanilla --dynamics_class Dubins3D --tMax 1 --pretrain --pretrain_iters 1000 --num_epochs 11000 --counter_end 10000 --num_nl 128 --set_mode reach --lr 5e-5 --num_iterative_refinement 10 --MPC_batch_size 1000 --num_MPC_batches 3 --num_MPC_data_samples 500 --numpoints 5000 --use_wandb --wandb_project MPC --wandb_name Dubins_vanilla --wandb_group Dubins3D --wandb_entity YOUR_WANDB_NAME --not_use_MPC

# to learn the safety problem
python run_experiment.py --mode train --experiment_name Dubins_avoid --dynamics_class Dubins3D --tMax 1 --pretrain --pretrain_iters 1000 --num_epochs 13000 --counter_end 10000 --num_nl 128 --set_mode avoid --lr 5e-5 --num_iterative_refinement 10 --MPC_batch_size 1000 --num_MPC_batches 3 --num_MPC_data_samples 500 --numpoints 5000 --use_wandb --wandb_project MPC --wandb_name Dubins_avoid --wandb_group Dubins3D --wandb_entity zeyuanfe

############################################## Quadrotor ##########################################
# To train (exp time: 3h on RTX 4090)
python run_experiment.py --mode train --experiment_name Quadrotor --use_wandb --wandb_entity YOUR_WANDB_NAME --wandb_project MPC --wandb_name Quadrotor --wandb_group Quadrotor --dynamics_class Quadrotor --tMax 1 --pretrain --pretrain_iters 1000 --num_epochs 104000 --counter_end 100000 --num_nl 512 --collisionR 0.5 --collective_thrust_max 20 --set_mode avoid --lr 2e-5 --num_MPC_batches 20 

############################################## F1 ##########################################
# To train (exp time: 5h on RTX 4090)
python run_experiment.py --mode train --experiment_name F1 --use_wandb --wandb_project MPC --wandb_name F1 --wandb_group F1tenth --wandb_entity YOUR_WANDB_NAME --dynamics_class F1tenth --tMax 1.0 --num_epochs 230000 --counter_end 200000 --num_nl 512 --lr 2e-5 --num_MPC_data_samples 10000 --val_time_resolution 6 --MPC_dt 0.02 --numpoints 30000 --pretrain --pretrain_iters 20000 --num_MPC_batches 30

############################################## 40D system #########################################################
# To train (exp time: 2h on RTX 4090)
python run_experiment.py --mode train --experiment_name LessLinear40D --use_wandb --wandb_project MPC --wandb_name LessLinear40D --wandb_group LessLinearND --wandb_entity YOUR_WANDB_NAME --dynamics_class LessLinearND --tMax 1.0 --pretrain --pretrain_iters 1000 --num_epochs 120000 --counter_end 100000 --num_nl 512 --lr 1e-5 --numpoints 15000 --N 40 --goalR 0.25 --gamma 20 --mu 0 --alpha 0 --val_time_resolution 6 --num_MPC_data_samples 1000 --MPC_batch_size 5000 --num_MPC_batches 12 --MPC_dt 0.034 
# note that you need to run validate_ND to visualize the interesting BRT slices for this
# To run verification
python run_experiment.py --mode test --experiment_name LessLinear40D --checkpoint_toload -1 --data_step run_basic_recovery
# To plot validation. Plot available under ./runs/VD/basic_BRTs.png
python run_experiment.py --mode test --experiment_name LessLinear40D --checkpoint_toload -1 --data_step plot_ND



############################################# Dubins 9D BRT ############################################
python run_experiment.py --mode train --experiment_name Dubins9D --dynamics_class Dubins9D --tMax 3 --pretrain --pretrain_iters 3000 --num_epochs 30000 --counter_end 25000 --num_nl 256 --lr 3e-5 --num_iterative_refinement 10 --MPC_batch_size 1000 --num_MPC_batches 15 --num_MPC_data_samples 500 --numpoints 20000 --use_wandb --wandb_project MPC --wandb_name Dubins9D --wandb_group Dubins9D --wandb_entity zeyuanfe --MPC_dt 0.1 --time_till_refinement 1.0

############################################# Dubins 10D FRS ############################################
python run_experiment.py --mode train --experiment_name Dubins10D_vanilla3 --dynamics_class Dubins10D --tMax 4.5 --num_epochs 100000 --counter_end 90000 --num_nl 512 --lr 3e-6 --MPC_batch_size 10000 --num_MPC_batches 60 --num_MPC_data_samples 5000 --numpoints 100000 --use_wandb --wandb_project MPC --wandb_name Dubins10D_vanilla3 --wandb_group Dubins10D --wandb_entity zeyuanfe --MPC_dt 0.1 --not_refine_dataset --minWith FRS --MPC_receding_horizon 1 --deepReach_model vanilla --num_target_samples 3000 --MPC_loss_type l1 --MPC_decay_scheme exponential --pretrained_model Dubins10D_vanilla  --MPC_finetune_lambda 10 --MPC_style receding



python run_experiment.py --mode train --experiment_name Dubins10D_PDEcurr_conserv2 --dynamics_class Dubins10D --tMax 4 --num_epochs 10000 --counter_start 29999 --counter_end 30000 --num_nl 512 --lr 1e-6 --MPC_batch_size 10000 --num_MPC_batches 60 --num_MPC_data_samples 3000 --numpoints 30000 --use_wandb --wandb_project MPC --wandb_name Dubins10D_PDEcurr_conserv2 --wandb_group Dubins10D --wandb_entity zeyuanfe --MPC_dt 0.1 --not_refine_dataset --minWith FRS --MPC_receding_horizon 1 --deepReach_model vanilla --num_target_samples 3000 --MPC_loss_type l2 --MPC_importance_init 0.003 --MPC_decay_scheme exponential --PDE_curr --MPC_finetune_lambda 100 --pretrained_model Dubins10D_PDEcurr_conserv_vanilla --MPC_style receding



############################################# Dubins 13D FRS ############################################
python run_experiment.py --mode train --experiment_name Dubins13D_vanilla --dynamics_class Dubins13D --tMax 4.5 --pretrain --pretrain_iters 10000 --num_epochs 100000 --counter_end 80000 --num_nl 512 --lr 1e-5 --MPC_batch_size 10000 --num_MPC_batches 60 --num_MPC_data_samples 5000 --numpoints 100000 --use_wandb --wandb_project MPC --wandb_name Dubins13D_vanilla --wandb_group Dubins13D --wandb_entity zeyuanfe --MPC_dt 0.1 --not_refine_dataset --minWith FRS --MPC_receding_horizon 1 --deepReach_model vanilla --num_target_samples 3000 --MPC_loss_type l1 --MPC_decay_scheme exponential --MPC_finetune_lambda 10 --MPC_style receding


python run_experiment.py --mode train --experiment_name Dubins13D_vanilla3 --dynamics_class Dubins13D --tMax 4.5  --num_epochs 10000 --counter_start 79999 --counter_end 80000 --num_nl 512 --lr 1e-6 --MPC_batch_size 10000 --num_MPC_batches 60 --num_MPC_data_samples 5000 --numpoints 100000 --use_wandb --wandb_project MPC --wandb_name Dubins13D_vanilla3 --wandb_group Dubins13D --wandb_entity zeyuanfe --MPC_dt 0.1 --not_refine_dataset --minWith FRS --MPC_receding_horizon 1 --deepReach_model vanilla --num_target_samples 3000 --MPC_loss_type l1 --MPC_decay_scheme exponential --MPC_finetune_lambda 10 --MPC_style receding --pretrained_model Dubins13D --MPC_data_path ./data/dubins13D_20aug

python run_experiment.py --mode train --experiment_name Dubins13D_conservative3 --dynamics_class Dubins13D --tMax 4.5  --num_epochs 40000 --counter_end 30000 --num_nl 512 --lr 6e-6 --MPC_batch_size 10000 --num_MPC_batches 60 --num_MPC_data_samples 20000 --numpoints 100000 --use_wandb --wandb_project MPC --wandb_name Dubins13D_conservative3 --wandb_group Dubins13D --wandb_entity zeyuanfe --MPC_dt 0.1 --not_refine_dataset --minWith FRS --MPC_receding_horizon 1 --deepReach_model vanilla --num_target_samples 3000 --MPC_loss_type l1 --MPC_decay_scheme exponential --MPC_finetune_lambda 3 --MPC_style receding --pretrained_model Dubins13D_conservative --MPC_data_path ./data/dubins13D
############################################# Dubins 3D FRS ############################################
python run_experiment.py --mode train --experiment_name Dubins3DFRS --dynamics_class Dubins3DFRS --tMax 1 --pretrain --pretrain_iters 2000 --num_epochs 30000 --counter_end 27000 --num_nl 256 --lr 3e-5 --num_iterative_refinement 10 --MPC_batch_size 1000 --num_MPC_batches 10 --num_MPC_data_samples 500 --numpoints 20000 --use_wandb --wandb_project MPC --wandb_name Dubins3DFRS --wandb_group Dubins3DFRS --wandb_entity zeyuanfe --MPC_dt 0.02 --not_refine_dataset --minWith FRS --MPC_finetune_lambda 10 --MPC_receding_horizon 1 --deepReach_model vanilla

python run_experiment.py --mode train --experiment_name Dubins3DFRS_vanilla --dynamics_class Dubins3DFRS --tMax 1 --pretrain --pretrain_iters 1000 --num_epochs 30000 --counter_end 29000 --num_nl 256 --lr 3e-5 --num_iterative_refinement 10 --MPC_batch_size 1000 --num_MPC_batches 15 --num_MPC_data_samples 500 --numpoints 20000 --use_wandb --wandb_project MPC --wandb_name Dubins3DFRS_vanilla --wandb_group Dubins3DFRS --wandb_entity zeyuanfe --MPC_dt 0.02 --time_till_refinement 1.0 --minWith FRS --not_use_MPC 