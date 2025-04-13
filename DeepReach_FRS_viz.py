import numpy as np
import matplotlib.pyplot as plt
import torch
import math
from tqdm import tqdm
from dynamics import dynamics
from utils import MPC_FRS, modules

torch.manual_seed(1)
np.random.seed(1)


def plotBRTImages(costs, state_traj, x_resolution, y_resolution,x_min, x_max, y_min, y_max, ax1, ax2, color="black"):    
    BRT_img = costs.detach().cpu().numpy().reshape(x_resolution, y_resolution).T
    max_value = np.amax(BRT_img[~np.isnan(BRT_img)])
    min_value = np.amin(BRT_img[~np.isnan(BRT_img)])
    # We'll also create a grey background into which the pixels will fade
    greys = np.full((*BRT_img.shape, 3), 70, dtype=np.uint8)
    imshow_kwargs = {
        'vmax': max_value,
        'vmin': min_value,
        'cmap': 'RdYlBu',
        'extent': (x_min, x_max, y_min, y_max),
        'origin': 'lower',
    }
    ax1.imshow(greys)
    s1=ax1.imshow(BRT_img, **imshow_kwargs)
    
    
    # ax2.imshow(1*(BRT_img <= 0), cmap='bwr',
    #             origin='lower', extent=(x_min, x_max, y_min, y_max))
    X, Y = np.meshgrid(xs, ys)
    zero_contour = ax2.contour(X, 
                                Y, 
                                BRT_img, 
                                levels=[0.0],  
                                colors=color,  
                                linewidths=2,    
                                linestyles='--')  
    if state_traj is not None:
        for i in range(state_traj.shape[0]):
            ax2.plot(state_traj[i,:,0], state_traj[i,:,1], 'black')
    
    

if __name__ == "__main__":
   
    device = torch.device('cuda')
    
   
    dynamics_ = dynamics.Dubins13D()
    dynamics_.deepReach_model="vanilla"
    # dynamics_ = dynamics.Dubins3DFRS()
    
    x_res=100
    y_res=100
    plot_config = dynamics_.plot_config()
    state_test_range = dynamics_.state_test_range()
    x_min, x_max = state_test_range[plot_config['x_axis_idx']]
    y_min, y_max = state_test_range[plot_config['y_axis_idx']]
    z_min, z_max = state_test_range[plot_config['z_axis_idx']]

       
    xs = torch.linspace(x_min, x_max, x_res)
    ys = torch.linspace(y_min, y_max, y_res)
    xys = torch.cartesian_prod(xs, ys).to(device)
    initial_condition_tensor=torch.zeros(x_res*y_res, dynamics_.state_dim).to(device)
    state_slices=[0, 0, 0, 6, 15, 0.5, 1, 0, 0, 0.1, 1, -2, 0]
    initial_condition_tensor[:, :] = torch.tensor(state_slices)
    initial_condition_tensor[:, plot_config['x_axis_idx']] = xys[:, 0]
    initial_condition_tensor[:, plot_config['y_axis_idx']] = xys[:, 1]
    # initial_condition_tensor[:, plot_config['z_axis_idx']] = z_max*0.0

    

    model = modules.SingleBVPNet(in_features=dynamics_.input_dim, out_features=1, type="sine", mode='mlp',
                             final_layer_factor=1., hidden_features=512, num_hidden_layers=3, 
                             periodic_transform_fn=dynamics_.periodic_transform_fn)
    model.cuda()
    model.load_state_dict(torch.load(
            "./runs/Dubins13D_optimistic3/training/checkpoints/model_epoch_31000.pth")["model"])
    # model.load_state_dict(torch.load(
    #         "./runs/Dubins13D_optimistic/training/checkpoints/model_epoch_2000.pth")["model"])

    all_values=[]
    fig = plt.figure(figsize=(6, 6))
    fig2 = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot(1, 1, 1 )
    ax2 = fig2.add_subplot(1, 1, 1)
    colors = [
        "#1f77b4",  # muted blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#17becf",  # cyan
    ]
    for step in range(8):
        T=0.5*(step+1)
        values=torch.ones(initial_condition_tensor.shape[0])*torch.finfo().max
        vel_low=state_slices[4] + (min(state_slices[6]-math.exp(state_slices[8]), state_slices[10]-math.exp(state_slices[12])))*T
        vel_low=max(vel_low,0)
        vel_high=state_slices[4] + (max(state_slices[6]+math.exp(state_slices[8]), state_slices[10]+math.exp(state_slices[12])))*T
        vel_high=min(vel_high,30)
        
        for vel in tqdm(torch.linspace(vel_low,vel_high,25)):
            # for theta in torch.linspace(-math.pi/3,math.pi/3, 25):
            mean_angular_v=state_slices[5]*(1-T/4.5) + state_slices[9]*(T/4.5)
            theta_low=mean_angular_v*T + min(-math.exp(state_slices[7]), -math.exp(state_slices[11]))*T
            theta_high=mean_angular_v*T + max(math.exp(state_slices[7]), math.exp(state_slices[11]))*T
            # print(theta_low,theta_high, theta_high-theta_low)
            for theta in torch.linspace(theta_low,theta_high, 25):
                final_state_tensor=initial_condition_tensor.clone()
                final_state_tensor[...,2]=theta*1.0
                final_state_tensor[...,3]=vel*1.0
                

                coords=torch.cat((torch.ones(x_res*y_res,1).to(device)*T,final_state_tensor),dim=-1)
                model_results = model(
                                    {'coords': dynamics_.coord_to_input(coords).cuda()})

                states = dynamics_.input_to_coord(
                    model_results['model_in'].detach())[..., 1:]

                values_ = dynamics_.io_to_value(
                    model_results['model_in'].detach(), model_results['model_out'].squeeze(dim=-1)).detach().cpu()
                values= torch.minimum(values,values_)
        values+=2.5
        plotBRTImages(values, None, x_resolution=x_res, y_resolution=y_res,x_min=x_min,x_max=x_max,y_min=y_min, y_max=y_max,
                       ax1=ax1, ax2=ax2, color=colors[step])
    
    
    # plt.show()
    fig.savefig("./data/deepreach_heatmap.png")
    fig2.savefig("./data/deepreach_FRS.png")

