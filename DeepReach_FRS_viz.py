import numpy as np
import matplotlib.pyplot as plt
import torch
import math
from tqdm import tqdm
from dynamics import dynamics
from utils import MPC_FRS, modules

torch.manual_seed(1)
np.random.seed(1)


def plotBRTImages(costs, state_traj, x_resolution, y_resolution,x_min, x_max, y_min, y_max):
    fig = plt.figure(figsize=(6, 6))
    fig2 = plt.figure(figsize=(6, 6))

        
    ax = fig.add_subplot(1, 1, 1 )
    
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
    ax.imshow(greys)
    s1=ax.imshow(BRT_img, **imshow_kwargs)
    fig.colorbar(s1)
    ax2 = fig2.add_subplot(1, 1, 1)

    ax2.imshow(1*(BRT_img <= 0), cmap='bwr',
                origin='lower', extent=(x_min, x_max, y_min, y_max))
    if state_traj is not None:
        for i in range(state_traj.shape[0]):
            ax2.plot(state_traj[i,:,0], state_traj[i,:,1], 'black')
    fig.savefig("./data/deepreach_heatmap.png")
    fig2.savefig("./data/deepreach_FRS.png")
    

if __name__ == "__main__":
   
    device = torch.device('cuda')
    
   
    dynamics_ = dynamics.Dubins10D()
    dynamics_.deepReach_model="vanilla"
    # dynamics_ = dynamics.Dubins3DFRS()
    T = 4
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
    initial_condition_tensor[:, :] = torch.tensor(plot_config['state_slices'])
    initial_condition_tensor[:, plot_config['x_axis_idx']] = xys[:, 0]
    initial_condition_tensor[:, plot_config['y_axis_idx']] = xys[:, 1]
    initial_condition_tensor[:, plot_config['z_axis_idx']] = z_max*0.0

    

    model = modules.SingleBVPNet(in_features=dynamics_.input_dim, out_features=1, type="sine", mode='mlp',
                             final_layer_factor=1., hidden_features=512, num_hidden_layers=3, 
                             periodic_transform_fn=dynamics_.periodic_transform_fn)
    model.cuda()
    model.load_state_dict(torch.load(
            "./runs/Dubins10D_vanilla/training/checkpoints/model_final.pth")["model"])
    # model.load_state_dict(torch.load(
    #         "./runs/Dubins10D_PDEcurr_conserv2/training/checkpoints/model_final.pth")["model"])

    values=torch.ones(initial_condition_tensor.shape[0])*torch.finfo().max
    for vel in tqdm(torch.linspace(1,30,15)):
        for theta in torch.linspace(-math.pi,math.pi, 15):
            final_state_tensor=initial_condition_tensor.clone()
            final_state_tensor[...,2]=theta*1.0
            final_state_tensor[...,3]=vel*1.0
            
            final_state_tensor[...,4]=6
            final_state_tensor[...,5]=-0
            final_state_tensor[...,6]=-2
            final_state_tensor[...,7]=-1
            final_state_tensor[...,8]=0.
            coords=torch.cat((torch.ones(x_res*y_res,1).to(device)*T,final_state_tensor),dim=-1)
            model_results = model(
                                {'coords': dynamics_.coord_to_input(coords).cuda()})

            states = dynamics_.input_to_coord(
                model_results['model_in'].detach())[..., 1:]

            values_ = dynamics_.io_to_value(
                model_results['model_in'].detach(), model_results['model_out'].squeeze(dim=-1)).detach().cpu()
            values= torch.minimum(values,values_)
    values-=0.08
    # final_state_tensor=initial_condition_tensor.clone()
    # final_state_tensor[...,7]=0.2
    # coords=torch.cat((torch.ones(x_res*y_res,1).to(device)*T,final_state_tensor),dim=-1)
    # model_results = model(
    #                     {'coords': dynamics_.coord_to_input(coords).cuda()})

    # states = dynamics_.input_to_coord(
    #     model_results['model_in'].detach())[..., 1:]

    # values = dynamics_.io_to_value(
    #     model_results['model_in'].detach(), model_results['model_out'].squeeze(dim=-1)).detach().cpu()

   

    plotBRTImages(values, None, x_resolution=x_res, y_resolution=y_res,x_min=x_min,x_max=x_max,y_min=y_min, y_max=y_max)
    plt.show()

