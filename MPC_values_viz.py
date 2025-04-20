import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from dynamics import dynamics
from utils import MPC, modules
import math
import os
# mpl.use('Agg')
torch.manual_seed(1)
np.random.seed(1)

ROLLOUT_NUM = 100


def plotBRTImages(costs, x_resolution, y_resolution, x_min, x_max, y_min, y_max):
    fig = plt.figure(figsize=(6, 6))
    fig2 = plt.figure(figsize=(6, 6))

    ax = fig.add_subplot(1, 1, 1)

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
    s1 = ax.imshow(BRT_img, **imshow_kwargs)
    fig.colorbar(s1)
    ax2 = fig2.add_subplot(1, 1, 1)

    ax2.imshow(1*(BRT_img <= 0), cmap='bwr',
               origin='lower', extent=(x_min, x_max, y_min, y_max))
    ensure_folder("./data/")
    fig.savefig("./data/heatmap.png")
    fig2.savefig("./data/BRT.png")


def ensure_folder(path):
    """
    Checks if a folder exists; creates it if it doesn't.

    Args:
        path (str): Path to the folder.
    """
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    dynamics_ = dynamics.ParameterizedVertDrone2D(9.8, 12.0, 1.0)
    T = 1.2
    x_res = 100
    y_res = 100
    plot_config = dynamics_.plot_config()
    state_test_range = dynamics_.state_test_range()
    x_min, x_max = state_test_range[plot_config['x_axis_idx']]
    y_min, y_max = state_test_range[plot_config['y_axis_idx']]
    z_min, z_max = state_test_range[plot_config['z_axis_idx']]

    xs = torch.linspace(x_min, x_max, x_res)
    ys = torch.linspace(y_min, y_max, y_res)
    xys = torch.cartesian_prod(xs, ys).to(device)
    initial_condition_tensor = torch.zeros(x_res*y_res, dynamics_.state_dim)
    initial_condition_tensor[:, :] = torch.tensor(plot_config['state_slices'])
    initial_condition_tensor[:, plot_config['x_axis_idx']] = xys[:, 0]
    initial_condition_tensor[:, plot_config['y_axis_idx']] = xys[:, 1]
    initial_condition_tensor[:, plot_config['z_axis_idx']] = z_max*0.5

    mpc = MPC.MPC(horizon=None, receding_horizon=1, dT=0.02, num_samples=100,
                  dynamics_=dynamics_, device=device, mode="MPC", sample_mode="gaussian",
                  style='direct', num_iterative_refinement=10)

    costs = []
    for i in tqdm(range(4)):
        costs0, state_trajs, _, _ = mpc.get_batch_data(
            initial_condition_tensor[i*2500:(i+1)*2500, ...], T)
        costs.append(costs0)
    costs = torch.cat(costs, dim=0)
    plotBRTImages(costs, x_resolution=x_res, y_resolution=y_res,
                  x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
    plt.show()
__all__ = ['run_quadrotor_mppi']
