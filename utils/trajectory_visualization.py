import plotly.graph_objects as go
import plotly as plt
import numpy as np

# start_point = np.array([[4.1], [0], [1.5]])
# goal_points = np.array([[3.0, 3.0, 1.0], [4.1, 5.5, 1.5], [7.0, 1.0, 1.0]])
# cylinder_infos = [[2.8, 1.5, 0.4], [5.4, 1.5, 0.40], [1.5, 2.8, 0.40], [4.1, 2.8, 0.40], [6.7, 2.8, 0.40], [2.8, 4.1, 0.40], [5.4, 4.1, 0.40]]


def gen_sphere(x, y, z, radius, resolution=100):
    """Return the coordinates for plotting a sphere centered at (x,y,z)
       j is a special char in this context denoting the number of grid points
    """
    u, v = np.mgrid[0:2*np.pi:resolution*2j, 0:np.pi:resolution*1j]
    X = radius * np.cos(u) * np.sin(v) + x
    Y = radius * np.sin(u) * np.sin(v) + y
    Z = radius * np.cos(v) + z
    return (X, Y, Z)


def data_for_cylinder_along_z(center_x, center_y, radius, height_z):
    z = np.linspace(-height_z/2.0, height_z/2.0, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius*np.cos(theta_grid) + center_x
    y_grid = radius*np.sin(theta_grid) + center_y
    return x_grid, y_grid, z_grid


def visualize_env(out_traj, start_pos, cylinder_info, goal_points, add_tolerance_region=True, tol=1.0, fname=None):

    # generate cylindrical surfaces plots
    cylinder_surfaces = []

    for i in range(len(cylinder_info)):
        Xc, Yc, Zc = data_for_cylinder_along_z(
            cylinder_info[i][0], cylinder_info[i][1], cylinder_info[i][2], 3)
        cl = plt.colors.DEFAULT_PLOTLY_COLORS[i % 6]
        cylinder_surfaces.append(go.Surface(
            x=Xc, y=Yc, z=Zc, colorscale=[cl, cl], showscale=False))

    # plot trajectory
    conv_traj = out_traj
    traj = go.Scatter3d(
        x=conv_traj[:, 0],
        y=conv_traj[:, 1],
        z=conv_traj[:, 2],
        mode='lines',
        marker=dict(color='red'),
        name='Quadrotor trajectory'  # Add a name for the legend
    )

    # plot and label goal points
    sig_points = go.Scatter3d(
        x=goal_points[:, 0],
        y=goal_points[:, 1],
        z=goal_points[:, 2],
        mode='markers+text',
        marker=dict(
            color='yellow',
            symbol='diamond',
            size=7
        ),
        text=['goal'+str(i+1) for i in range(goal_points.shape[0])]
    )

    # plot starting point
    start_point = go.Scatter3d(
        x=start_pos[0],
        y=start_pos[1],
        z=start_pos[2],
        mode='markers+text',
        marker=dict(
            color='blue',
            symbol='circle',
            size=7
        ),
        text=['start']
    )

    # format axes
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='X', range=[-1, 8.2]),
            yaxis=dict(title='Y', range=[-1, 8.2]),
            zaxis=dict(title='Z', range=[-5, 5]),
            aspectratio=dict(x=1, y=1, z=0.75)
        )
    )

    # generate spheres showing tolerance around goals
    tol_spheres = []
    cl0 = plt.colors.DEFAULT_PLOTLY_COLORS[0]

    if add_tolerance_region:
        for goal in goal_points:
            sphere_x, sphere_y, sphere_z = gen_sphere(
                x=goal[0], y=goal[1], z=goal[2], radius=tol)
            sphere_surf = go.Surface(
                x=sphere_x, y=sphere_y, z=sphere_z, opacity=0.25, showscale=False, colorscale=[cl0, cl0])
            tol_spheres.append(sphere_surf)

    # generate figure
    fig = go.Figure(data=cylinder_surfaces +
                    [traj, sig_points, start_point] + tol_spheres, layout=layout)
    fig.update_layout(title='Trajectory through obstacles', autosize=False,
                      width=1000, height=500,
                      margin=dict(l=50, r=50, b=50, t=50))
    if fname is None:
        fig.show()
    else:
        img_path = "imgs/%s.html" % fname
        fig.write_html(img_path)
        # fig.show()
        # plt.close('all')
