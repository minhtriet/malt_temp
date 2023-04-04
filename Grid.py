# import gin-config
import numpy as np

class Grid:

    def __init__(self, x_0=-1, L=1, T=1, X_DIM=1, T_DIM=1):
        """
        Training grid
        Regular grid
        Bounding grid
        Parameters
        ----------
        L
        T
        """
        # Generating the training grid

        x_interval = [x_0, L]
        t_interval = [0, T]

        intervals = [x_interval, t_interval]

        intv_array = np.vstack(intervals).T

        # Regular grid
        x_0, x_L = x_interval
        t_0, t_L = t_interval
        dx = (x_L - x_0) / X_DIM
        dt = (t_L - t_0) / T_DIM

        grid = np.mgrid[t_0 + dt : t_L + dt : dt, x_0:x_L:dx]

        data = np.hstack([grid[1].flatten()[:, None], grid[0].flatten()[:, None]])

        data_init = np.linspace(*x_interval, X_DIM)
        u_init = (data_init**2) * np.cos(np.pi * data_init)[:, None]

        # Boundary grids
        data_boundary_x0 = np.hstack(
            [
                x_interval[0] * np.ones((T_DIM, 1)),
                np.linspace(*t_interval, T_DIM)[:, None],
            ]
        )

        data_boundary_xL = np.hstack(
            [
                x_interval[-1] * np.ones((T_DIM, 1)),
                np.linspace(*t_interval, T_DIM)[:, None],
            ]
        )

        data_boundary_t0 = np.hstack(
            [
                np.linspace(*x_interval, X_DIM)[:, None],
                t_interval[0] * np.ones((X_DIM, 1)),
            ]
        )
    def visualize(self):
        import matplotlib.pyplot as plt
        # Visualizing the training mesh
        plt.scatter(*np.split(data, 2, axis=1))
        plt.scatter(*np.split(data_boundary_x0, 2, axis=1))
        plt.scatter(*np.split(data_boundary_xL, 2, axis=1))
        plt.scatter(*np.split(data_boundary_t0, 2, axis=1))

        plt.show()
        plt.close()
