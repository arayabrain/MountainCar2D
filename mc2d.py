# 1.0 - Acer 2017/07/27 13:27

import numpy as np
from numpy import array as npa
import pandas as pd
import matplotlib.pyplot as plt


def dist(a, b):
    return np.sum((a - b) ** 2, axis=1) ** (1 / 2)


def symRange(v, centre=0):
    v -= centre
    if np.abs(np.max(v)) > np.abs(np.min(v)):
        lim = npa([-np.abs(np.max(v)), np.abs(np.max(v))])
    else:
        lim = npa([-np.abs(np.min(v)), np.abs(np.min(v))])

    lim += centre
    return lim


class MountainCar2D:
    def __init__(self, dt=0.1, map_size=(1, 1), minDist=0.1):
        """
        You can add/locate attractors/repulsor and change initial state after the object instance creation:
        env = ForceField()
        
        # attractor: [x, y, force]
        # force > 0: attraction, Force < 0: repulsion         
        env.attractors = npa([[0.70, 0.70, -0.02],
                                        [0.60, 0.60, -0.02],
                                        [0.78, 0.58, -0.02],  
                                        [0.20, 0.25,  0.04]])

         # initial state  x, y, vx, vy
        env.initState = npa([25, 30, 0, 0])
        
        :param dt: simulation time step
        :param map_size: map size
        :param minDist: virual miminal distance between agent and attractors
        """
        self.dt = dt
        self.map_size = map_size
        self.minDist = minDist

        self.attractors = npa([[0.70, 0.70, -0.02],
                               [0.60, 0.60, -0.02],
                               [0.78, 0.58, -0.02],
                               [0.20, 0.25,  0.04]])

        self.initState = npa([0.25, 0.30, 0, 0])  # initial state  x, y, vx, vy
        self.state = self.initState

    def reset(self):
        self.state = self.initState

    def step(self, a):
        """
        :param a: action. numpy array (x, y)
        :return: new state 
        """

        # arrange array
        state = self.state.astype(np.float64)
        state = state[None]
        a = a[None]

        # Compute distance
        d = dist(state[:, 0:2], self.attractors[:, 0:2])[:, None]
        d_min = np.tile(npa([self.minDist]), d.shape[0])[:, None]
        d = np.max(np.hstack([d, d_min]), axis=1)

        # compute force
        F = ((1 / d ** 2) * self.attractors[:, 2]) / d
        Fxy = F[:, None] * (self.attractors[:, 0:2] - state[:, 0:2])
        Fxy_sum = np.sum(Fxy, axis=0) + a

        # update V
        Vxy = state[:, 2:4] + Fxy_sum * self.dt
        state[:, 2:4] = Vxy

        # update X
        state[:, 0:2] += state[:, 2:4] * self.dt

        # define boundary behaviors
        if state[:, 0] < 0:
            state[:, 0] = 0
            state[:, 2] = 0

        if state[:, 1] < 0:
            state[:, 1] = 0
            state[:, 3] = 0

        if state[:, 0] > self.map_size[0]:
            state[:, 0] = self.map_size[0]
            state[:, 2] = 0

        if state[:, 1] > self.map_size[1]:
            state[:, 1] = self.map_size[1]
            state[:, 3] = 0

        self.state = state.squeeze()
        return self.state

    def plot(self):
        """ Plot the current enviroment """
        colorLim = symRange(self.attractors[:, 2])
        plt.scatter(self.attractors[:, 0], self.attractors[:, 1],
                    s=60, c=self.attractors[:, 2], edgecolors='grey', cmap='bwr_r', vmin=colorLim[0], vmax=colorLim[1])
        plt.plot(self.state[0], self.state[1], 'o', c='seagreen')
        plt.xlim([0, self.map_size[0]])
        plt.ylim([0, self.map_size[1]])

    def find_uniqueGrid(self, hist_xy, nBin=(21, 21)):
        binx = np.linspace(0, self.map_size[0], nBin[0])
        biny = np.linspace(0, self.map_size[1], nBin[1])
        ix = np.digitize(hist_xy[:, 0], binx) - 1
        iy = np.digitize(hist_xy[:, 1], biny) - 1
        ixy = npa([ix, iy]).transpose()
        ixy_unique = np.vstack({tuple(row) for row in ixy})
        xy_unique = npa([binx[ixy_unique[:, 0]], binx[ixy_unique[:, 1]]]).transpose()
        return ixy_unique, xy_unique

    def gridVisitingFreq(self, hist_xy, nBin=(21, 21)):
        """ count xy visiting time"""
        binx = np.linspace(0, self.map_size[0], nBin[0])
        biny = np.linspace(0, self.map_size[1], nBin[1])
        ix = np.digitize(hist_xy[:, 0], binx) - 1
        iy = np.digitize(hist_xy[:, 1], biny) - 1
        df = pd.DataFrame({'ix': ix, 'iy': iy, 'c': 1})
        c = df.groupby(['ix', 'iy']).count()
        cc = npa(c)
        cComb = list(c.index)
        ic = np.hstack([cComb, cc])

        icReturn = ic.astype(np.float)
        icReturn[:, 0] = binx[ic[:, 0]]
        icReturn[:, 1] = biny[ic[:, 1]]
        return icReturn

    def cal_coverageRate(self, hist_xy, nBin=(21, 21)):
        """
        calculate coverage rate from position hisotyr (x, y)
        :param hist_xy: 2D numpy array: row: sample; column: x, y
        :param nBin: number of bins for each dimantion to discretize the whole map
        :return: rate
        """
        ixy_unique, _ = self.find_uniqueGrid(hist_xy, nBin)
        coverageRate = ixy_unique.shape[0] / (nBin[0] * nBin[1])
        return coverageRate

    def plot_coverage(self, hist_xy, nBin=(21, 21), marker_size=None):
        """
        Plot visited xy areas
        :param hist_xy: 
        :param nBin: 
        :param marker_size: 
        :return: 
        """
        _, xy_unique = self.find_uniqueGrid(hist_xy, nBin)
        if marker_size is None:
            marker_size = 60
        plt.scatter(xy_unique[:, 0], xy_unique[:, 1],
                    marker='s', c='lightgray', edgecolors='silver', s=marker_size, alpha=0.7)
        plt.xlim([0, self.map_size[0]])
        plt.ylim([0, self.map_size[1]])

    def plot_coverage_freq(self, hist_xy, nBin=(21, 21), marker_size=None):
        ic = self.gridVisitingFreq(hist_xy, nBin)
        if marker_size is None:
            marker_size = 60
        plt.scatter(ic[:, 0], ic[:, 1],
                    marker='s', c=ic[:, 2], edgecolors='silver', s=marker_size, alpha=0.6, cmap='YlGn')
        plt.xlim([0, self.map_size[0]])
        plt.ylim([0, self.map_size[1]])

    def demo(self, nStep=1000, isRandomWalk=False, isUsingJupyter=False, pause=0.01):
        plt.ion()
        fig = plt.figure(figsize=(5, 5))
        h_state = self.state[None]

        for i in range(nStep):
            plt.cla()
            self.plot_coverage_freq(h_state[:, 0:2])
            self.plot()
            plt.title(i)

            if isUsingJupyter:
                fig.canvas.draw()
            else:
                plt.pause(pause)

            if isRandomWalk:
                self.step(np.random.uniform(-100, 100, [1, 2]))
            else:
                self.step(0)

            h_state = np.vstack([h_state, self.state[None]])
