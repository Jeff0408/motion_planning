# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
import numpy as np

from src.rrt.rrt import RRT
from src.search_space.search_space import SearchSpace
from src.utilities.plotting import Plot

X_dimensions = np.array([(0.2, 4.7), (0.2, 4.7), (0, 20)])  # dimensions of Search Space
# obstacles
'''
#maze
Obstacles = np.array(
    [(-10.0, -10.0, 0.0, -9.0, 10.0, 6.0),(-9.0, -10.0, 0.0, 9.0, -9.0, 6.0),(9.0, -10.0, 0.0, 10.0, 10.0, 6.0),(-8.0, 9.0, 0.0, 9.0, 10.0, 6.0),
     (-8.0, -8.0, 0.0, -7.0, 8.0, 6.0),(-7.0, -8.0, 0.0, 6.0, -7.0, 6.0),(6.0, -8.0, 0.0, 7.0, 0.0, 6.0),(6.0, 1.0, 0.0, 7.0, 8.0, 6.0),(-7.0, 7.0, 0.0, 6.0, 8.0, 6.0),
     (-5.0, 1.0, 0.0, -4.0, 5.0, 6.0),(-5.0, -5.0, 0.0, -4.0, 0.0, 6.0),(-4.0, -5.0, 0.0, 3.0, -4.0, 6.0),(3.0, -5.0, 0.0, 4.0, 5.0, 6.0),(-4.0, 4.0, 0.0, 3.0, 5.0, 6.0),
     (-3.0, -3.0, 0.0, -2.0, 3.0, 6.0),(-2.0, -3.0, 0.0, -1.0, -2.0, 6.0),(0.0, -3.0, 0.0, 1.0, -2.0, 6.0),(1.0, -3.0, 0.0, 2.0, 3.0, 6.0),(-2.0, 2.0, 0.0, -1.0, 3.0, 6.0),(0.0, 2.0, 0.0, 1.0, 3.0, 6.0)])
#monza
Obstacles = np.array(
    [(1.0,  0.0,  0.0, 1.1,  19.0, 5.0),(2.1,  1.0,  0.0, 2.2,  20.0, 5.0),(3.2,  0.0,  0.0, 3.3,  19.0, 5.0)])

#window
Obstacles = np.array(
    [(0,  2,  0,  10,  2.5,  1.5),(0,  2,  4.5,  10,  2.5,  6),(0,  2,  1.5,  3,  2.5,  4.5),(7,  2,  1.5,  10,  2.5,  4.5),
     (3,  0,  2.4,  7,  0.5,  4.5),(0,  15,  0,  10,  20,  1),(0,  15,  1,  10,  16,  3.5),(0,  18,  4.5,  10,  19,  6)])


#flappy bird
Obstacles = np.array(
    [(3.10, 0.0, 2.10, 3.90, 5.0, 6.0),(9.10, 0.0, 2.10, 9.90, 5.0, 6.0),(15.10, 0.0, 2.10, 15.90, 5.0, 6.0),
     (0.10, 0.0, 0.0, 0.90, 5.0, 3.90),(6.10, 0.0, 0.0, 6.90, 5.0, 3.90),(12.10, 0.0, 0.0, 12.90, 5.0, 3.90),(18.10, 0.0, 0.0, 18.90, 5.0, 3.90)])
'''
Obstacles = np.array(
    [
     (0.0, 0.0, 0.8, 5.0, 2.5, 1.0),(0.0, 0.0, 0.8, 2.5, 5.0, 1.0),(0.0, 0.0, 2.8, 2.5, 5.0, 3.0),(0.0, 2.5, 2.8, 5.0, 5.0, 3.0),(0.0, 2.5, 4.8, 5.0, 5.0, 5.0),
     (2.5, 0.0, 4.8, 5.0, 5.0, 5.0),(2.5, 0.0, 6.8, 5.0, 5.0, 7.0),(0.0, 0.0, 6.8, 5.0, 2.5, 7.0),(0.0, 0.0, 8.8, 5.0, 2.5, 9.0),(0.0, 0.0, 8.8, 2.5, 5.0, 9.0),
     (0.0, 0.0, 10.8, 2.5, 5.0, 11.0),(0.0, 2.5, 10.8, 5.0, 5.0, 11.0),(0.0, 2.5, 12.8, 5.0, 5.0, 13.0),(2.5, 0.0, 12.8, 5.0, 5.0, 13.0),(2.5, 0.0, 14.8, 5.0, 5.0, 15.0),
     (0.0, 0.0, 14.8, 5.0, 2.5, 15.0),(0.0, 0.0, 16.8, 5.0, 2.5, 17.0),(0.0, 0.0, 16.8, 2.5, 5.0, 17.0),(0.0, 0.0, 18.8, 2.5, 5.0, 19.0),(0.0, 2.5, 18.8, 5.0, 5.0, 19.0)])


x_init = (2.5, 4.0, 0.5)  # starting location
x_goal = (4.0, 2.5, 19.5)  # goal location
Q = np.array([(1, 4)])  # length of tree edges
r = 0.1  # length of smallest edge to check for intersection with obstacles
max_samples = 20480  # max number of samples to take before timing out
prc = 0.1  # probability of checking for a connection to goal

# create Search Space
X = SearchSpace(X_dimensions, Obstacles)

# create rrt_search
rrt = RRT(X, Q, x_init, x_goal, max_samples, r, prc)
path = rrt.rrt_search()

# plot
plot = Plot("rrt_3d")
plot.plot_tree(X, rrt.trees)
if path is not None:
    plot.plot_path(X, path)
plot.plot_obstacles(X, Obstacles)
plot.plot_start(X, x_init)
plot.plot_goal(X, x_goal)
plot.draw(auto_open=True)
