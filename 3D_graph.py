#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 21:22:31 2019

@author: theone
"""

#from mpl_toolkits import mplot3d
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from sys import argv


fig = plt.figure()
ax = plt.axes(projection="3d")
X2=0.001, 0.001, 0.001, 0.001, 0.01, 0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 0.1, 1, 1, 1, 1
X= np.log10(X2) 
Y=100, 500, 1000, 5000, 100, 500, 1000, 5000, 100, 500, 1000, 5000, 100, 500, 1000, 5000
Z=0.74269, 0.74076, 0.74149, 0.74149, 0.71832, 0.71832, 0.71832, 0.71832, 0.635047, 0.634805, 0.635529, 0.634564, 0.634081, 0.632391, 0.630702, 0.631426
ax.scatter3D(X, Y, Z, c=Z, cmap='jet');
ax.set_xlabel('log of learning rate')
ax.set_ylabel('number of interation')
ax.set_zlabel('accuracy')
plt.savefig('teste.pdf')
plt.show()


"""
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_trisurf(X, Y, Z, cmap=cm.jet, linewidth=0.1)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
"""

z_points = 15 * np.random.random(100)
x_points = np.cos(z_points) + 0.1 * np.random.randn(100)
y_points = np.sin(z_points) + 0.1 * np.random.randn(100)

ax.scatter3D(x_points, y_points, z_points, c=z_points, cmap='jet');

#plt.show()