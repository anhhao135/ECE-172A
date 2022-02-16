'''
ECE 172A, Homework 2 Robot Traversal
Author: regreer@ucsd.edu
For use by UCSD ECE 172A students only.
'''

import numpy as np
import matplotlib.pyplot as plt

initial_loc = np.array([0,0])
final_loc = np.array([100,100])
sigma = np.array([[50,0],[0,50]])
mu = np.array([[30, 10], [60, 80]])

def f(x, y):
	return ((final_loc[0]-x)**2 + (final_loc[1]-y)**2)/20000 + 10000*(1/(2*np.pi*np.linalg.det(sigma)))*np.exp(-.5*(np.matmul(np.array([x-mu[0,0], y-mu[0,1]]),np.matmul(np.linalg.pinv(sigma), np.atleast_2d(np.array([x-mu[0,0], y-mu[0,1]])).T)))[0]) + 10000*(1/(2*np.pi*np.linalg.det(sigma)))*np.exp(-.5*(np.matmul(np.array([x-mu[1,0], y-mu[1,1]]),np.matmul(np.linalg.pinv(sigma), np.array([x-mu[1,0], y-mu[1,1]])))))

x = np.linspace(0, 100, 100)
y = np.linspace(0, 100, 100)
z = f(x[:,None], y[None,:])
z = np.rot90(np.fliplr(z))


#plot 3d

#fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(x, y, z, 100)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('3D Contour')
#plt.show()

#plot 2d contour with quiver

fig2 = plt.figure()
plt.contour(x,y,z)


dy, dx = np.gradient(z,1,1)
plt.quiver(x,y,dx,dy,width=0.001,headwidth=2) #comment this line out to toggle quiver arrows

step_size = 100
current_loc = initial_loc.astype(float)
close_enough_metric = 0.0001

plan_path = True #change to True to plan path


if plan_path == True:
	try:

		while True:

			dx_ = -dx[(int(current_loc[1])), (int(current_loc[0]))]
			dy_ = -dy[(int(current_loc[1])), (int(current_loc[0]))]
			

			current_loc[0] = current_loc[0] + step_size * dx_
			current_loc[1] = current_loc[1] + step_size * dy_	

			plt.plot(current_loc[0], current_loc[1], marker="*",color="g")

			if np.linalg.norm(np.array([dx_, dy_])) < close_enough_metric:
				break

	except:

		plt.show()


plt.show()