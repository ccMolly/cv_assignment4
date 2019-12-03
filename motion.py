from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
from matplotlib import pyplot as plt
import numpy as np
image_points = sio.loadmat("sfm_points.mat")['image_points']
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(121)
ax1 = fig.add_subplot(122)

ax.scatter(image_points[0, :, 1], image_points[1, :, 1],
           marker='.', color='blue', s=40, label='class 1')
ax.set_xlabel('variable X')
ax.set_ylabel('variable Y')

ax1.scatter(image_points[0, :, 5], image_points[1, :, 5],
           marker='.', color='blue', s=40, label='class 1')
ax1.set_xlabel('variable X')
ax1.set_ylabel('variable Y')


for i in range(10):
    centroid_x = sum(image_points[0, :, i])/600
    centroid_y = sum(image_points[1, :, i])/600
    if i == 0:
        print("T0:",centroid_x,centroid_y)
    for j in range(600):
        image_points[0, j, i] -= centroid_x
        image_points[1, j, i] -= centroid_y

image_points = np.reshape(image_points,(20,600))


u, s, v = np.linalg.svd(image_points)
#print(len(u),u.size)
#print(s)
#print(len(v),v.size)
s = np.diag(s[0:3])
M = np.dot(u[:,:3],s)
print("M0:",M[:2,:])
v = v.T
world_points = v[:,:3]
print("First 10 points:",world_points[:10,:])

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
#print((world_points[:, 0]))

ax.scatter(world_points[:, 0], world_points[:, 1], world_points[:, 2],

           marker='.', color='blue', s=40, label='class 1')

ax.set_xlabel('variable X')
ax.set_ylabel('variable Y')
ax.set_zlabel('variable Z')

plt.title('3D Scatter Plot')

plt.show()