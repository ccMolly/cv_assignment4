import cv2
import numpy as np
from scipy import linalg
img_points = open("image.txt")
world_points = open("world.txt")
lines = img_points.readlines()
img_xs = lines[0].split('\n')[0].split('  ')[1:]
img_ys = lines[1].split('\n')[0].split('  ')[1:]
lines = world_points.readlines()
world_xs = lines[0].split('\n')[0].split('  ')[1:]
world_ys = lines[1].split('\n')[0].split('  ')[1:]
world_zs = lines[2].split('\n')[0].split('  ')[1:]

A = []
X = []
Y = []
Z = []
cor = [1]*10
for i in range(10):
    X.append(float(world_xs[i]))
    Y.append(float(world_ys[i]))
    Z.append(float(world_zs[i]))
    A.append([float(world_xs[i]), float(world_ys[i]), float(world_zs[i]), 1, 0, 0, 0, 0, -1 * float(img_xs[i]) * float(world_xs[i]), -1 * float(img_xs[i]) * float(world_ys[i]), -1 * float(img_xs[i]) * float(world_zs[i]), -1 * float(img_xs[i])])
    A.append([0, 0, 0, 0, float(world_xs[i]), float(world_ys[i]),  float(world_zs[i]), 1, -1 * float(img_ys[i]) * float(world_xs[i]), -1 * float(img_ys[i]) * float(world_ys[i]), -1 * float(img_ys[i]) * float(world_zs[i]), -1 * float(img_ys[i])])
A = np.matrix(A)
u, s, v = np.linalg.svd(A)
h = np.reshape(v[11], (3, 4))
#print(h)
W = []
W.append(X)
W.append(Y)
W.append(Z)
W.append(cor)
W = np.matrix(W)
proj_img = np.dot(h,W)
#print(proj_img)
x = []
y = []
for i in range(10):
    s = 1/proj_img[2].item(i)
    x.append(proj_img[0].item(i)*s)
    y.append(proj_img[1].item(i)*s)
with open("result.txt","w") as f:
    for i in range(10):
        f.writelines('  '+str(x[i]))
    f.writelines('\n')
    for i in range(10):
        f.writelines('  '+str(y[i]))

K,R = linalg.rq(h[:,:3])
#print(K.size)
#print(R.size)

# make diagonal of K positive
T = np.diag(np.sign(np.diag(K)))
if linalg.det(T) < 0:
    T[1,1] *= -1
#print(T.size)

K = np.dot(K,T)
R = np.dot(T,R) # T is its own inverse
t = np.dot(linalg.inv(K),h[:,3])
#print(K)
#print(R)
#print(t)
c = -np.dot(R.T, t)
print(c)
u, s, v = np.linalg.svd(h)
v = (1/(v[3].item(3)))*v
print(v[3])