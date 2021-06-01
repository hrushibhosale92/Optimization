
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

def f(x,y):
    return x**2 + y**2

def df(cur_pos):
    return [2*cur_pos[0],2*cur_pos[1]]


def compute_step(cur_pos):
    grad = df(cur_pos)
    x = cur_pos[0] - grad[0] * learning_rate 
    y = cur_pos[1] - grad[1] * learning_rate
    return (x,y)

cur_pos = (-8,9)#tuple(np.random.randint(-10,10,2))
next_pos = (-10,5)
learning_rate = 0.05
iters = 0
eps = 1e-10
max_iter = 500
cor_d = []
cor_d.append(cur_pos)
cur_val = f(cur_pos[0],cur_pos[1])
zz = []
zz.append(cur_val)
print(cur_pos)
for i in range(500):
    next_pos = compute_step(cur_pos)
    cor_d.append(next_pos)
    
    #print(cor_d[-1])
    cur_pos = next_pos
    new_val = f(next_pos[0],next_pos[1])
    zz.append(new_val)
    if abs(cur_val-new_val) < eps:
        break
    cur_pos = next_pos
    cur_val = f(cur_pos[0],cur_pos[1])
    
    


print("minima of the given equation ia at : ",next_pos)


xx = [cor_d[i][0] for i in range(len(cor_d))]
yy = [cor_d[i][1] for i in range(len(cor_d))]





def fig3(ax,a,b,c):
    #fig = plt.figure()
    plt.cla()

    x = np.linspace(-10, 10, 30)
    y = np.linspace(-10, 10, 30)

    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    ax.contour3D(X, Y, Z,18)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    #cset = ax.contour(X, Y, Z, 18,zdir='z', offset=-10)
    
    ax.scatter3D(a,b,c, c = 'r');
    plt.pause(0.5)
    

ax = plt.axes(projection="3d")
for i in range(len(xx)):
    
    fig3(ax,xx[:i],yy[:i],zz[:i])
    


