import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return np.cos(14.5*x-0.3)+ x * (x +0.2) + 0.1


x_old = -2.2
E1 = f(x_old)

for i in range(1000):
    h = np.random.uniform(-0.05,0.05)
    x_new = x_old + h
    E2 = f(x_new)
    if (E2 - E1) < 0:
        x_old = x_new
        E1 = E2
    

print("minima for the function is :",x_old)


a,b = -3,3
x = np.linspace(a,b,100)
y = f(x)


plt.plot(x,y,color='k')
plt.scatter(-2.2,f(-2.2),label='start point')
plt.scatter(x_old,f(x_old),label='end point')
plt.xlabel("x")
plt.ylabel('f(x)')
plt.legend(loc='upper center')
plt.grid()
plt.show()


