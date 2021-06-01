
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2 

x_old = -2
E1 = f(x_old)

x_s = []
E_s = []
x_s.append(x_old)
E_s.append(E1)
for i in range(50):
    h = np.random.uniform(-.5,.5)
    x_new = x_old + h
    E2 = f(x_new)
    
    x_s.append(x_new)
    E_s.append(E2)
    
    x_old = x_new 
        
x_min = x_s[np.argmin(E_s)]   

start_point = -6
end_point = 6

x = np.linspace(start_point,end_point,100)
y = f(x)
plt.plot(x,y)
plt.show()

for i in range(len(x_s)-1):
    plt.plot(x,y)
    plt.plot(x_s[i:i+2],E_s[i:i+2])
    plt.scatter(x_s[i:i+2],E_s[i:i+2])
    plt.grid()
    plt.pause(0.5)
