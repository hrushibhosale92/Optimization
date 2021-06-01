import numpy as np


# define function for given equation
def f(x):
    return 2*x**2 * np.cos(x) - 5*x

# define derivative of the function to calculate slope/gradient    
def gradient(x):
    return 4*x*np.cos(x) - 2*x**2 * np.sin(x) - 5
    

#stopping criterion
def within_tolerance(x_old,x_new,epsilon):
    return abs(x_new-x_old) <epsilon 

def gradient_descent(x_old,learning_rate,max_iteration,epsilon):
    iteration = 0
    close_enough = False 
    while iteration < max_iteration and not close_enough:

        x_new = x_old - learning_rate * gradient(x_old)
        #taking a smaller step to the direction of the gradients 
        print(iteration,x_new)
        # checking for stopping criterion
        close_enough = within_tolerance(x_old,x_new,epsilon)
        #updating x value with the new value
        x_old = x_new
        iteration = iteration + 1
    print('Minimum od the given function is at x = ',x_new)
    
gradient_descent(5,0.01,1000,1e-7)
#Minimum od the given function is at x =  3.782980483368431
gradient_descent(-3,0.01,1000,1e-7)
#Minimum od the given function is at x =  -3.4831568986663117
gradient_descent(-2.1,0.01,1000,1e-7)
#Minimum od the given function is at x =  -3.483156937889133
gradient_descent(1.5,0.01,1000,1e-7)
#Minimum od the given function is at x =  3.7829802243610486
