# define function for given equation
def f(x,y):
    return 1 - (x**2+y**2)

# define derivative of the function to calculate slope/gradient
def compute_gradients(x,y):
    return -2*x,-2*y


#stopping criterion
def within_tolerance(x_old,y_old,x_new,y_new,epsilon):
    return abs(x_new-x_old) <epsilon and abs(y_new-y_old)<epsilon 
        

def grad_des(x_old,y_old,learning_rate,max_iteration,epsilon):
    iteration = 0
    close_enough = False 
    while iteration < max_iteration and not close_enough:
        # find the direction of the slope
        grad_x,grad_y = compute_gradients(x_old,y_old)
       
        #taking a step towrds the direction of gradient
        x_new = x_old + learning_rate * grad_x
        y_new = y_old + learning_rate * grad_y
        
        #checking for termination condition
        close_enough = within_tolerance(x_old,y_old,x_new,y_new,epsilon)
        print(iteration,x_new,y_new)
        x_old,y_old = x_new,y_new
        
        iteration = iteration + 1
    print('minimum of the given eqn is at ', x_new,y_new)
    
grad_des(4,-6,0.1,10000,1e-7)