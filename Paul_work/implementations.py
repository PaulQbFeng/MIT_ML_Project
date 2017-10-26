import numpy as np 
import math
from helpers import batch_iter

def compute_mse(y, tx, w):
    
    N = len(y)
    e = y - tx.dot(w)
    
    return 1/N*np.transpose(e).dot(e)


def build_poly(x, degree):

    phi = np.ones((len(x), x.shape[1]))
    for deg in range(1, degree+1):
        phi = np.c_[phi, np.power(x, deg)]
    return phi


def compute_gradient(y, tx, w):
    
    N = len(y)
    e = y - tx.dot(w)
    
    return -1/N*np.transpose(tx).dot(e)


def least_squares_GD(y, tx, initial_w, max_iter, gamma):

    # Define parameters to store w and loss

    w = initial_w
    
    for n_iter in range(max_iter):
       
        grad = compute_gradient(y,tx,w)
        loss = compute_mse(y,tx,w)
        
        w = w - gamma*grad
        
        if n_iter %100 == 0: 
            print("Current iteration={i}, loss={l}".format(i=n_iter, l=loss))
     
    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):


    batch_size = 1

    # Define parameters to store w and loss
    loss = 0
    w = initial_w

    for n_iter, [minib_y, minib_tx] in enumerate(batch_iter(y, tx, batch_size, max_iters)):
        
        grad = compute_gradient(minib_y, minib_tx, w)
        loss = compute_mse(y, tx, w)

        if n_iter %100 == 0: 
            print("Current iteration={i}, loss={l}".format(i=n_iter, l=loss))
     
        w = w - gamma * grad

    return w, loss


def least_squares(y, tx):

    A = np.transpose(tx).dot(tx)
    B = np.transpose(tx).dot(y)
    
    w = np.linalg.solve(A,B)
    
    loss = compute_mse(y,tx,w)
    
    return w, loss


def ridge_regression(y, tx, lambda_):
    
    N = len(y)
  
    a = lambda_*2*N*np.eye(tx.shape[1])
    A = np.transpose(tx).dot(tx) + a 
    B = np.transpose(tx).dot(y)
    
    w = np.linalg.solve(A,B)
    
    loss =  compute_mse(y,tx,w) + lambda_*np.transpose(w).dot(w)
    
    return w, loss


def sigmoid(t):

    return 1/(1 + np.exp(-t))
    #return 1/(np.exp(np.logaddexp(0,-t)))
    
def calculate_loss(y, tx, w):

    #sum_m = np.log(1+np.exp(tx.dot(w))) - y*tx.dot(w)
    sum_m = np.logaddexp(0,tx.dot(w)) - y*tx.dot(w) 
    
    return sum_m.sum()
    
def calculate_gradient(y, tx, w):

    return np.transpose(tx).dot(sigmoid(tx.dot(w))-y)

def learning_by_gradient_descent(y, tx, w, gamma):
    
    loss = calculate_loss(y,tx,w)
    grad = calculate_gradient(y,tx,w)
    
    w = w - gamma*grad 
    
    return loss, w

def calculate_hessian(y, tx, w):
    
    print("calcul_hessian_enter")
    
    N = y.shape[0]
    
    a = np.eye(N)
    S = sigmoid(tx.dot(w))*(1-sigmoid(tx.dot(w)))

    hess = np.transpose(tx).dot(a*S).dot(tx)
    
    return hess

def logistic_regression_newton(y, tx, w):
    
    loss = calculate_loss(y,tx,w)
    print("loss ok")
    gradient = calculate_gradient(y,tx,w)
    print("grad ok")
    hessian = calculate_hessian(y,tx,w)
    print("hess ok")
    
    return loss, gradient, hessian 

def penalized_logistic_regression(y, tx, w, lambda_):

    loss = calculate_loss(y,tx,w) + lambda_/2*np.transpose(w).dot(w)
    gradient = calculate_gradient(y,tx,w) + lambda_*w 
    #hessian = calculate_hessian(y,tx,w) + lambda_*np.eye(tx.shape[1],dtype = int)
    
    return loss,gradient

def learning_by_newton_method(y, tx, w):

    loss, gradient, hessian = logistic_regression_newton(y,tx,w)
    
    print("log regre newton ok")
    
    w = w - gamma*np.linalg.inv(hessian).dot(gradient)
    
    print("learning ok")
    return loss, w

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
   
    loss, gradient = penalized_logistic_regression(y,tx,w,lambda_)   

    w = w - gamma*gradient
    
    return loss, w

def logistic_regression(y, tx, w, max_iter, threshold, gamma, losses):
    
    for iter in range(max_iter):
        
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w,gamma)
        
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
            
        losses.append(loss)
        #stop condition
        if (len(losses) > 1) and (np.abs(losses[-1] - losses[-2])) < threshold:
            break
    
    return w, losses 

def logistic_regression_newton_descent(y, tx, w, max_iter, threshold, gamma, losses):
    
    for iter in range(max_iter):
        
        loss, w = learning_by_newton_method(y, tx, w)
        
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
            print(w[:,0])
    
        losses.append(loss)
        #stop condition
        if (len(losses) > 1) and (np.abs(losses[-1] - losses[-2])) < threshold:
            break
        
    return w, losses 

def logistic_regression_penalized(y, tx, w, max_iter, threshold, gamma, losses,lambda_):
    
    start_time = datetime.datetime.now()
    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma,lambda_)

        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
            print(w[:,0])
    
        losses.append(loss)
        #stop condition
        if (len(losses) > 1) and (np.abs(losses[-1] - losses[-2])) < threshold:
            break

    return w, losses 

def logistic_regression_test_gamma(y, tx, w, max_iter,gammas, rmse_gamma, weight_gamma):
    
    for gamma in gammas: 
        print("Current gamma={i}".format(i=gamma))
        
        weight = []
        losses = []
        
        w = np.zeros((tx.shape[1], 1))
        
        for iter in range(max_iter):
        
            loss, w = learning_by_gradient_descent(y, tx, w, gamma)
            
            if iter % 10 == 0:
            
                losses.append(loss)
                weight.append(w)
                
                
            if iter % 1000 == 0:
                print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
                
            
    
        indexMin = losses.index(min(losses))
        
        weight_gamma.append(weight[indexMin])
    
        rmse_gamma.append(math.sqrt(2*losses[indexMin]))
        
        print(rmse_gamma)
        print(weight_gamma)
    
    
    return weight_gamma, rmse_gamma 
        
   