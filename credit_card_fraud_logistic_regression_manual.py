import pandas as pd
import tensorflow as tf
import numpy as np
import sklearn as skl
import math 
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore

scaler = StandardScaler()
def load_data(filename): ##Defining the function of loading the dataset into the environment
    data = pd.read_csv(filename, header=None)
    df = data.iloc[1:, :] ## Removing the first row of the dataset which contains the column names, as it is not needed for training the model.
    X = df.iloc[:, :-1].values ##Defining features (Xn)
    y = df.iloc[:, -1].values ##Defining targets (y)
    return X, y

X, y = load_data(r"C:\Users\ernes\OneDrive\Desktop\card_transdata.csv")
X_std = scaler.fit_transform(X)
print("Shape of X_train (Standardized): ", X_std.shape) ##(100,000,001, 7) - prints out the shape of X_train a.k.a features
print("Shape of y_train: ", y.shape) ##(100,000,001, ) - prints out the shape of y_train a.k.a targets

print("Type of X_train: ", type(X))
print("Type of y_train: ", type(y))

X_train, X_test, y_train, y_test = skl.model_selection.train_test_split(X_std, y, test_size = 0.4, random_state = None) ##Splitting the dataset into training and testing sets (ratio 0.6:0.4)

def head(arr, n=10):
    return arr[:n] ##Defining a function to return the first 10 elements of an array

print("The first 10 rows of X_train: ", head(X_train) )
print("The first 10 rows of y_train: ", head(y_train) )
print("Length of training set: ", len(y_train))
print("Length of testing set: ", len(y_test))

## Rudimental print strings to check the data types following
## print(type(X_train))
## print(type(w))
## print(X_train.dtype)
## print(w.dtype)

X_train = X_train.astype(np.float64) ##Converting X_train from object to float
y_train = y_train.astype(np.float64) ##Converting y_train from object to float

##Defining sigmoid function
def sigmoid(z):
    g_z = 1/(1+ np.exp(-z))
    return g_z


def compute_cost(X_train, y_train, w, b, *argv):

    m, n = X_train.shape

    Tcost = 0
    Jwb =0.0
    epsilon = 1e-5
    for i in range(m):
        zi =np.dot(X_train[i], w) + b
        fwb = sigmoid(zi)
        Jwb += -((y_train[i]*np.log(fwb + epsilon))+((1-y_train[i])*np.log(1-fwb + epsilon)))
    Tcost = Jwb/m
    return Tcost 

def compute_cost_reg(X, y, w, b, lambda_ = 1):

    m, n = X_train.shape
    cost = 0.0

    cost_without_reg = compute_cost(X, y, w, b) 
    
    reg_cost = 0.
    
    
    for j in range(n):
        reg_cost += (w[j]**2) 
    reg_cost = (lambda_/(2*m)) * reg_cost      
    total_cost = cost_without_reg + reg_cost

    return total_cost

##Testing cost function
m, n = X_train.shape
initial_w = np.zeros(n)
initial_b = 0.
cost = (compute_cost_reg(X_train, y_train, initial_w, initial_b))

print("Cost at both parameters being equal to zero:  {:.3f}".format(cost))

def gradient(X, y, w, b):

    m, n = X_train.shape

    dw = np.zeros(w.shape)
    db = 0.
    for i in range(m):
        fwb_i = sigmoid(np.dot(X[i], w) + b)
        epsilon = 1*np.exp(1) -15
        fwb = np.clip(fwb_i, epsilon, 1 - epsilon)
        err_i = fwb_i - y[i]
        for j in range(n):
            dw[j] = dw[j] + err_i * X[i, j]
        db = db + err_i
    dw = dw/m
    db = db/m
    return dw, db


def compute_gradient_reg(X, y, w, b, lambda_ = 1): 
    m, n = X_train.shape
    
    db, dw = gradient(X, y, w, b)    
    for j in range(n):
        dw = dw + (lambda_/m) * w             
        
    return db, dw

## Testing gradient descent:
initial_w = np.zeros(n) 
initial_b = 0.

dw, db = gradient(X_train, y_train, initial_w, initial_b)
print(f'dw at initial parameters = null: {dw}')
print(f'db at initial parameters = null: {db}')

def gradient_descent(X, y, w_in, b_in, f_J_wb, gradient_f, a, iter_count, l):
    m = len(X_train)
    J_his = []
    w_his = []
    for i in range(iter_count):
        dj_db, dj_dw = gradient_f(X, y, w_in, b_in, l)   

        # Update Parameters using w, b, alpha and gradient
        w_in = w_in - a * dw               
        b_in = b_in - a * db              
       
        # Save cost J at each iteration
        if i<100000:    
            cost =  f_J_wb(X, y, w_in, b_in, l)
            J_his.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(iter_count/10) == 0 or i == (iter_count-1):
            w_his.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_his[-1]):8.2f}   ")
        
    return w_in, b_in, J_his, w_his

m, n = X_train.shape
initial_w = 0.01*np.random.random(7)
initial_b = 0.1
lambda_ = 0.1

# Some gradient descent settings
iterations = 100000
alpha = 0.0001

w,b, J_history,_ = gradient_descent(X_train ,y_train, initial_w, initial_b, 
                                   compute_cost_reg, compute_gradient_reg, alpha, iterations, lambda_)

print(f"Minimized cost parameters for the model: w: {w}")
print(f"Minimized cost parameters for the model: b: {b}")


## Trained parameters:
##Minimized cost parameters for the model: w:[ 0.57858505  0.30758961  1.28651701  0.09698618 -0.20277999 -0.28138014 0.59232   ]
##Minimized cost parameters for the model: b:-4.036378792940089

def predict(X, w, b): 
    
    m, n = X.shape   
    p = np.zeros(m)
    for i in range(m):   
        z_wb = 0
        for j in range(n): 
            z_wb += w[j] * X[i][j]
        
        z_wb += b
        f_wb = sigmoid(z_wb)
        p[i] = 1 if f_wb > 0.5 else 0
        
    return p

# Testing predict code
tmp_w = np.array
tmp_b = 0.3    
tmp_X = np.random.randn(4, 7) - 0.5

tmp_p = predict(tmp_X, tmp_w, tmp_b)
print(f'Output of predict: shape {tmp_p.shape}, value {tmp_p}')

#Compute accuracy on our training set
p = predict(X_train, w,b)
score = np.mean(p == y_train) * 100
print(f"Train Accuracy: {score} ")

##Plotting the confusion matrix 

cm = metrics.confusion_matrix(y_test, p)
print(cm)

plt.figure(figsize=(2,2))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'rocket');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15)
plt.show()