import pandas as pd
import tensorflow as tf
import numpy as np
import sklearn as skl
import math 
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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

model = Sequential([
    tf.keras.Input(shape=(X_train.shape[1],)),
    Dense(units=1, activation="sigmoid", name = "layer1")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss=tf.keras.losses.BinaryCrossentropy()
)
model.fit(X_train,y_train, epochs = 30)

W1, b1 = model.get_layer("layer1").get_weights()
print("W1:\n", W1, "\nb1:", b1)

predictions = model.predict(X_test)
print("predictions = \n", predictions)

yhat = np.zeros_like(predictions)
for i in range(len(predictions)):
    if predictions[i] >= 0.5:
        yhat[i] = 1
    else:
        yhat[i] = 0
print(f"decisions = \n{yhat}")