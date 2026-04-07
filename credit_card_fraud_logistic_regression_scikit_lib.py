import numpy as np
import sklearn as skl
import math
import pandas as pd
from sklearn.linear_model import LogisticRegression
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

LogReg = LogisticRegression()
LogReg.fit(X_train, y_train) 
p = LogReg.predict(X_test) 
score = LogReg.score(X_test, y_test)
print(f"The accuracy score of the logistic regression model is: {score}%")

cm = metrics.confusion_matrix(y_test, p)
print(cm)

plt.figure(figsize=(2,2))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'rocket');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15)
plt.show()