from optparse import Values
import KNN as kn
import numpy as np
import math
import pandas as pd
from matplotlib.colors import ListedColormap
import seaborn as sns
cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])
def accuracy(y_true,y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy
data = pd.read_csv('D:/Python code/KFall_Data.csv')
data = data.sample(frac = 1)
print(data.shape)
print(data.info())
X = np.array(data.drop(['Description','Task ID','Trial ID'],1))
print("Shape of X:",X.shape)
print(X)
y = np.array(data["Description"])
print("Shape of y:",y.shape)
a= pd.crosstab(index = y, columns="col_0")
print(a)
from sklearn import preprocessing
encoder = preprocessing.LabelEncoder()
y = encoder.fit_transform(y)
print(y)
labels = encoder.classes_
s_f = 0.8
n_train = math.floor(s_f * X.shape[0])
n_test = math.ceil((1-s_f) * X.shape[0])
X_train = X[:n_train]
y_train = y[:n_train]
X_test = X[n_train:]
y_test = y[n_train:]
print("Total Number of rows in train:",X_train.shape[0])
print("Total Number of rows in test:",X_test.shape[0])
#Before spliting
print("X:")
print(X)
print("y:")
print(y)
#After Spliting
print("X_train:")
print(X_train)
print("\ny_train:")
print(y_train)
print("\nX_test")
print(X_test)
print("\ny_test")
print(y_test)
clf = kn.KNN(K=3)
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
print("KNN classfication accurarcy",accuracy(y_test,predictions))

def confusion_matrix(actual, predicted):
    classes = np.unique(np.concatenate((actual,predicted)))
    confusion_mtx = np.empty((len(classes),len(classes)),dtype=np.int)
    for i,a in enumerate(classes):
        for j,p in enumerate(classes):
            confusion_mtx[i,j] = np.where((actual==a)*(predicted==p))[0].shape[0]
    return confusion_mtx

print("Confusion matrix:",confusion_matrix(y_test,predictions))
import matplotlib.pyplot as plt
df_cnf_matrix = pd.DataFrame(confusion_matrix(y_test,predictions), index = labels, columns = labels)
sns.heatmap(df_cnf_matrix, annot=True, fmt='d',  cmap=plt.cm.rainbow)
plt.show()