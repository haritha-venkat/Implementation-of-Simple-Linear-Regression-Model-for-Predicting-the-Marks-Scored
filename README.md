# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import the required libraries and read the dataframe.

2. Assign hours to X and scores to Y.

3. Implement training set and test set of the dataframe

4. Plot the required graph both for test data and training data.

5. Find the values of MSE , MAE and RMSE


## Program and Output:

Program to implement the simple linear regression model for predicting the marks scored.

Developed by: HARITHASHREE.V

RegisterNumber:  212222230046



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
```
![image](https://github.com/haritha-venkat/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121285701/7aaa373d-28a8-4456-a75a-7aaa561c07a3)

```python
df.tail()
```
![image](https://github.com/haritha-venkat/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121285701/c942638b-9aee-4d04-a895-aa144e090da6)

```python
X=df.iloc[:,:-1].values
X
```
![image](https://github.com/haritha-venkat/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121285701/a1790825-459a-4e4b-90e5-d2f9142b83f8)

```python
Y=df.iloc[:,:-1].values
Y
```
![image](https://github.com/haritha-venkat/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121285701/d690517f-cbe6-4c64-988e-8ee7e35188df)

```python
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
```
![image](https://github.com/haritha-venkat/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121285701/42309c8d-81d1-4b4c-ba0f-6358ae56765f)

```python
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,regressor.predict(X_train),color="black")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
![image](https://github.com/haritha-venkat/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121285701/089401f8-fa8e-4965-a151-b4d178338159)

```python
plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs scores (test set)")
plt.xlabel("Hours")
plt.ylabel("scores")
plt.show()
```
![image](https://github.com/haritha-venkat/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121285701/11c4399a-04f9-4f40-895e-87764bb73892)

```python
mse=mean_squared_error(Y_test,Y_pred)
print("MSE = ",mse)
mae=mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```
![image](https://github.com/haritha-venkat/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121285701/12bbfeb4-1f1c-404b-bf9a-a3eeec59a6a7)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
