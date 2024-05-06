# EX.NO:2 Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
# DATE:
## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:

```python
Program to implement the simple linear regression model for predicting the marks scored.

Developed by:    HARITHASHREE.V
RegisterNumber:  212222230046
```

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error , mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
```

```python
df.tail()
```

```python
X= df.iloc[:,:-1].values
X
```

```python
Y = df.iloc[:,1].values
Y
```

```python
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/3,random_state=0)
```

```python
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
```

```python
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,regressor.predict(X_train),color="black")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```

```python
plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs scores (test set)")
plt.xlabel("Hours")
plt.ylabel("scores")
plt.show()
```

```python
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:

#### Head Values
![image](https://github.com/haritha-venkat/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121285701/747192b6-f483-4d87-b6bd-3028b893f31a)


#### Tail Values
![image](https://github.com/haritha-venkat/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121285701/0852cd81-cf12-48e5-a17f-4d23125ca761)


#### X and Y values
![image](https://github.com/haritha-venkat/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121285701/50187495-67bf-419f-af85-5d5bccea073e)

![image](https://github.com/haritha-venkat/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121285701/ffc0f349-a6bf-45df-a2ae-e05cb44da33b)

####  Prediction Values
![image](https://github.com/haritha-venkat/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121285701/53887a7d-0747-4cc7-a775-826f644376fc)


#### MSE,MAE and RMSE
![image](https://github.com/haritha-venkat/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121285701/d44471f8-0ba7-4ca8-93ea-9329b9eba440)


#### Training Set
![image](https://github.com/haritha-venkat/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121285701/700a63eb-3ff6-4346-a1bc-296e827f6e01)


#### Testing Set
![image](https://github.com/haritha-venkat/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121285701/0ee278cb-3963-4d0f-931b-352550aba760)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
