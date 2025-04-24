# Implementation of Logistic Regression Using Gradient Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load dataset and remove irrelevant columns (sl_no, salary).Convert categorical features to numeric codes using .astype('category').cat.codes.
2. Extract input features X and target variable Y.Initialize model parameters (theta) randomly.
3. Define the sigmoid function and loss function.Apply gradient descent to minimize the loss and update theta.
4. Use the learned theta to predict outcomes.Calculate accuracy by comparing predictions to actual labels.Use the model to predict placement for a custom input.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Prajin S
RegisterNumber:  212223230151
*/
```
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('Placement_Data.csv')
df
df=df.drop(['sl_no','salary'],axis=1)
df['gender']=df['gender'].astype('category')
df['ssc_b']=df['ssc_b'].astype('category')
df['hsc_b']=df['hsc_b'].astype('category')
df['hsc_s']=df['hsc_s'].astype('category')
df['degree_t']=df['degree_t'].astype('category')
df['workex']=df['workex'].astype('category')
df['specialisation']=df['specialisation'].astype('category')
df['status']=df['status'].astype('category')
df.dtypes
df['gender']=df['gender'].cat.codes
df['ssc_b']=df['ssc_b'].cat.codes
df['hsc_b']=df['hsc_b'].cat.codes
df['hsc_s']=df['hsc_s'].cat.codes
df['degree_t']=df['degree_t'].cat.codes
df['workex']=df['workex'].cat.codes
df['specialisation']=df['specialisation'].cat.codes
df['status']=df['status'].cat.codes
df
X=df.iloc[:,:-1].values
Y=df.iloc[:,-1].values
Y
theta=np.random.randn(X.shape[1])
y=Y

def sigmoid(z):
    return 1/(1+np.exp(-z))

def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))

def gradient_descent(theta,X,y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(X.dot(theta))
        gradient=X.T.dot(h-y)/m
        theta-=alpha*gradient
    return theta

theta=gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)

def predict(theta,X):
    h=sigmoid(X.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred

y_pred=predict(theta,X)

acc=np.mean(y_pred.flatten()==y)
print("Accuracy=",acc)
print(y_pred)
print(Y)

xn=np.array([0,87,0,95,0,2,78,2,0,0,1,0])
ypr=predict(theta,xn)
ypr
```


## Output:
![image](https://github.com/user-attachments/assets/e36f167f-939b-4135-be46-288c9ab817d8)

![image](https://github.com/user-attachments/assets/88c6f037-2556-4f0a-847b-7afb819c3623)


![image](https://github.com/user-attachments/assets/d5f446c3-4baf-4f2d-a2da-b06227e3cfe4)

![image](https://github.com/user-attachments/assets/350a2d1b-df23-491a-aa56-8c0b07655374)

![image](https://github.com/user-attachments/assets/ba55073c-90d0-4d1b-b329-132e8a06d9a0)

![image](https://github.com/user-attachments/assets/cfc80c17-14d1-46b7-b583-f16595a0df99)

![image](https://github.com/user-attachments/assets/cf9e4e43-1356-4c79-979f-7e54ff1b0ad2)

![image](https://github.com/user-attachments/assets/670f3710-e0a9-4eb2-9b80-11b6a5e6e4eb)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

