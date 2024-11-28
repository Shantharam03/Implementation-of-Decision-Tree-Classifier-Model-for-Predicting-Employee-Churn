# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas
2. Import Decision tree classifier
3. Fit the data in the model
4. Find the accuracy score 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Shantharam M
RegisterNumber:  24900113
import pandas as pd
data = pd.read_csv("/content/Employee.csv")
data.head()
print(data.head())

data.info()
data.isnull().sum()
data["left"].value_counts()

print(data.head())

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])


x = data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years",
         "salary"]]
print(x.head())


y = data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
print(accuracy)

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(6,8))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)

plt.show()

*/

```

## Output:

![Screenshot 2024-11-28 190403](https://github.com/user-attachments/assets/7f77ba80-96be-4f6a-9934-25f06d4a273d)
![Screenshot 2024-11-28 190501](https://github.com/user-attachments/assets/c717a6ee-5322-429d-8a34-0ac5c126896d)
![Screenshot 2024-11-28 190551](https://github.com/user-attachments/assets/13ebeab8-35ae-4cd2-8f12-58d05457671d)
![Screenshot 2024-11-28 190633](https://github.com/user-attachments/assets/1dea8063-d80b-4121-9240-fc7d4a5db503)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
