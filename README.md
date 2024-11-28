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
![Screenshot 2024-11-28 225217](https://github.com/user-attachments/assets/cad0c493-d24e-4867-a9a4-7ba41b7f57e9)
![Screenshot 2024-11-28 225227](https://github.com/user-attachments/assets/33fccc50-fde2-494c-a50f-0638b07bc765)
![Screenshot 2024-11-28 225236](https://github.com/user-attachments/assets/f3672a12-78d4-4aaf-9ca9-9ae83ffcdd9c)
![Screenshot 2024-11-28 225242](https://github.com/user-attachments/assets/c6b92b9b-9778-4a90-ab1a-35c75b6d075f)
![Screenshot 2024-11-28 225314](https://github.com/user-attachments/assets/61aa1432-2f6f-4178-bb02-652a9f898a9a)
![Screenshot 2024-11-28 225323](https://github.com/user-attachments/assets/1a9c0061-77bd-4d5c-80de-bd3ec0d58768)
![Screenshot 2024-11-28 225342](https://github.com/user-attachments/assets/91795b8a-3890-4135-8c5d-34b67e746530)
![Screenshot 2024-11-28 225353](https://github.com/user-attachments/assets/1e313bca-8877-445f-bf8c-7ad3c5629e03)
![Screenshot 2024-11-28 225405](https://github.com/user-attachments/assets/2d88efbf-8fa2-428d-8d3f-b047b8562b4f)




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
